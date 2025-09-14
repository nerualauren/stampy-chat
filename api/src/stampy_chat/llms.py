from typing import TypedDict, Literal, Generator, Sequence, Any, Callable, Optional
import re

import anthropic
import openai
from google import genai
from stampy_chat.settings import ANTHROPIC, OPENAI, GOOGLE, OPENROUTER, MODELS, Settings
from stampy_chat.env import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY
from stampy_chat.citations import Message, retrieve_docs
from stampy_chat.prompts import format_blocks


class LLMChunk(TypedDict):
    type: Literal["thinking", "response"]
    text: str


class Tool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


# Built-in tool: retrieve_docs
RETRIEVE_DOCS_TOOL = Tool(
    name="retrieve_docs",
    description="Retrieve relevant documents from the knowledge base. Use this when you need to find information about AI safety, alignment, or related topics to answer user questions accurately.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents. Should be a clear, specific question or topic."
            }
        },
        "required": ["query"]
    }
)


def execute_tool(tool_name: str, tool_input: dict[str, Any], settings: Settings, conversation_context=None, callbacks=None) -> str:
    """Execute a tool function and return the result as a string."""
    if tool_name == "retrieve_docs":
        query = tool_input.get("query", "")
        blocks = retrieve_docs(query, settings)

        # Update citation indexes with offset for monotonic numbering
        if conversation_context:
            for block in blocks:
                if "reference" in block:
                    # Convert reference to integer, add offset, convert back
                    try:
                        ref_num = int(block["reference"])
                        block["reference"] = str(ref_num + conversation_context.citation_id_offset)
                    except (ValueError, TypeError):
                        # If reference isn't numeric, leave as is
                        pass

            # Update the offset for future tool calls
            conversation_context.citation_id_offset += len(blocks)
            conversation_context.accumulated_citations.extend(blocks)

            # Notify callbacks about accumulated citations
            if callbacks:
                for callback in callbacks:
                    callback.on_citations_accumulated(blocks)

        return format_blocks(blocks)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def split_system(history: Sequence[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join([x["content"] for x in history if x["role"] == "system"])
    history = [x for x in history if x["role"] != "system"]
    return system, history


class ThinkingState:
    """Manages custom thinking block parsing state"""
    def __init__(self):
        self.accumulated_text = "<thinking>"
        self.thinking_sent = 0
        self.in_thinking_block = True
        self.thinking_end_pattern = re.compile(r'(.*?)</thinking>(.*)', re.DOTALL)


class StreamState:
    """Manages streaming response reconstruction state"""
    def __init__(self):
        self.message_content = []
        self.stop_reason = None
        self.tool_uses = []
        self.current_text = ""
        self.current_tool_use = None
        self.current_tool_json = ""


class ResponseAccumulator:
    """Accumulates response across multiple model calls for tool use with thinking"""
    def __init__(self, custom_thinking: bool):
        self.parts = []
        self.thinking_state = ThinkingState() if custom_thinking else None
        self.thinking_finished = False

    def get_accumulated_content(self) -> str:
        """Get the current accumulated response content for continuation"""
        if self.thinking_state and not self.thinking_finished:
            return self.thinking_state.accumulated_text
        return "".join(self.parts)


def handle_thinking_delta(thinking_state: ThinkingState, delta_text: str) -> Generator[LLMChunk, None, None]:
    """Process custom thinking block text and yield thinking/response chunks"""
    thinking_state.accumulated_text += delta_text

    if thinking_state.in_thinking_block:
        # Check if we've reached the end of thinking
        match = thinking_state.thinking_end_pattern.search(thinking_state.accumulated_text)
        if match:
            # Found </thinking> - extract thinking content and any response after
            thinking_content = match.group(1)
            response_content = match.group(2)

            # Extract just the thinking part (excluding the <thinking> tag)
            thinking_text = thinking_content[len("<thinking>"):]

            # Send any remaining thinking content we haven't sent yet
            remaining_thinking = thinking_text[thinking_state.thinking_sent:]
            if remaining_thinking:
                yield LLMChunk(type="thinking", text=remaining_thinking)

            # Switch to response mode
            thinking_state.in_thinking_block = False

            # Send any response content that came after </thinking>
            if response_content:
                yield LLMChunk(type="response", text=response_content)
        else:
            # Still in thinking block, send new thinking content
            thinking_text = thinking_state.accumulated_text[len("<thinking>"):]

            # Check if current accumulated text might contain a partial "</thinking>" at the end
            # Be conservative and only send content that we're sure won't be part of the closing tag
            safe_thinking_text = thinking_text
            for partial in ["</think", "</thin", "</thi", "</th", "</t", "</"]:
                if thinking_text.endswith(partial):
                    safe_thinking_text = thinking_text[:-len(partial)]
                    break

            new_thinking = safe_thinking_text[thinking_state.thinking_sent:]
            if new_thinking:
                yield LLMChunk(type="thinking", text=new_thinking)
                thinking_state.thinking_sent = len(safe_thinking_text)
    else:
        # In response mode, just send as response
        yield LLMChunk(type="response", text=delta_text)


def _update_response_accumulator(accumulator: ResponseAccumulator, message_content: list) -> None:
    """Update response accumulator with new message content"""
    if accumulator.thinking_state and not accumulator.thinking_state.in_thinking_block:
        accumulator.thinking_finished = True
        # Extract response content from message_content
        for content in message_content:
            if content["type"] == "text":
                accumulator.parts.append(content["text"])


def _extract_tool_uses_nonstreaming(response) -> tuple[list, list]:
    """Extract tool uses from non-streaming response"""
    tool_uses = []
    response_content = []

    for content_block in response.content:
        if content_block.type == 'text':
            response_content.append({"type": "text", "text": content_block.text})
        elif content_block.type == 'tool_use':
            tool_uses.append(content_block)
            response_content.append({
                "type": "tool_use",
                "id": content_block.id,
                "name": content_block.name,
                "input": content_block.input
            })

    return tool_uses, response_content


def _execute_tool_uses_nonstreaming(tool_uses: list, settings: Optional[Settings], conversation_context, callbacks) -> list:
    """Execute tool uses for non-streaming mode"""
    tool_results = []
    for tool_use in tool_uses:
        if settings is None:
            settings = Settings()
        result = execute_tool(tool_use.name, tool_use.input, settings, conversation_context, callbacks)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": result
        })
    return tool_results


def process_stream_events(
    stream_response,
    response_accumulator: Optional[ResponseAccumulator],
) -> Generator[LLMChunk, None, tuple[list, str, list]]:
    """Process streaming events and yield chunks, returning final state"""
    stream_state = StreamState()

    with stream_response as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    stream_state.current_text = ""
                elif event.content_block.type == "tool_use":
                    stream_state.current_tool_use = {
                        "type": "tool_use",
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": {}
                    }
                    stream_state.current_tool_json = ""

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    stream_state.current_text += event.delta.text

                    if response_accumulator and response_accumulator.thinking_state:
                        yield from handle_thinking_delta(response_accumulator.thinking_state, event.delta.text)
                    else:
                        yield LLMChunk(type="response", text=event.delta.text)

                elif event.delta.type == "thinking_delta":
                    yield LLMChunk(type="thinking", text=event.delta.thinking)
                elif event.delta.type == "input_json_delta":
                    stream_state.current_tool_json += event.delta.partial_json

            elif event.type == "content_block_stop":
                if stream_state.current_text:
                    stream_state.message_content.append({"type": "text", "text": stream_state.current_text})
                    stream_state.current_text = ""
                elif stream_state.current_tool_use:
                    # Parse the complete JSON input
                    import json
                    try:
                        stream_state.current_tool_use["input"] = json.loads(stream_state.current_tool_json)
                    except json.JSONDecodeError:
                        stream_state.current_tool_use["input"] = {}

                    stream_state.message_content.append(stream_state.current_tool_use)
                    stream_state.tool_uses.append(stream_state.current_tool_use)
                    stream_state.current_tool_use = None
                    stream_state.current_tool_json = ""

            elif event.type == "message_delta":
                if hasattr(event.delta, 'stop_reason'):
                    stream_state.stop_reason = event.delta.stop_reason

    return stream_state.message_content, stream_state.stop_reason, stream_state.tool_uses


def execute_tool_calls(
    tool_uses: list,
    settings: Settings,
    conversation_context,
    callbacks
) -> list:
    """Execute tool calls and return tool results"""
    tool_results = []
    for tool_use in tool_uses:
        if settings is None:
            settings = Settings()
        result = execute_tool(tool_use["name"], tool_use["input"], settings, conversation_context, callbacks)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": result
        })
    return tool_results


def create_anthropic_request(
    client,
    current_history: list[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int,
    custom_thinking: bool,
    tools: Optional[list[Tool]],
    stream: bool,
    response_accumulator: Optional[ResponseAccumulator] = None
):
    """Create and execute Anthropic API request"""
    params = {}
    if thinking_budget > 0 and not custom_thinking:
        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    if tools:
        params["tools"] = [{"name": tool["name"], "description": tool["description"], "input_schema": tool["input_schema"]} for tool in tools]

    system, msg_history = split_system(current_history)
    msg_history = list(msg_history)

    # Add preloaded assistant message for custom thinking
    # Only add this if there are no assistant messages already (i.e., first call)
    if custom_thinking:
        # Check if the last message in history is from assistant (indicating previous tool use)
        has_assistant_messages = any(msg["role"] == "assistant" for msg in msg_history)
        if not has_assistant_messages:
            accumulated_content = response_accumulator.get_accumulated_content() if response_accumulator else "<thinking>"
            msg_history.append({"role": "assistant", "content": accumulated_content})
        else:
            pass  # Let model continue from tool results

    try:
        return client.messages.create(
            model=model,
            messages=msg_history,
            system=system,
            max_tokens=max_tokens,
            stream=stream,
            **params,
        )
    except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
        print("WARNING: falling back to google due to anthropic api error:", e)
        raise  # Re-raise to trigger fallback


def call_anthropic(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = True,
    tools: Optional[list[Tool]] = None,
    settings: Optional[Settings] = None,
    conversation_context=None,
    callbacks=None,
    custom_thinking: bool = False,
) -> Generator[LLMChunk, None, None]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    current_history = list(history)

    # Initialize response accumulator for custom thinking across tool calls
    response_accumulator = ResponseAccumulator(custom_thinking) if custom_thinking else None

    try:
        iteration_count = 0
        while True:
            iteration_count += 1
            try:
                response = create_anthropic_request(
                    client, current_history, model, max_tokens, thinking_budget,
                    custom_thinking, tools, stream, response_accumulator
                )
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                print("WARNING: falling back to google due to anthropic api error:", e)
                return call_google(current_history, model, max_tokens, thinking_budget, stream, tools, settings, conversation_context, callbacks)

            if stream:
                message_content, stop_reason, tool_uses = yield from process_stream_events(
                    response, response_accumulator
                )

                # Update accumulator with new content
                if response_accumulator:
                    _update_response_accumulator(response_accumulator, message_content)

                if stop_reason == 'tool_use' and tool_uses:
                    # Execute tools and add results to history normally
                    current_history.append({"role": "assistant", "content": message_content})
                    tool_results = execute_tool_calls(tool_uses, settings, conversation_context, callbacks)
                    current_history.append({"role": "user", "content": tool_results})

                    # Safety check for infinite loops
                    if iteration_count > 10:
                        print("ERROR: Too many iterations, breaking to prevent infinite loop")
                        break
                    continue
                else:
                    return
            else:
                # Non-streaming mode
                if hasattr(response, 'stop_reason') and response.stop_reason == 'tool_use':
                    tool_uses, response_content = _extract_tool_uses_nonstreaming(response)

                    # Process text content through thinking logic if needed
                    for content_block in response.content:
                        if content_block.type == 'text':
                            text = content_block.text

                            if response_accumulator and response_accumulator.thinking_state:
                                for chunk in handle_thinking_delta(response_accumulator.thinking_state, text):
                                    yield chunk
                            else:
                                yield LLMChunk(type="response", text=text)

                    # Normal tool execution - add to history
                    current_history.append({"role": "assistant", "content": response_content})
                    tool_results = _execute_tool_uses_nonstreaming(tool_uses, settings, conversation_context, callbacks)
                    current_history.append({"role": "user", "content": tool_results})

                    # Safety check for infinite loops
                    if iteration_count > 10:
                        print("ERROR: Too many iterations, breaking to prevent infinite loop")
                        break
                    continue
                else:
                    for content_block in response.content:
                        if content_block.type == 'text':
                            text = content_block.text

                            if response_accumulator and response_accumulator.thinking_state:
                                for chunk in handle_thinking_delta(response_accumulator.thinking_state, text):
                                    yield chunk
                            else:
                                yield LLMChunk(type="response", text=text)
                    return
    except Exception as e:
        print(f"Error in call_anthropic: {e}")
        raise


def anthropic_stream(response):
    for event in response:
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                yield LLMChunk(type="thinking", text=event.delta.thinking)
            elif event.delta.type == "text_delta":
                yield LLMChunk(type="response", text=event.delta.text)


def call_openai(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = False,
    tools: Optional[list[Tool]] = None,
    settings: Optional[Settings] = None,
    conversation_context=None,
    callbacks=None,
) -> Generator[LLMChunk, None, None]:
    if tools:
        raise NotImplementedError("Tool use is not yet implemented for OpenAI provider")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    system, history = split_system(history)
    params = {}
    if thinking_budget > 0:
        params["reasoning"] = {"effort": "medium"}

    response = client.responses.create(
        model=model,
        instructions=system,
        input=history,
        max_output_tokens=max_tokens,
        stream=True,
        **params,
    )
    if stream:
        return openai_stream(response)
    else:
        return response.choices[0].message.content


def openai_stream(response):
    for event in response:
        if event.type == "response.output_text.delta":
            yield LLMChunk(type="response", text=event.delta)


def call_google(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = False,
    tools: Optional[list[Tool]] = None,
    settings: Optional[Settings] = None,
    conversation_context=None,
    callbacks=None,
) -> Generator[LLMChunk, None, None]:
    if tools:
        raise NotImplementedError("Tool use is not yet implemented for Google provider")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    system, history = split_system(history)

    # Convert to Gemini's Content format
    contents = []
    for msg in history:
        # Map assistant to model for Gemini
        role = "model" if msg["role"] == "assistant" else msg["role"]
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Build config parameters
    config_params = {
        "max_output_tokens": max_tokens,
        "thinking_config": genai.types.ThinkingConfig(thinking_budget=thinking_budget),
    }

    # Add system instruction if provided
    if system:
        config_params["system_instruction"] = system

    config = genai.types.GenerateContentConfig(**config_params)

    # Use streaming API
    if stream:
        response = client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        def do_stream():
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield LLMChunk(type="response", text=chunk.text)

        return do_stream()
    else:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )

        return response.text


def call_openrouter(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = True,
    tools: Optional[list[Tool]] = None,
    settings: Optional[Settings] = None,
    conversation_context=None,
    callbacks=None,
) -> Generator[LLMChunk, None, None]:
    if tools:
        raise NotImplementedError("Tool use is not yet implemented for OpenRouter provider")

    # Remove "openrouter/" prefix to get the actual model name
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    system, history = split_system(history)
    
    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    
    # Build parameters
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    # Add reasoning/thinking support for compatible models  
    extra_body = {}
    if thinking_budget > 0:
        # Use exact token count for more precise control (minimum 1024 per OpenRouter docs)
        extra_body["reasoning"] = {"max_tokens": thinking_budget}
        
    # Optional headers for attribution
    extra_headers = {}
    
    try:
        if extra_headers:
            params["extra_headers"] = extra_headers
        if extra_body:
            params["extra_body"] = extra_body
        
        response = client.chat.completions.create(**params)
        
        if stream:
            return openrouter_stream(response)
        else:
            return response.choices[0].message.content
    except Exception as e:
        print(f"WARNING: OpenRouter API error: {e}")
        # Could implement fallback to other providers like Anthropic does
        raise


def openrouter_stream(response):
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            # Handle reasoning tokens if present
            if hasattr(delta, 'reasoning') and delta.reasoning:
                yield LLMChunk(type="thinking", text=delta.reasoning)
            # Handle regular content
            if delta.content:
                yield LLMChunk(type="response", text=delta.content)


def query_llm(
    history: Sequence[Message],
    settings: Settings,
    stream: bool = True,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
    tools: Optional[list[Tool]] = None,
    conversation_context=None,
    callbacks=None,
    custom_thinking: bool = False,
) -> Generator[LLMChunk, None, None]:
    provider = settings.model_provider
    if provider == ANTHROPIC:
        func = call_anthropic
    elif provider == OPENAI:
        func = call_openai
    elif provider == GOOGLE:
        func = call_google
    elif provider == OPENROUTER:
        func = call_openrouter
    else:
        raise ValueError(f"Unknown provider: {provider}")

    model_info = MODELS[settings.model]
    thinking_budget = thinking_budget if thinking_budget is not None else settings.thinking_budget
    if model_info.can_think == "always" or (model_info.can_think and thinking_budget > 0):
        thinking_budget = max(model_info.min_think, thinking_budget)
    else:
        thinking_budget = 0

    # Disable thinking for tool use mode to avoid API format conflicts
    if tools:
        thinking_budget = 0

    if provider == ANTHROPIC:
        # Temporarily disable custom thinking for tool use to test basic functionality
        use_custom_thinking = custom_thinking and tools is None
        return func(
            history,
            settings.model_id,
            max_tokens if max_tokens is not None else settings.max_response_tokens,
            thinking_budget,
            stream=stream,
            tools=tools,
            settings=settings,
            conversation_context=conversation_context,
            callbacks=callbacks,
            custom_thinking=use_custom_thinking,
        )
    else:
        return func(
            history,
            settings.model_id,
            max_tokens if max_tokens is not None else settings.max_response_tokens,
            thinking_budget,
            stream=stream,
            tools=tools,
            settings=settings,
            conversation_context=conversation_context,
            callbacks=callbacks,
        )
