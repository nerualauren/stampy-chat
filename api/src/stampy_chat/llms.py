from typing import TypedDict, Literal, Generator, Sequence, Any, Callable, Optional
import json

import anthropic
import openai
from google import genai
from stampy_chat.settings import ANTHROPIC, OPENAI, GOOGLE, Settings
from stampy_chat.env import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from stampy_chat.citations import Message


class LLMChunk(TypedDict):
    type: Literal["thinking", "response"]
    text: str


class Tool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool and return its result as a string.

    This is a simple executor for built-in test tools.
    In production, this would dispatch to actual tool implementations.
    """
    if tool_name == "calculator":
        operation = tool_input.get("operation")
        a = tool_input.get("a")
        b = tool_input.get("b")

        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        else:
            return f"Unknown operation: {operation}"

        return json.dumps({"result": result})
    else:
        return f"Unknown tool: {tool_name}"


def can_think_anthropic(model: str, thinking_budget: int) -> bool:
    return (
        model.startswith("claude-sonnet-3.7")
        or model.startswith("claude-sonnet-4")
        or model.startswith("claude-opus-4")
    ) and thinking_budget >= 1024


def can_think_openai(model: str, thinking_budget: int) -> bool:
    return not (model.startswith("gpt-4o") or model.startswith("gpt-4.1-")) and bool(
        thinking_budget
    )


def split_system(history: Sequence[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join([x["content"] for x in history if x["role"] == "system"])
    history = [x for x in history if x["role"] != "system"]
    return system, history


def call_anthropic(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = True,
    tools: Optional[list[Tool]] = None,
):
    """Call Anthropic API with support for streaming, thinking, and tool use.

    Returns:
    - For non-streaming without tools: str
    - For streaming or with tools: Generator[LLMChunk, None, None]

    When tools are provided and the model wants to use them, this function
    will automatically handle the tool execution loop:
    1. Stream thinking/response chunks
    2. If stop_reason is "tool_use", execute the tool(s)
    3. Send tool results back to the API
    4. Continue streaming until stop_reason is not "tool_use"
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system, messages = split_system(history)
    messages = list(messages)  # Make mutable copy

    # Build base params
    params = {}
    if can_think_anthropic(model, thinking_budget):
        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if tools:
        params["tools"] = tools

    # Non-streaming without tools: simple case, return string
    if not stream and not tools:
        try:
            response = client.messages.create(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                stream=False,
                **params,
            )
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            print("WARNING: falling back to google due to anthropic api error:", e)
            return call_google(messages, model, max_tokens, thinking_budget, False, None)

    # Streaming or tools: use generator
    return _call_anthropic_generator(client, messages, system, model, max_tokens, stream, params, tools, thinking_budget)


def _call_anthropic_generator(
    client, messages, system, model, max_tokens, stream, params, tools, thinking_budget
) -> Generator[LLMChunk, None, None]:
    """Generator version of call_anthropic for streaming and/or tool use."""
    # Tool use loop: continue until stop_reason is not "tool_use"
    max_iterations = 10  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            response = client.messages.create(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                stream=stream,
                **params,
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            print("WARNING: falling back to google due to anthropic api error:", e)
            return call_google(messages, model, max_tokens, thinking_budget, stream, tools)

        if stream:
            # Process streaming response and collect content blocks
            message_content, stop_reason, thinking_blocks = yield from _anthropic_stream_with_tools(response)

            # Check if we need to continue with tool execution
            if stop_reason == "tool_use" and tools:
                # Extract tool use blocks from message content
                tool_uses = [block for block in message_content if block.get("type") == "tool_use"]

                if tool_uses:
                    # Build assistant message content: thinking blocks (if any) + text/tool_use blocks
                    # Thinking blocks MUST come first, then other content
                    assistant_content = thinking_blocks + message_content

                    # Add assistant message with all content (thinking + text + tool_use blocks)
                    messages.append({"role": "assistant", "content": assistant_content})

                    # Execute tools and build tool results
                    tool_results = []
                    for tool_use in tool_uses:
                        result = execute_tool(tool_use["name"], tool_use["input"])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": result
                        })

                    # Add user message with tool results
                    messages.append({"role": "user", "content": tool_results})

                    # Continue the loop to get next response
                    continue

            # Done - stop_reason is not "tool_use" or no tools to execute
            return

        else:
            # Non-streaming mode
            # If no tools, return string for backward compatibility
            if not tools:
                return response.content[0].text

            # With tools, yield chunks and handle tool loop
            content_text = ""
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                    yield LLMChunk(type="response", text=block.text)

            # Check for tool use
            if response.stop_reason == "tool_use":
                # Build message content with all blocks (thinking, text, tool_use)
                # Preserve full block structure including thinking blocks with signatures
                message_content = []
                for block in response.content:
                    if block.type == "thinking":
                        # Preserve complete thinking block including signature
                        # Convert the block to dict to get all fields
                        thinking_dict = block.model_dump(mode="json")
                        message_content.append(thinking_dict)
                    elif block.type == "text":
                        message_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        message_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })

                messages.append({"role": "assistant", "content": message_content})

                # Execute tools
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Done
            return

    # Safety: exceeded max iterations
    print(f"WARNING: Exceeded max iterations ({max_iterations}) in tool use loop")
    return


def _anthropic_stream_with_tools(response) -> Generator[LLMChunk, None, tuple[list, str, list]]:
    """Stream Anthropic response and collect content blocks for tool use.

    Yields LLMChunk items and returns (message_content, stop_reason, thinking_blocks) at the end.

    Thinking blocks must be preserved when continuing with tool results, so we track them separately.
    """
    message_content = []
    thinking_blocks = []
    stop_reason = None
    current_block = None
    current_tool_input_json = ""
    current_thinking = ""

    with response as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    current_block = {"type": "text", "text": ""}
                elif event.content_block.type == "tool_use":
                    current_block = {
                        "type": "tool_use",
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": {}
                    }
                    current_tool_input_json = ""
                elif event.content_block.type == "thinking":
                    current_thinking = ""

            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    # Yield thinking chunk and accumulate
                    yield LLMChunk(type="thinking", text=event.delta.thinking)
                    current_thinking += event.delta.thinking
                elif event.delta.type == "text_delta":
                    # Yield response chunk and accumulate
                    yield LLMChunk(type="response", text=event.delta.text)
                    if current_block and current_block["type"] == "text":
                        current_block["text"] += event.delta.text
                elif event.delta.type == "input_json_delta":
                    # Accumulate tool input JSON
                    current_tool_input_json += event.delta.partial_json

            elif event.type == "content_block_stop":
                # Check if we just finished a thinking block
                if current_thinking:
                    thinking_blocks.append({
                        "type": "thinking",
                        "thinking": current_thinking
                    })
                    current_thinking = ""

                if current_block:
                    if current_block["type"] == "tool_use":
                        # Parse complete JSON input
                        try:
                            current_block["input"] = json.loads(current_tool_input_json)
                        except json.JSONDecodeError:
                            current_block["input"] = {}

                    message_content.append(current_block)
                    current_block = None

            elif event.type == "message_delta":
                if hasattr(event.delta, "stop_reason"):
                    stop_reason = event.delta.stop_reason

    return message_content, stop_reason, thinking_blocks


def call_openai(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = False,
    tools: Optional[list[Tool]] = None,
) -> Generator[LLMChunk, None, None]:
    if tools:
        raise NotImplementedError("Tool use not yet supported for OpenAI provider")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    system, history = split_system(history)
    params = {}
    if can_think_openai(model, thinking_budget):
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
) -> Generator[LLMChunk, None, None]:
    if tools:
        raise NotImplementedError("Tool use not yet supported for Google provider")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    system, history = split_system(history)

    # Convert to Gemini's Content format
    contents = []
    for msg in history:
        # Map assistant to model for Gemini
        role = "model" if msg["role"] == "assistant" else msg["role"]
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    if "2.5-pro" in model:
        thinking_budget = max(thinking_budget or 0, 128)

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


def query_llm(
    history: Sequence[Message],
    settings: Settings,
    stream: bool = True,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
    tools: Optional[list[Tool]] = None,
) -> Generator[LLMChunk, None, None]:
    provider = settings.completions_model_provider
    if provider == ANTHROPIC:
        func = call_anthropic
    elif provider == OPENAI:
        func = call_openai
    elif provider == GOOGLE:
        func = call_google
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return func(
        history,
        settings.completions_model_name,
        max_tokens if max_tokens is not None else settings.max_response_tokens,
        thinking_budget if thinking_budget is not None else settings.thinking_budget,
        stream=stream,
        tools=tools,
    )
