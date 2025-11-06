from typing import TypedDict, Literal, Generator, Sequence, Callable
import json

import anthropic
import openai
from google import genai
from stampy_chat.settings import ANTHROPIC, OPENAI, GOOGLE, Settings
from stampy_chat.env import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from stampy_chat.citations import Message


class LLMChunk(TypedDict, total=False):
    type: Literal["thinking", "response", "tool_use", "tool_result"]
    text: str
    # For tool_use chunks
    tool_name: str
    tool_id: str
    tool_input: dict
    # For tool_result chunks
    tool_result: str


def execute_calculator(expression: str) -> str:
    """Simple calculator tool for testing."""
    try:
        # Only allow basic math operations for safety
        allowed = set("0123456789+-*/().%** ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "A simple calculator that can evaluate mathematical expressions. Supports basic arithmetic operations (+, -, *, /, **, %).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5 + 3')"
                }
            },
            "required": ["expression"]
        }
    }
]

TOOL_EXECUTORS: dict[str, Callable] = {
    "calculator": lambda input: execute_calculator(input["expression"])
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name with the given input."""
    if name not in TOOL_EXECUTORS:
        return f"Error: Unknown tool '{name}'"
    try:
        return TOOL_EXECUTORS[name](tool_input)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


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
    tools: list[dict] | None = None,
) -> Generator[LLMChunk, None, None]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    params = {}
    if can_think_anthropic(model, thinking_budget):
        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if tools:
        params["tools"] = tools

    system, history = split_system(history)

    try:
        if stream:
            return anthropic_stream_with_tools(
                client, model, list(history), system, max_tokens, params
            )
        else:
            response = client.messages.create(
                model=model,
                messages=history,
                system=system,
                max_tokens=max_tokens,
                stream=False,
                **params,
            )
            return response.content[0].text
    except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
        print("WARNING: falling back to google due to anthropic api error:", e)
        return call_google(history, model, max_tokens, thinking_budget, stream)


def anthropic_stream_with_tools(
    client: anthropic.Anthropic,
    model: str,
    messages: list,
    system: str,
    max_tokens: int,
    params: dict,
) -> Generator[LLMChunk, None, None]:
    """
    Stream responses from Anthropic with tool use support.
    Handles multiple turns of tool use until completion.
    """
    # Loop until we get a response without tool use
    while True:
        # Make the streaming request
        response = client.messages.create(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            stream=True,
            **params,
        )

        # Track content blocks for this turn
        current_content = []
        current_block = None
        current_block_index = 0

        # Process the stream
        for event in response:
            if event.type == "content_block_start":
                # Starting a new content block
                if event.content_block.type == "thinking":
                    current_block = {"type": "thinking", "thinking": ""}
                elif event.content_block.type == "text":
                    current_block = {"type": "text", "text": ""}
                elif event.content_block.type == "tool_use":
                    current_block = {
                        "type": "tool_use",
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": {}
                    }
                current_content.append(current_block)
                current_block_index = event.index

            elif event.type == "content_block_delta":
                # Update the current block and yield chunk
                if event.delta.type == "thinking_delta":
                    current_block["thinking"] += event.delta.thinking
                    yield LLMChunk(type="thinking", text=event.delta.thinking)

                elif event.delta.type == "text_delta":
                    current_block["text"] += event.delta.text
                    yield LLMChunk(type="response", text=event.delta.text)

                elif event.delta.type == "input_json_delta":
                    # Accumulate JSON for tool input
                    if "input_json" not in current_block:
                        current_block["input_json"] = ""
                    current_block["input_json"] += event.delta.partial_json

            elif event.type == "content_block_stop":
                # Block is complete, parse tool input if needed
                if current_block and current_block.get("type") == "tool_use":
                    if "input_json" in current_block:
                        try:
                            current_block["input"] = json.loads(current_block["input_json"])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse tool input JSON: {e}")
                            current_block["input"] = {}
                        del current_block["input_json"]

        # Check if any tools were used
        tool_uses = [b for b in current_content if b.get("type") == "tool_use"]

        if not tool_uses:
            # No tool use, we're done
            break

        # Add assistant's message to history
        messages.append({
            "role": "assistant",
            "content": current_content
        })

        # Execute tools and yield tool use/result chunks
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use["name"]
            tool_id = tool_use["id"]
            tool_input = tool_use["input"]

            # Yield tool_use chunk
            yield LLMChunk(
                type="tool_use",
                tool_name=tool_name,
                tool_id=tool_id,
                tool_input=tool_input
            )

            # Execute the tool
            result = execute_tool(tool_name, tool_input)

            # Yield tool_result chunk
            yield LLMChunk(
                type="tool_result",
                tool_id=tool_id,
                tool_result=result
            )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result
            })

        # Add tool results as user message
        messages.append({
            "role": "user",
            "content": tool_results
        })


def call_openai(
    history: Sequence[Message],
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
    stream: bool = False,
) -> Generator[LLMChunk, None, None]:
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
) -> Generator[LLMChunk, None, None]:
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
) -> Generator[LLMChunk, None, None]:
    provider = settings.completions_model_provider

    # Prepare common arguments
    kwargs = {
        "history": history,
        "model": settings.completions_model_name,
        "max_tokens": max_tokens if max_tokens is not None else settings.max_response_tokens,
        "thinking_budget": thinking_budget if thinking_budget is not None else settings.thinking_budget,
        "stream": stream,
    }

    # Add tools for Anthropic if enabled
    if provider == ANTHROPIC:
        if settings.enable_tools:
            kwargs["tools"] = TOOL_DEFINITIONS
        return call_anthropic(**kwargs)
    elif provider == OPENAI:
        return call_openai(**kwargs)
    elif provider == GOOGLE:
        return call_google(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
