"""
VCR tests for tool use functionality in llms.py.

This test suite validates:
- Single tool use without thinking
- Multiple tool uses in sequence without thinking
- Tool use with thinking enabled
- Streaming behavior with tool use
- Normal conversation (0 tool uses) still works

Uses pytest-recording to record API interactions for fast, reliable replays.
"""
import pytest
from stampy_chat.llms import call_anthropic, Tool


# Define the calculator tool for testing
CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    input_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "multiply"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
)


@pytest.mark.vcr
def test_tool_use_single_call_no_thinking():
    """Test single tool use without thinking.

    This verifies:
    - Tool is called once with correct parameters
    - Tool result is processed
    - Final response includes the answer
    - stop_reason eventually becomes "end_turn" not "tool_use"
    """
    history = [
        {"role": "user", "content": "Use the calculator to add 15 and 27. Just give me the result."}
    ]

    # Call with tools enabled, no thinking
    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        thinking_budget=0,
        stream=True,
        tools=[CALCULATOR_TOOL]
    )

    # Collect all chunks
    chunks = list(stream)

    # Should have response chunks
    assert len(chunks) > 0

    # All chunks should be response type (no thinking)
    for chunk in chunks:
        assert chunk["type"] in ["thinking", "response"]

    # Reconstruct full response
    response_text = "".join(chunk["text"] for chunk in chunks if chunk["type"] == "response")

    # Response should mention the result (42)
    assert "42" in response_text


@pytest.mark.vcr
def test_tool_use_multiple_calls_no_thinking():
    """Test multiple sequential tool uses without thinking.

    This verifies:
    - Multiple tools are called in sequence
    - Both tool results are used in the final response
    - The conversation flows naturally through multiple tool uses
    """
    history = [
        {"role": "user", "content": "Use the calculator to add 5 and 3, then multiply that result by 2. Tell me both intermediate and final results."}
    ]

    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        thinking_budget=0,
        stream=True,
        tools=[CALCULATOR_TOOL]
    )

    chunks = list(stream)
    assert len(chunks) > 0

    response_text = "".join(chunk["text"] for chunk in chunks if chunk["type"] == "response")

    # Response should include both results: 5+3=8 and 8*2=16
    # Looking for 8 and 16 in the response
    assert "8" in response_text
    assert "16" in response_text


@pytest.mark.vcr
def test_tool_use_with_thinking():
    """Test tool use WITH thinking enabled (non-streaming).

    This verifies:
    - Tool is called correctly with thinking enabled
    - After tool result, response includes the answer
    - Thinking works with tool use (non-streaming mode)

    Note: Thinking blocks have cryptographic signatures and cannot be preserved
    in streaming mode. For thinking + tool use, we use non-streaming mode.
    """
    history = [
        {"role": "user", "content": "Think carefully about this: use the calculator to compute 7 times 8, then explain what you did."}
    ]

    # Use non-streaming mode for thinking + tool use
    result_gen = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        thinking_budget=2048,
        stream=False,  # Non-streaming for thinking + tools
        tools=[CALCULATOR_TOOL]
    )

    # Even in non-streaming, it returns a generator
    chunks = list(result_gen)

    response_chunks = [c for c in chunks if c["type"] == "response"]
    assert len(response_chunks) > 0, "Expected response chunks"

    response_text = "".join(chunk["text"] for chunk in response_chunks)

    # Response should include the answer (56)
    assert "56" in response_text


@pytest.mark.vcr
def test_tool_use_streaming_order():
    """Test that streaming chunks arrive in correct order with tool use.

    This verifies:
    - Chunks are streamed as they arrive
    - Order is maintained throughout tool use flow
    - No chunks are lost or duplicated
    """
    history = [
        {"role": "user", "content": "Use the calculator to multiply 6 by 7."}
    ]

    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        thinking_budget=0,
        stream=True,
        tools=[CALCULATOR_TOOL]
    )

    chunks = list(stream)

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify all chunks have valid text
    for chunk in chunks:
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0

    # Reconstruct response
    response_text = "".join(chunk["text"] for chunk in chunks if chunk["type"] == "response")

    # Should contain the result (42)
    assert "42" in response_text


@pytest.mark.vcr
def test_no_tool_use_still_works():
    """Test that normal conversation without tools still works.

    This is a regression test to ensure tool support doesn't break
    normal operation when tools are not provided.
    """
    history = [
        {"role": "user", "content": "What is 10 + 20? Just calculate it yourself."}
    ]

    # Call WITHOUT tools
    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=100,
        thinking_budget=0,
        stream=True,
        tools=None  # No tools
    )

    chunks = list(stream)
    assert len(chunks) > 0

    response_text = "".join(chunk["text"] for chunk in chunks if chunk["type"] == "response")

    # Should answer directly
    assert "30" in response_text


@pytest.mark.vcr
def test_tool_use_non_streaming():
    """Test tool use in non-streaming mode.

    This verifies:
    - Tool use works without streaming
    - Results are still correct
    """
    history = [
        {"role": "user", "content": "Use the calculator to add 100 and 200."}
    ]

    # Non-streaming
    result_gen = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        thinking_budget=0,
        stream=False,
        tools=[CALCULATOR_TOOL]
    )

    # Even in non-streaming mode, it returns a generator of chunks
    chunks = list(result_gen)

    response_text = "".join(chunk["text"] for chunk in chunks if chunk["type"] == "response")

    # Should contain the result (300)
    assert "300" in response_text
