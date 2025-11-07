"""
Comprehensive VCR tests for llms.py streaming and thinking functionality.

This test suite validates:
- Non-streaming calls
- Streaming without thinking
- Streaming with thinking (extended thinking budget)

Uses pytest-recording to record API interactions for fast, reliable replays.
"""
import os
import pytest
from stampy_chat.llms import call_anthropic


@pytest.mark.vcr
def test_call_anthropic_non_streaming_no_thinking():
    """Test basic non-streaming, non-thinking call.

    This verifies that:
    - call_anthropic with stream=False returns a plain string
    - The response is valid and contains expected content
    - Minimal API usage with simple prompt
    """
    # Use a simple prompt to minimize API usage during recording
    history = [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ]

    # Call with stream=False, thinking_budget=0
    response = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        thinking_budget=0,
        stream=False
    )

    # Assert response is a string
    assert isinstance(response, str)
    assert len(response) > 0

    # Should contain the answer (4 or "four")
    assert any(word in response.lower() for word in ["4", "four"])


@pytest.mark.vcr
def test_call_anthropic_streaming_no_thinking():
    """Test streaming without thinking.

    This verifies that:
    - call_anthropic with stream=True returns a generator
    - All chunks are type="response"
    - No chunks have type="thinking"
    - Chunks contain valid text
    """
    history = [
        {"role": "user", "content": "Say 'hello' in one word."}
    ]

    # Call with stream=True, thinking_budget=0
    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        thinking_budget=0,
        stream=True
    )

    # Collect all chunks
    chunks = list(stream)

    # Assert we got chunks
    assert len(chunks) > 0

    # Assert all chunks are type="response"
    for chunk in chunks:
        assert chunk["type"] == "response"
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0

    # Assert no chunks have type="thinking"
    thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
    assert len(thinking_chunks) == 0

    # Verify we got meaningful content
    full_text = "".join(chunk["text"] for chunk in chunks)
    assert len(full_text) > 0
    assert "hello" in full_text.lower()


@pytest.mark.vcr
def test_call_anthropic_streaming_with_thinking():
    """Test streaming WITH thinking.

    This verifies that:
    - call_anthropic with stream=True and thinking_budget=2048 works
    - We get both thinking and response chunks
    - Thinking chunks come before response chunks
    - Thinking content contains analysis/reasoning
    - Response content contains the answer
    """
    # Use a prompt that will trigger thinking
    history = [
        {"role": "user", "content": "Analyze this problem step by step: What is 15 * 17?"}
    ]

    # Call with stream=True, thinking_budget=2048 (minimum for thinking)
    # Note: max_tokens must be > thinking_budget
    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        thinking_budget=2048,
        stream=True
    )

    # Collect chunks into two lists
    thinking_chunks = []
    response_chunks = []

    for chunk in stream:
        if chunk["type"] == "thinking":
            thinking_chunks.append(chunk)
        elif chunk["type"] == "response":
            response_chunks.append(chunk)

    # Assert we got both thinking and response chunks
    assert len(thinking_chunks) > 0, "Expected thinking chunks but got none"
    assert len(response_chunks) > 0, "Expected response chunks but got none"

    # Verify thinking content exists and contains reasoning
    thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
    assert len(thinking_text) > 0
    # Thinking should contain some analysis (numbers, calculation-related words)
    assert any(word in thinking_text.lower() for word in ["15", "17", "multiply", "calculation", "step"])

    # Verify response content contains the answer
    response_text = "".join(chunk["text"] for chunk in response_chunks)
    assert len(response_text) > 0
    # Should contain the answer (255)
    assert "255" in response_text


@pytest.mark.vcr
def test_call_anthropic_streaming_thinking_chunk_order():
    """Test that thinking chunks come before response chunks.

    This verifies the ordering behavior of streaming with thinking enabled.
    """
    history = [
        {"role": "user", "content": "Think step by step: What is 8 * 9?"}
    ]

    stream = call_anthropic(
        history=history,
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        thinking_budget=2048,
        stream=True
    )

    # Track the order of chunk types
    chunk_types = []
    for chunk in stream:
        chunk_types.append(chunk["type"])

    # Assert we have both types
    assert "thinking" in chunk_types
    assert "response" in chunk_types

    # Find first occurrence of each type
    first_thinking_idx = chunk_types.index("thinking")
    first_response_idx = chunk_types.index("response")

    # Assert thinking comes before response
    assert first_thinking_idx < first_response_idx, \
        f"Expected thinking chunks before response chunks, but got first thinking at {first_thinking_idx} and first response at {first_response_idx}"
