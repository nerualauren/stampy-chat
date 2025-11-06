"""Test the tool use state machine logic with mocks."""
from unittest.mock import Mock, MagicMock
from stampy_chat.llms import anthropic_stream_with_tools, LLMChunk


def test_single_tool_use_flow():
    """Test the state machine handles a single tool use correctly."""

    # Mock client
    mock_client = Mock()

    # Create mock events for the stream
    # First request: thinking + text + tool_use
    tool_block = Mock(spec=["type", "id", "name"])
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "calculator"

    first_stream = [
        Mock(type="content_block_start", index=0, content_block=Mock(type="thinking")),
        Mock(type="content_block_delta", delta=Mock(type="thinking_delta", thinking="Let me calculate...")),
        Mock(type="content_block_stop"),

        Mock(type="content_block_start", index=1, content_block=Mock(type="text")),
        Mock(type="content_block_delta", delta=Mock(type="text_delta", text="I'll use the calculator.")),
        Mock(type="content_block_stop"),

        Mock(type="content_block_start", index=2, content_block=tool_block),
        Mock(type="content_block_delta", delta=Mock(type="input_json_delta", partial_json='{"expression": "2+2"}')),
        Mock(type="content_block_stop"),
    ]

    # Second request: text response (no tool use)
    second_stream = [
        Mock(type="content_block_start", index=0, content_block=Mock(type="text")),
        Mock(type="content_block_delta", delta=Mock(type="text_delta", text="The result is 4.")),
        Mock(type="content_block_stop"),
    ]

    # Mock the messages.create method to return our streams
    mock_client.messages.create = Mock(side_effect=[first_stream, second_stream])

    # Run the function
    messages = []
    chunks = list(anthropic_stream_with_tools(
        client=mock_client,
        model="test-model",
        messages=messages,
        system="test system",
        max_tokens=100,
        params={"tools": [{"name": "calculator"}]},
    ))

    # Verify we got the expected chunks
    thinking_chunks = [c for c in chunks if c.get("type") == "thinking"]
    response_chunks = [c for c in chunks if c.get("type") == "response"]
    tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]
    tool_result_chunks = [c for c in chunks if c.get("type") == "tool_result"]

    assert len(thinking_chunks) == 1
    assert thinking_chunks[0]["text"] == "Let me calculate..."

    assert len(response_chunks) == 2  # "I'll use the calculator." + "The result is 4."

    assert len(tool_use_chunks) == 1
    assert tool_use_chunks[0]["tool_name"] == "calculator"
    assert tool_use_chunks[0]["tool_input"]["expression"] == "2+2"

    assert len(tool_result_chunks) == 1
    assert "4" in tool_result_chunks[0]["tool_result"]

    # Verify we made 2 API calls (initial + after tool use)
    assert mock_client.messages.create.call_count == 2


def test_no_tool_use_flow():
    """Test the state machine handles responses without tool use."""

    mock_client = Mock()

    # Single request with no tool use
    stream = [
        Mock(type="content_block_start", index=0, content_block=Mock(type="text")),
        Mock(type="content_block_delta", delta=Mock(type="text_delta", text="The answer is 4.")),
        Mock(type="content_block_stop"),
    ]

    mock_client.messages.create = Mock(return_value=stream)

    messages = []
    chunks = list(anthropic_stream_with_tools(
        client=mock_client,
        model="test-model",
        messages=messages,
        system="test system",
        max_tokens=100,
        params={},
    ))

    # Should only have response chunks
    response_chunks = [c for c in chunks if c.get("type") == "response"]
    tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]

    assert len(response_chunks) == 1
    assert len(tool_use_chunks) == 0

    # Should make only 1 API call
    assert mock_client.messages.create.call_count == 1


def test_multiple_tool_uses():
    """Test the state machine handles multiple sequential tool uses."""

    mock_client = Mock()

    # First request: tool use
    tool_1_block = Mock(spec=["type", "id", "name"])
    tool_1_block.type = "tool_use"
    tool_1_block.id = "tool_1"
    tool_1_block.name = "calculator"
    first_stream = [
        Mock(type="content_block_start", index=0, content_block=tool_1_block),
        Mock(type="content_block_delta", delta=Mock(type="input_json_delta", partial_json='{"expression": "10+5"}')),
        Mock(type="content_block_stop"),
    ]

    # Second request: another tool use
    tool_2_block = Mock(spec=["type", "id", "name"])
    tool_2_block.type = "tool_use"
    tool_2_block.id = "tool_2"
    tool_2_block.name = "calculator"
    second_stream = [
        Mock(type="content_block_start", index=0, content_block=tool_2_block),
        Mock(type="content_block_delta", delta=Mock(type="input_json_delta", partial_json='{"expression": "15*3"}')),
        Mock(type="content_block_stop"),
    ]

    # Third request: final response
    third_stream = [
        Mock(type="content_block_start", index=0, content_block=Mock(type="text")),
        Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Final answer: 45")),
        Mock(type="content_block_stop"),
    ]

    mock_client.messages.create = Mock(side_effect=[first_stream, second_stream, third_stream])

    messages = []
    chunks = list(anthropic_stream_with_tools(
        client=mock_client,
        model="test-model",
        messages=messages,
        system="test system",
        max_tokens=100,
        params={"tools": [{"name": "calculator"}]},
    ))

    # Should have 2 tool uses and 2 tool results
    tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]
    tool_result_chunks = [c for c in chunks if c.get("type") == "tool_result"]

    assert len(tool_use_chunks) == 2
    assert len(tool_result_chunks) == 2

    # First tool use: 10+5
    assert tool_use_chunks[0]["tool_input"]["expression"] == "10+5"
    assert "15" in tool_result_chunks[0]["tool_result"]

    # Second tool use: 15*3
    assert tool_use_chunks[1]["tool_input"]["expression"] == "15*3"
    assert "45" in tool_result_chunks[1]["tool_result"]

    # Should make 3 API calls
    assert mock_client.messages.create.call_count == 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
