import json
from pathlib import Path
import vcr
import pytest
from stampy_chat.llms import call_anthropic, LLMChunk, TOOL_DEFINITIONS
from stampy_chat.settings import Settings

# Configure VCR
vcr_config = vcr.VCR(
    cassette_library_dir=str(Path(__file__).parent / "fixtures" / "vcr_cassettes"),
    record_mode="once",  # Only record if cassette doesn't exist
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    filter_headers=["authorization", "x-api-key"],
)


class TestAnthropicWithTools:
    """Test Anthropic API with streaming, thinking, and tool use."""

    @vcr_config.use_cassette("anthropic_with_calculator_tool.yaml")
    def test_calculator_tool_use(self):
        """Test that Claude can use the calculator tool."""
        history = [
            {
                "role": "user",
                "content": "What is 123 * 456? Use the calculator tool to compute this."
            }
        ]

        chunks = list(call_anthropic(
            history=history,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            thinking_budget=2048,
            stream=True,
            tools=TOOL_DEFINITIONS,
        ))

        # Extract chunks by type
        thinking_chunks = [c for c in chunks if c.get("type") == "thinking"]
        response_chunks = [c for c in chunks if c.get("type") == "response"]
        tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]
        tool_result_chunks = [c for c in chunks if c.get("type") == "tool_result"]

        # Should have thinking (model plans to use calculator)
        assert len(thinking_chunks) > 0, "Should have thinking chunks"

        # Should have tool use
        assert len(tool_use_chunks) == 1, "Should have exactly one tool use"
        tool_use = tool_use_chunks[0]
        assert tool_use["tool_name"] == "calculator"
        assert "123 * 456" in str(tool_use["tool_input"])

        # Should have tool result
        assert len(tool_result_chunks) == 1, "Should have exactly one tool result"
        tool_result = tool_result_chunks[0]
        assert "56088" in tool_result["tool_result"]

        # Should have final response mentioning the result
        response_text = "".join(c["text"] for c in response_chunks)
        assert "56088" in response_text or "56,088" in response_text

    @vcr_config.use_cassette("anthropic_without_tool_use.yaml")
    def test_no_tool_use_when_not_needed(self):
        """Test that Claude doesn't use tools when not needed."""
        history = [
            {
                "role": "user",
                "content": "What is 2 + 2? Just answer directly, no need for tools."
            }
        ]

        chunks = list(call_anthropic(
            history=history,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            thinking_budget=2048,
            stream=True,
            tools=TOOL_DEFINITIONS,
        ))

        # Should have response chunks
        response_chunks = [c for c in chunks if c.get("type") == "response"]
        assert len(response_chunks) > 0, "Should have response chunks"

        # Should NOT have tool use
        tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]
        assert len(tool_use_chunks) == 0, "Should not use tools for simple math"

        # Response should contain the answer
        response_text = "".join(c["text"] for c in response_chunks)
        assert "4" in response_text

    @vcr_config.use_cassette("anthropic_multiple_tool_uses.yaml")
    def test_multiple_tool_uses(self):
        """Test that Claude can use tools multiple times in one conversation."""
        history = [
            {
                "role": "user",
                "content": "Calculate 10 + 5, then calculate the result times 3. Use the calculator for both."
            }
        ]

        chunks = list(call_anthropic(
            history=history,
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            thinking_budget=2048,
            stream=True,
            tools=TOOL_DEFINITIONS,
        ))

        # Should have multiple tool uses (at least 2)
        tool_use_chunks = [c for c in chunks if c.get("type") == "tool_use"]
        assert len(tool_use_chunks) >= 2, f"Should have at least 2 tool uses, got {len(tool_use_chunks)}"

        # Should have corresponding tool results
        tool_result_chunks = [c for c in chunks if c.get("type") == "tool_result"]
        assert len(tool_result_chunks) == len(tool_use_chunks), \
            "Should have equal number of tool uses and results"

        # Final response should contain the final result (45)
        response_chunks = [c for c in chunks if c.get("type") == "response"]
        response_text = "".join(c["text"] for c in response_chunks)
        assert "45" in response_text


def test_chunk_types():
    """Test that LLMChunk TypedDict accepts all expected fields."""
    # Test thinking chunk
    thinking: LLMChunk = {"type": "thinking", "text": "Let me think..."}
    assert thinking["type"] == "thinking"

    # Test response chunk
    response: LLMChunk = {"type": "response", "text": "Here's my answer"}
    assert response["type"] == "response"

    # Test tool_use chunk
    tool_use: LLMChunk = {
        "type": "tool_use",
        "tool_name": "calculator",
        "tool_id": "toolu_123",
        "tool_input": {"expression": "2+2"}
    }
    assert tool_use["type"] == "tool_use"

    # Test tool_result chunk
    tool_result: LLMChunk = {
        "type": "tool_result",
        "tool_id": "toolu_123",
        "tool_result": "4"
    }
    assert tool_result["type"] == "tool_result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
