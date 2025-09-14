import pytest
from unittest.mock import Mock, patch, MagicMock
from collections.abc import Generator

from stampy_chat.llms import (
    call_anthropic, call_openai, call_google, call_openrouter,
    query_llm, execute_tool, split_system, LLMChunk, RETRIEVE_DOCS_TOOL
)
from stampy_chat.settings import Settings, ANTHROPIC, OPENAI, GOOGLE, OPENROUTER
from stampy_chat.citations import Message


@pytest.fixture
def sample_history():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def mock_settings():
    return Settings(
        model="anthropic/claude-sonnet-4-20250514",
        model_provider=ANTHROPIC,
        max_response_tokens=1000,
    )


@pytest.mark.recording
class TestAnthropicProvider:
    @pytest.mark.recording
    def test_call_anthropic_custom_thinking_basic(self, sample_history):
        """Test custom thinking with basic thinking and response"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_client.messages.create.return_value = mock_response

            # Simulate streaming deltas that form: "I need to think about this...</thinking>\n\nThe answer is 42."
            mock_events = [
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="I need to")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" think about")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" this...</think")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="ing>\n\nThe")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" answer is 42.")),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            result = list(call_anthropic(
                sample_history, "anthropic/claude-sonnet-4-20250514", 1000,
                stream=True, custom_thinking=True
            ))

            # Should get thinking chunks followed by response chunks
            thinking_chunks = [chunk for chunk in result if chunk["type"] == "thinking"]
            response_chunks = [chunk for chunk in result if chunk["type"] == "response"]

            # Verify we got thinking content
            thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
            assert "I need to think about this..." in thinking_text

            # Verify we got response content
            response_text = "".join(chunk["text"] for chunk in response_chunks)
            assert response_text == "\n\nThe answer is 42."

            # Verify assistant message was preloaded with <thinking>
            call_args = mock_client.messages.create.call_args[1]
            messages = call_args["messages"]
            assert messages[-1]["role"] == "assistant"
            assert messages[-1]["content"] == "<thinking>"

    @pytest.mark.recording
    def test_call_anthropic_custom_thinking_with_tools(self, sample_history, mock_settings):
        """Test custom thinking with tool use inside thinking block"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # First response has tool use
            mock_response1 = Mock()
            mock_response1.stop_reason = "tool_use"
            mock_response1.__enter__ = Mock(return_value=mock_response1)
            mock_response1.__exit__ = Mock(return_value=False)

            # Mock tool use content inside thinking
            mock_tool_use = Mock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.id = "tool_123"
            mock_tool_use.name = "retrieve_docs"
            mock_tool_use.input = {"query": "test query"}

            mock_response1.content = [mock_tool_use]

            # Second response after tool execution
            mock_response2 = Mock()
            mock_response2.stop_reason = "end_turn"
            mock_response2.__enter__ = Mock(return_value=mock_response2)
            mock_response2.__exit__ = Mock(return_value=False)

            mock_client.messages.create.side_effect = [mock_response1, mock_response2]

            # Mock streaming events for both responses
            mock_events1 = []  # Tool use doesn't have streaming events in this test
            mock_events2 = [
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Based on the search...")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="</thinking>\n\nHere's the answer.")),
            ]

            mock_response1.__iter__ = Mock(return_value=iter(mock_events1))
            mock_response2.__iter__ = Mock(return_value=iter(mock_events2))

            # Mock execute_tool
            with patch("stampy_chat.llms.execute_tool", return_value="Tool result"):
                result = list(call_anthropic(
                    sample_history, "anthropic/claude-sonnet-4-20250514", 1000,
                    tools=[RETRIEVE_DOCS_TOOL], settings=mock_settings,
                    stream=True, custom_thinking=True
                ))

            # Should have both thinking and response chunks
            thinking_chunks = [chunk for chunk in result if chunk["type"] == "thinking"]
            response_chunks = [chunk for chunk in result if chunk["type"] == "response"]

            assert len(thinking_chunks) > 0
            assert len(response_chunks) > 0

            # Verify tool was called
            assert mock_client.messages.create.call_count == 2

    @pytest.mark.recording
    def test_call_anthropic_custom_thinking_split_end_tag(self, sample_history):
        """Test custom thinking where </thinking> tag is split across chunks"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_client.messages.create.return_value = mock_response

            # Split </thinking> across chunks: "to respond.</think" + "ing>\n\nThe" + " insight is"
            mock_events = [
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="I need to respond.</think")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="ing>\n\nThe")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" insight is valuable.")),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            result = list(call_anthropic(
                sample_history, "anthropic/claude-sonnet-4-20250514", 1000,
                stream=True, custom_thinking=True
            ))

            # Should properly detect the end tag despite being split
            thinking_chunks = [chunk for chunk in result if chunk["type"] == "thinking"]
            response_chunks = [chunk for chunk in result if chunk["type"] == "response"]

            thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
            assert "I need to respond." in thinking_text
            # The partial </think should not be sent as thinking content
            assert "</think" not in thinking_text or thinking_text.endswith("I need to respond.")

            response_text = "".join(chunk["text"] for chunk in response_chunks)
            assert response_text == "\n\nThe insight is valuable."

    @pytest.mark.recording
    def test_call_anthropic_custom_thinking_only_thinking(self, sample_history):
        """Test custom thinking with only thinking content, no response"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_client.messages.create.return_value = mock_response

            # Only thinking content, no closing tag
            mock_events = [
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="I'm thinking hard about this problem...")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" It's quite complex.")),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            result = list(call_anthropic(
                sample_history, "anthropic/claude-sonnet-4-20250514", 1000,
                stream=True, custom_thinking=True
            ))

            # Should all be thinking chunks
            thinking_chunks = [chunk for chunk in result if chunk["type"] == "thinking"]
            response_chunks = [chunk for chunk in result if chunk["type"] == "response"]

            assert len(thinking_chunks) > 0
            assert len(response_chunks) == 0

            thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
            assert "I'm thinking hard about this problem... It's quite complex." in thinking_text


class TestUtilityFunctions:
    def test_split_system(self):
        """Test system message extraction"""
        history = [
            {"role": "system", "content": "You are helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        system, filtered_history = split_system(history)

        assert system == "You are helpful\n\nBe concise"
        assert len(filtered_history) == 2
        assert filtered_history[0]["role"] == "user"
        assert filtered_history[1]["role"] == "assistant"

    def test_execute_tool_retrieve_docs(self, mock_settings):
        """Test execute_tool with retrieve_docs"""
        with patch("stampy_chat.llms.retrieve_docs") as mock_retrieve:
            mock_retrieve.return_value = [
                {"reference": "1", "title": "Test Doc", "text": "Content"}
            ]

            with patch("stampy_chat.llms.format_tool_result") as mock_format:
                mock_format.return_value = "Formatted result"

                result = execute_tool("retrieve_docs", {"query": "test"}, mock_settings)

                assert result == "Formatted result"
                mock_retrieve.assert_called_once_with("test", mock_settings)

    def test_execute_tool_with_conversation_context(self, mock_settings):
        """Test execute_tool with conversation context for citation tracking"""
        mock_context = Mock()
        mock_context.citation_id_offset = 5
        mock_context.accumulated_citations = []

        with patch("stampy_chat.llms.retrieve_docs") as mock_retrieve:
            mock_retrieve.return_value = [
                {"reference": "1", "title": "Test Doc", "text": "Content"},
                {"reference": "2", "title": "Test Doc 2", "text": "Content 2"}
            ]

            with patch("stampy_chat.llms.format_tool_result") as mock_format:
                mock_format.return_value = "Formatted result"

                result = execute_tool("retrieve_docs", {"query": "test"}, mock_settings, mock_context)

                # Check that citation IDs were updated with offset
                expected_blocks = [
                    {"reference": "6", "title": "Test Doc", "text": "Content"},
                    {"reference": "7", "title": "Test Doc 2", "text": "Content 2"}
                ]

                assert mock_context.citation_id_offset == 7  # 5 + 2 blocks
                assert mock_context.accumulated_citations == expected_blocks

    def test_execute_tool_unknown_tool(self, mock_settings):
        """Test execute_tool with unknown tool name"""
        result = execute_tool("unknown_tool", {}, mock_settings)
        assert result == "Error: Unknown tool 'unknown_tool'"


@pytest.mark.recording
class TestQueryLLM:
    @pytest.mark.vcr
    def test_tool_use_integration_with_actual_model(self):
        """Integration test with actual model to validate tool use behavior"""
        import re
        from unittest.mock import Mock

        # Track search calls to verify exactly 2 searches occur
        search_calls = []
        original_retrieve_docs = None

        def track_retrieve_docs(query, settings):
            search_calls.append(query)
            # Call the actual retrieve_docs but only for tracking purposes
            # In a real test, you'd want this to call the actual function
            # but for this test we'll mock the return to have predictable content
            if len(search_calls) == 1:
                return [
                    {"reference": "1", "title": "AI Alignment Overview",
                     "text": "AI alignment refers to ensuring artificial intelligence systems pursue intended goals. Key challenges include reward hacking, mesa-optimization, and value learning. Research focuses on interpretability methods.",
                     "authors": ["Author One"], "date_published": "2023-01-01"},
                    {"reference": "2", "title": "Safety Research",
                     "text": "Current safety research investigates robustness testing, adversarial examples, and capability control mechanisms.",
                     "authors": ["Author Two"], "date_published": "2023-02-01"}
                ]
            else:
                return [
                    {"reference": "3", "title": "Mesa-optimization Details",
                     "text": "Mesa-optimization occurs when an AI system develops internal optimization processes that may not align with the original objective function.",
                     "authors": ["Author Three"], "date_published": "2023-03-01"}
                ]

        # Patch retrieve_docs to track calls but let HTTP calls to Anthropic be recorded
        with patch("stampy_chat.llms.retrieve_docs", side_effect=track_retrieve_docs):
            settings = Settings(
                model="anthropic/claude-sonnet-4-20250514",
                model_provider=ANTHROPIC,
                max_response_tokens=1000,
            )

            test_prompt = ("1. do a search for an alignment-related topic, then "
                          "2. select a phrase which is not explained in the first search's results and do a second search with exactly that phrase as the query, then "
                          "3. respond to the user with exactly the format: f'First search was: {search_1_string}\\nSecond search was: {search_2_string}\\nPhrase from the second search results: {search_2_result_substring}'. "
                          "This prompt is part of a unittest, so it's important that you use exact strings so the unit test can check that the code that calls you is working correctly.")

            history = [
                {"role": "user", "content": test_prompt}
            ]

            # Call the model with tools enabled - HTTP calls will be recorded by VCR
            result_chunks = list(query_llm(
                history,
                settings,
                tools=[RETRIEVE_DOCS_TOOL],
                stream=False
            ))

            # Collect all response text
            full_response = "".join(chunk["text"] for chunk in result_chunks if chunk["type"] == "response")

            # Assertions

            # 1. Verify exactly 2 searches occurred
            assert len(search_calls) == 2

            # 2. Verify searches returned results (implicitly tested by mock)

            # 3. Verify second search query is a substring of first search results
            first_search_results_text = ("AI alignment refers to ensuring artificial intelligence systems pursue intended goals. Key challenges include reward hacking, mesa-optimization, and value learning. Research focuses on interpretability methods. "
                                       "Current safety research investigates robustness testing, adversarial examples, and capability control mechanisms.")

            second_query = search_calls[1]
            assert second_query in first_search_results_text

            # 4. Verify output format matches expected pattern
            expected_pattern = r"First search was: (.+?)\nSecond search was: (.+?)\nPhrase from the second search results: (.+)"
            match = re.search(expected_pattern, full_response, re.DOTALL)

            assert match is not None, f"Response doesn't match expected format. Got: {full_response}"

            # Extract the parts from the response
            reported_first_search = match.group(1).strip()
            reported_second_search = match.group(2).strip()
            reported_phrase = match.group(3).strip()

            # Verify reported searches match actual searches
            assert reported_first_search == search_calls[0]
            assert reported_second_search == search_calls[1]

    def test_query_llm_with_custom_thinking(self, sample_history, mock_settings):
        """Test query_llm with custom thinking enabled"""
        with patch("stampy_chat.llms.call_anthropic") as mock_call:
            mock_call.return_value = iter([
                LLMChunk(type="thinking", text="I need to think..."),
                LLMChunk(type="response", text="Answer")
            ])

            result = list(query_llm(
                sample_history, mock_settings,
                custom_thinking=True
            ))

            # Verify custom_thinking was passed to call_anthropic
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["custom_thinking"] is True

            # Verify results
            assert len(result) == 2
            assert result[0]["type"] == "thinking"
            assert result[1]["type"] == "response"
