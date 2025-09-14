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
        model="anthropic/claude-3-5-sonnet-20241022",
        model_provider=ANTHROPIC,
        max_response_tokens=1000,
    )


@pytest.mark.recording
class TestAnthropicProvider:
    @pytest.mark.recording
    def test_call_anthropic_basic_stream(self, sample_history, mock_settings):
        """Test basic streaming response from Anthropic"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock streaming response
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response

            # Mock stream events
            mock_events = [
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Hello")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" there!")),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            # Test the generator
            result = list(call_anthropic(
                sample_history, "anthropic/claude-3-5-sonnet-20241022", 1000, stream=True
            ))

            assert len(result) == 2
            assert result[0]["type"] == "response"
            assert result[0]["text"] == "Hello"
            assert result[1]["type"] == "response"
            assert result[1]["text"] == " there!"

    @pytest.mark.recording
    def test_call_anthropic_with_thinking(self, sample_history):
        """Test Anthropic with thinking budget"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response

            mock_events = [
                Mock(type="content_block_delta", delta=Mock(type="thinking_delta", thinking="Let me think...")),
                Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Answer")),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            result = list(call_anthropic(
                sample_history, "anthropic/claude-3-5-sonnet-20241022", 1000,
                thinking_budget=2000, stream=True
            ))

            assert len(result) == 2
            assert result[0]["type"] == "thinking"
            assert result[0]["text"] == "Let me think..."
            assert result[1]["type"] == "response"
            assert result[1]["text"] == "Answer"

            # Verify thinking was enabled in API call
            call_args = mock_client.messages.create.call_args[1]
            assert "thinking" in call_args
            assert call_args["thinking"]["budget_tokens"] == 2000

    @pytest.mark.recording
    def test_call_anthropic_with_tools(self, sample_history, mock_settings):
        """Test Anthropic with tool use"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock response with tool use
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"

            # Mock tool use content
            mock_tool_use = Mock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.id = "tool_123"
            mock_tool_use.name = "retrieve_docs"
            mock_tool_use.input = {"query": "test query"}

            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            # Mock the streaming events for tool use
            mock_events = []
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            # Mock execute_tool
            with patch("stampy_chat.llms.execute_tool", return_value="Tool result"):
                result = list(call_anthropic(
                    sample_history, "anthropic/claude-3-5-sonnet-20241022", 1000,
                    tools=[RETRIEVE_DOCS_TOOL], settings=mock_settings, stream=True
                ))

            # Verify tools were passed to API
            call_args = mock_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert len(call_args["tools"]) == 1

    @pytest.mark.recording
    def test_call_anthropic_fallback_to_google(self, sample_history, mock_settings):
        """Test fallback to Google when Anthropic fails"""
        with patch("stampy_chat.llms.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock rate limit error
            import anthropic
            mock_client.messages.create.side_effect = anthropic.RateLimitError("Rate limited")

            with patch("stampy_chat.llms.call_google") as mock_call_google:
                mock_call_google.return_value = iter([LLMChunk(type="response", text="Fallback")])

                result = list(call_anthropic(
                    sample_history, "anthropic/claude-3-5-sonnet-20241022", 1000
                ))

                assert len(result) == 1
                assert result[0]["text"] == "Fallback"
                mock_call_google.assert_called_once()


@pytest.mark.recording
class TestOpenAIProvider:
    @pytest.mark.recording
    def test_call_openai_basic(self, sample_history):
        """Test basic OpenAI response"""
        with patch("stampy_chat.llms.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_client.responses.create.return_value = mock_response

            # Mock streaming events
            mock_events = [
                Mock(type="response.output_text.delta", delta="Hello"),
                Mock(type="response.output_text.delta", delta=" world"),
            ]
            mock_response.__iter__ = Mock(return_value=iter(mock_events))

            result = list(call_openai(
                sample_history, "o1-preview", 1000, stream=True
            ))

            assert len(result) == 2
            assert result[0]["type"] == "response"
            assert result[0]["text"] == "Hello"

    @pytest.mark.recording
    def test_call_openai_with_thinking(self, sample_history):
        """Test OpenAI with reasoning"""
        with patch("stampy_chat.llms.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_client.responses.create.return_value = mock_response

            result = call_openai(
                sample_history, "o1-preview", 1000, thinking_budget=1000
            )

            # Verify reasoning was enabled
            call_args = mock_client.responses.create.call_args[1]
            assert "reasoning" in call_args
            assert call_args["reasoning"]["effort"] == "medium"

    @pytest.mark.recording
    def test_call_openai_tools_not_implemented(self, sample_history):
        """Test that OpenAI tools raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Tool use is not yet implemented"):
            list(call_openai(
                sample_history, "gpt-4", 1000, tools=[RETRIEVE_DOCS_TOOL]
            ))


@pytest.mark.recording
class TestGoogleProvider:
    @pytest.mark.recording
    def test_call_google_basic(self, sample_history):
        """Test basic Google response"""
        with patch("stampy_chat.llms.genai.Client") as mock_genai_class:
            mock_client = Mock()
            mock_genai_class.return_value = mock_client

            # Mock streaming response
            mock_chunk1 = Mock()
            mock_chunk1.text = "Hello"
            mock_chunk2 = Mock()
            mock_chunk2.text = " world"

            mock_client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]

            result = list(call_google(
                sample_history, "gemini-1.5-pro", 1000, stream=True
            ))

            assert len(result) == 2
            assert result[0]["type"] == "response"
            assert result[0]["text"] == "Hello"

    @pytest.mark.recording
    def test_call_google_with_system(self, sample_history):
        """Test Google with system instruction"""
        with patch("stampy_chat.llms.genai.Client") as mock_genai_class:
            mock_client = Mock()
            mock_genai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            call_google(sample_history, "gemini-1.5-pro", 1000, stream=False)

            # Verify system instruction was passed
            call_args = mock_client.models.generate_content.call_args
            config = call_args[1]["config"]
            assert hasattr(config, "system_instruction")

    @pytest.mark.recording
    def test_call_google_tools_not_implemented(self, sample_history):
        """Test that Google tools raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Tool use is not yet implemented"):
            list(call_google(
                sample_history, "gemini-1.5-pro", 1000, tools=[RETRIEVE_DOCS_TOOL]
            ))


@pytest.mark.recording
class TestOpenRouterProvider:
    @pytest.mark.recording
    def test_call_openrouter_basic(self, sample_history):
        """Test basic OpenRouter response"""
        with patch("stampy_chat.llms.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock streaming response
            mock_choice = Mock()
            mock_choice.delta = Mock()
            mock_choice.delta.content = "Hello world"
            mock_chunk = Mock()
            mock_chunk.choices = [mock_choice]

            mock_response = [mock_chunk]
            mock_client.chat.completions.create.return_value = mock_response

            result = list(call_openrouter(
                sample_history, "openrouter/anthropic/claude-3-haiku", 1000, stream=True
            ))

            assert len(result) == 1
            assert result[0]["type"] == "response"
            assert result[0]["text"] == "Hello world"

    @pytest.mark.recording
    def test_call_openrouter_with_reasoning(self, sample_history):
        """Test OpenRouter with reasoning tokens"""
        with patch("stampy_chat.llms.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = []
            mock_client.chat.completions.create.return_value = mock_response

            call_openrouter(
                sample_history, "openrouter/deepseek/deepseek-r1", 1000,
                thinking_budget=2000
            )

            # Verify reasoning was added to extra_body
            call_args = mock_client.chat.completions.create.call_args[1]
            assert "extra_body" in call_args
            assert call_args["extra_body"]["reasoning"]["max_tokens"] == 2000

    @pytest.mark.recording
    def test_call_openrouter_tools_not_implemented(self, sample_history):
        """Test that OpenRouter tools raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Tool use is not yet implemented"):
            list(call_openrouter(
                sample_history, "openrouter/anthropic/claude-3-haiku", 1000,
                tools=[RETRIEVE_DOCS_TOOL]
            ))


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
    @pytest.mark.recording
    def test_query_llm_anthropic(self, sample_history, mock_settings):
        """Test query_llm with Anthropic provider"""
        with patch("stampy_chat.llms.call_anthropic") as mock_call:
            mock_call.return_value = iter([LLMChunk(type="response", text="Test")])

            result = list(query_llm(sample_history, mock_settings))

            assert len(result) == 1
            assert result[0]["text"] == "Test"
            mock_call.assert_called_once()

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
                model="anthropic/claude-3-5-sonnet-20241022",
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
            assert len(search_calls) == 2, f"Expected exactly 2 search calls, got {len(search_calls)}: {search_calls}"

            # 2. Verify searches returned results (implicitly tested by mock)

            # 3. Verify second search query is a substring of first search results
            first_search_results_text = ("AI alignment refers to ensuring artificial intelligence systems pursue intended goals. Key challenges include reward hacking, mesa-optimization, and value learning. Research focuses on interpretability methods. "
                                       "Current safety research investigates robustness testing, adversarial examples, and capability control mechanisms.")

            second_query = search_calls[1]
            assert second_query in first_search_results_text, f"Second search query '{second_query}' not found in first search results"

            # 4. Verify output format matches expected pattern
            expected_pattern = r"First search was: (.+?)\nSecond search was: (.+?)\nPhrase from the second search results: (.+)"
            match = re.search(expected_pattern, full_response, re.DOTALL)

            assert match is not None, f"Response doesn't match expected format. Got: {full_response}"

            # Extract the parts from the response
            reported_first_search = match.group(1).strip()
            reported_second_search = match.group(2).strip()
            reported_phrase = match.group(3).strip()

            # Verify reported searches match actual searches
            assert reported_first_search == search_calls[0], f"Reported first search '{reported_first_search}' doesn't match actual '{search_calls[0]}'"
            assert reported_second_search == search_calls[1], f"Reported second search '{reported_second_search}' doesn't match actual '{search_calls[1]}'"

    @pytest.mark.recording
    def test_query_llm_openai(self, sample_history):
        """Test query_llm with OpenAI provider"""
        settings = Settings(model_provider=OPENAI, model="o1-preview")

        with patch("stampy_chat.llms.call_openai") as mock_call:
            mock_call.return_value = iter([LLMChunk(type="response", text="Test")])

            result = list(query_llm(sample_history, settings))

            assert len(result) == 1
            mock_call.assert_called_once()

    @pytest.mark.recording
    def test_query_llm_google(self, sample_history):
        """Test query_llm with Google provider"""
        settings = Settings(model_provider=GOOGLE, model="gemini-1.5-pro")

        with patch("stampy_chat.llms.call_google") as mock_call:
            mock_call.return_value = iter([LLMChunk(type="response", text="Test")])

            result = list(query_llm(sample_history, settings))

            assert len(result) == 1
            mock_call.assert_called_once()

    @pytest.mark.recording
    def test_query_llm_openrouter(self, sample_history):
        """Test query_llm with OpenRouter provider"""
        settings = Settings(model_provider=OPENROUTER, model="openrouter/anthropic/claude-3-haiku")

        with patch("stampy_chat.llms.call_openrouter") as mock_call:
            mock_call.return_value = iter([LLMChunk(type="response", text="Test")])

            result = list(query_llm(sample_history, settings))

            assert len(result) == 1
            mock_call.assert_called_once()

    def test_query_llm_unknown_provider(self, sample_history):
        """Test query_llm with unknown provider"""
        settings = Settings(model_provider="unknown", model="test")

        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            list(query_llm(sample_history, settings))

    @pytest.mark.recording
    def test_query_llm_with_thinking_budget(self, sample_history, mock_settings):
        """Test query_llm with thinking budget and model that supports thinking"""
        # Mock a model that supports thinking
        with patch("stampy_chat.llms.MODELS") as mock_models:
            mock_model_info = Mock()
            mock_model_info.can_think = True
            mock_model_info.min_think = 1024
            mock_models.__getitem__.return_value = mock_model_info

            with patch("stampy_chat.llms.call_anthropic") as mock_call:
                mock_call.return_value = iter([])

                list(query_llm(sample_history, mock_settings, thinking_budget=500))

                # Should use minimum thinking budget
                args = mock_call.call_args[0]
                thinking_budget_arg = args[3]  # thinking_budget is 4th positional arg
                assert thinking_budget_arg == 1024

    @pytest.mark.recording
    def test_query_llm_with_tools_and_context(self, sample_history, mock_settings):
        """Test query_llm with tools and conversation context"""
        mock_context = Mock()
        mock_callbacks = [Mock()]

        with patch("stampy_chat.llms.call_anthropic") as mock_call:
            mock_call.return_value = iter([])

            list(query_llm(
                sample_history, mock_settings,
                tools=[RETRIEVE_DOCS_TOOL],
                conversation_context=mock_context,
                callbacks=mock_callbacks
            ))

            # Verify tools, context, and callbacks were passed through
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["tools"] == [RETRIEVE_DOCS_TOOL]
            assert call_kwargs["conversation_context"] is mock_context
            assert call_kwargs["callbacks"] is mock_callbacks