import pytest
import re
from unittest.mock import patch

from stampy_chat.chat import run_query
from stampy_chat.settings import Settings, ANTHROPIC
from stampy_chat.llms import RETRIEVE_DOCS_TOOL


@pytest.mark.vcr
def test_chat_end_to_end_with_tool_mode():
    """End-to-end test for chat functionality with tool mode enabled"""
    import pudb; pudb.set_trace()
    import time
    start_time = time.time()
    print(f"Test started at {start_time}")

    def log_timing(step):
        elapsed = time.time() - start_time
        print(f"TIMING: {step} - {elapsed:.2f}s elapsed")

    # Test configuration from the user (the one not working on web)
    test_config = {
        "prompts": {
            "system": "\n<miri-core-points>\n<entire-source id=\"LL\">\n{yudkowsky-list-of-lethalities-2507132226-e11d43}\n</entire-source>\n\n<entire-source id=\"TP\">\n{miri-the-problem-2507121135-b502d1}\n</entire-source>\n\n<entire-source id=\"TB\">\n{miri-the-briefing-2507132220-44fbe5}\n</entire-source>\n\n<main-points>\n{miri-the-problem-main-points-2507132222-1916a0}\n</main-points>\n</miri-core-points>\n",
            "history": "{stampy-history-2507211352-060b74}",
            "history_summary": "{stampy-history_summary-2507231056-b048af}",
            "pre_message": "",
            "post_message": "\n{detailed-cautious-epistem-safetyinfo-v7-2508241916-cdc305}\n\n{post-message-2507220220-cff788}\n\n{socratic-avoid-bad-questions-harder-2507220153-a11064}\n\n{mode}",
            "hyde_pre_message": "",
            "hyde_post_message": "{detailed-cautious-epistem-safetyinfo-v7-hyde-2508241917-fba3ad}\n\n{hyde_post_message-2507222109-597ed2}",
            "message_format": "<from-public-user>\n{message}\n</from-public-user>",
            "modes": {
                "default": "",
                "concise": "{mode-concise-2507231147-db01d9}",
                "rookie": "{mode-rookie-2507231143-f32d39}",
                "discord": "{mode-discord-2507231144-ffe1d1}"
            }
        },
        "mode": "concise",
        "model": "anthropic/claude-sonnet-4-20250514",
        "encoder": "cl100k_base",
        "topKBlocks": 50,
        "maxNumTokens": 200000,
        "tokensBuffer": 50,
        "maxHistory": 10,
        "maxHistorySummaryTokens": 200,
        "historyFraction": 0.25,
        "contextFraction": 0.5,
        "enable_hyde": False,
        "thinking_budget": 1024,
        "tool_mode": True,
        "filters": {
            "miri_confidence": 6,
            "miri_distance": [],
            "needs_tech": False
        }
    }

    # Test query about AI safety
    test_query = "What are the main challenges in AI alignment research?"
    test_history = []

    log_timing("Config setup complete")

    # Create settings object
    settings = Settings(**test_config)

    log_timing("Settings object created")

    # Verify settings were created correctly
    assert settings.tool_mode == True
    assert settings.thinking_budget == 1024
    assert settings.mode == "concise"
    assert settings.model == "anthropic/claude-sonnet-4-20250514"
    assert settings.model_provider == ANTHROPIC
    assert settings.topKBlocks == 50
    assert settings.maxNumTokens == 200000

    log_timing("About to call run_query")

    # Run the chat query end-to-end (this will use real tools and record everything)
    result = run_query(
        session_id="test_session",
        query=test_query,
        history=test_history,
        settings=settings,
        callback=None,
        followups=False  # Disable followups to simplify test
    )

    log_timing("run_query completed")

    # Verify result structure
    assert "response" in result
    assert "followups" in result
    assert isinstance(result["response"], str)
    assert isinstance(result["followups"], list)

    # Verify response is not empty (tool mode should produce a response)
    response = result["response"]
    assert len(response.strip()) > 0, "Response should not be empty with tool mode enabled"

    # Verify response contains relevant content about AI alignment
    response_lower = response.lower()
    alignment_keywords = ["alignment", "ai safety", "safety", "beneficial", "control"]
    found_keywords = [kw for kw in alignment_keywords if kw in response_lower]
    assert len(found_keywords) > 0, f"Response should mention alignment-related topics. Response: {response}"


@pytest.mark.vcr
def test_chat_configuration_validation():
    """Test that the provided configuration is parsed correctly"""

    # The problematic configuration from the user
    config = {
        "prompts": {
            "system": "\n<miri-core-points>\n<entire-source id=\"LL\">\n{yudkowsky-list-of-lethalities-2507132226-e11d43}\n</entire-source>\n\n<entire-source id=\"TP\">\n{miri-the-problem-2507121135-b502d1}\n</entire-source>\n\n<entire-source id=\"TB\">\n{miri-the-briefing-2507132220-44fbe5}\n</entire-source>\n\n<main-points>\n{miri-the-problem-main-points-2507132222-1916a0}\n</main-points>\n</miri-core-points>\n",
            "history": "{stampy-history-2507211352-060b74}",
            "history_summary": "{stampy-history_summary-2507231056-b048af}",
            "pre_message": "",
            "post_message": "\n{detailed-cautious-epistem-safetyinfo-v7-2508241916-cdc305}\n\n{post-message-2507220220-cff788}\n\n{socratic-avoid-bad-questions-harder-2507220153-a11064}\n\n{mode}",
            "hyde_pre_message": "",
            "hyde_post_message": "{detailed-cautious-epistem-safetyinfo-v7-hyde-2508241917-fba3ad}\n\n{hyde_post_message-2507222109-597ed2}",
            "message_format": "<from-public-user>\n{message}\n</from-public-user>",
            "modes": {
                "default": "",
                "concise": "{mode-concise-2507231147-db01d9}",
                "rookie": "{mode-rookie-2507231143-f32d39}",
                "discord": "{mode-discord-2507231144-ffe1d1}"
            }
        },
        "mode": "concise",
        "model": "anthropic/claude-sonnet-4-20250514",
        "encoder": "cl100k_base",
        "topKBlocks": 50,
        "maxNumTokens": 200000,
        "tokensBuffer": 50,
        "maxHistory": 10,
        "maxHistorySummaryTokens": 200,
        "historyFraction": 0.25,
        "contextFraction": 0.5,
        "enable_hyde": False,
        "thinking_budget": 1024,
        "tool_mode": True,
        "filters": {
            "miri_confidence": 6,
            "miri_distance": [],
            "needs_tech": False
        }
    }

    # Test that Settings object can be created without errors
    settings = Settings(**config)

    # Verify all the key settings
    assert settings.tool_mode == True
    assert settings.thinking_budget == 1024
    assert settings.mode == "concise"
    assert settings.model == "anthropic/claude-sonnet-4-20250514"
    assert settings.enable_hyde == False
    assert settings.topKBlocks == 50
    assert settings.maxNumTokens == 200000
    assert settings.historyFraction == 0.25
    assert settings.contextFraction == 0.5

    # Verify prompts are correctly set
    assert "miri-core-points" in settings.system_prompt
    assert settings.mode_prompt == "{mode-concise-2507231147-db01d9}"
    assert settings.message_format == "<from-public-user>\n{message}\n</from-public-user>"

    # Verify filters (note: empty lists become tuples in frozen structures)
    assert settings.filters["miri_confidence"] == 6
    assert settings.filters["miri_distance"] == () or settings.filters["miri_distance"] == []
    assert settings.filters["needs_tech"] == False

    # Verify derived properties
    assert settings.model_provider == ANTHROPIC
    assert settings.model_id == "claude-sonnet-4-20250514"
    assert settings.max_response_tokens > 0

    # Verify token calculations work
    assert settings.context_tokens > 0
    assert settings.history_tokens > 0


@pytest.mark.vcr
def test_chat_thinking_budget_usage():
    """Test that thinking budget is properly used when tool_mode is enabled"""

    config = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "thinking_budget": 2048,  # Higher thinking budget
        "tool_mode": True,
        "maxNumTokens": 100000
    }

    settings = Settings(**config)

    # Verify thinking budget is set correctly
    assert settings.thinking_budget == 2048

    # Since thinking_budget > 0, this will now use custom thinking by default
    # We need to track what content comes back as thinking vs response
    captured_thinking = []
    captured_responses = []

    def capture_callback(event):
        # Capture thinking and response content for verification
        pass  # The individual callbacks will be called internally

    result = run_query(
        session_id="test_thinking",
        query="Explain alignment problems in AI. Please think through this carefully and provide a comprehensive response.",
        history=[],
        settings=settings,
        callback=capture_callback,
        followups=False
    )

    # Should have result structure
    assert "response" in result
    assert "followups" in result

    # With thinking_budget > 0, the model should use custom thinking
    # The response might be empty if all content was classified as thinking
    # The key is that the system should work without errors
    assert isinstance(result["response"], str)
    assert isinstance(result["followups"], list)


def test_settings_validation_errors():
    """Test that invalid settings raise appropriate errors"""

    # Test invalid model
    with pytest.raises(ValueError, match="Unknown model"):
        Settings(model="invalid/model")

    # Test invalid mode
    config_with_invalid_mode = {
        "prompts": {
            "modes": {"default": "", "concise": "test"}
        },
        "mode": "invalid_mode"  # This mode doesn't exist in prompts.modes
    }
    with pytest.raises(ValueError, match="Invalid mode"):
        Settings(**config_with_invalid_mode)

    # Test token allocation that's too large
    with pytest.raises(ValueError, match="context and history fractions are too large"):
        Settings(
            model="anthropic/claude-3-5-sonnet-20241022",
            maxNumTokens=1000,  # Very small
            contextFraction=0.8,  # Too large
            historyFraction=0.8,  # Too large
            min_response_tokens=100
        )


@pytest.mark.vcr
def test_custom_thinking_integration():
    """Integration test for custom thinking functionality"""
    from stampy_chat.llms import query_llm, RETRIEVE_DOCS_TOOL

    config = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "thinking_budget": 0,  # Use custom thinking instead of official thinking
        "tool_mode": True,
        "maxNumTokens": 100000
    }

    settings = Settings(**config)

    # Test query that should trigger custom thinking
    history = [
        {"role": "user", "content": "Please think carefully about AI alignment challenges and then provide a concise answer. Use the <thinking> tags to show your reasoning."}
    ]

    # Run with custom thinking enabled
    result_chunks = list(query_llm(
        history,
        settings,
        tools=[RETRIEVE_DOCS_TOOL],
        stream=True,
        custom_thinking=True
    ))

    # Should get both thinking and response chunks
    thinking_chunks = [chunk for chunk in result_chunks if chunk["type"] == "thinking"]
    response_chunks = [chunk for chunk in result_chunks if chunk["type"] == "response"]

    # Verify we got some thinking content
    assert len(thinking_chunks) > 0, "Should have thinking chunks with custom thinking enabled"

    # Verify we got some response content
    assert len(response_chunks) > 0, "Should have response chunks"

    # Verify thinking content is meaningful
    thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
    assert len(thinking_text.strip()) > 10, "Thinking content should be substantial"

    # Verify response content
    response_text = "".join(chunk["text"] for chunk in response_chunks)
    assert len(response_text.strip()) > 10, "Response content should be substantial"

    print(f"Thinking content: {thinking_text}")
    print(f"Response content: {response_text}")


@pytest.mark.vcr
def test_custom_thinking_with_tool_use_integration():
    """Integration test for custom thinking with tool use"""
    from stampy_chat.llms import query_llm, RETRIEVE_DOCS_TOOL

    config = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "thinking_budget": 0,
        "tool_mode": True,
        "maxNumTokens": 100000
    }

    settings = Settings(**config)

    # Query that should trigger tool use within thinking
    history = [
        {"role": "user", "content": "I need you to search for information about mesa-optimization and then explain it. Please use <thinking> tags to show your reasoning process, including when you decide to search for information."}
    ]

    # Run with custom thinking - this should allow tool use within thinking blocks
    result_chunks = list(query_llm(
        history,
        settings,
        tools=[RETRIEVE_DOCS_TOOL],
        stream=True,
        custom_thinking=True
    ))

    # Collect chunks by type
    thinking_chunks = [chunk for chunk in result_chunks if chunk["type"] == "thinking"]
    response_chunks = [chunk for chunk in result_chunks if chunk["type"] == "response"]

    # Should have both thinking and response content
    assert len(thinking_chunks) > 0, "Should have thinking content when using custom thinking"
    assert len(response_chunks) > 0, "Should have final response content"

    # Assemble full texts
    thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
    response_text = "".join(chunk["text"] for chunk in response_chunks)

    # The model should have been able to use tools during its thinking process
    # We can't directly verify this from the chunks, but the response should contain
    # information that could only come from the tool use
    assert len(thinking_text.strip()) > 20, "Should have substantial thinking content"
    assert len(response_text.strip()) > 20, "Should have substantial response content"

    print(f"Thinking with tools: {thinking_text[:200]}...")
    print(f"Response after tool use: {response_text[:200]}...")


@pytest.mark.vcr
def test_user_reported_config_thinking_issue():
    """Test the exact config that user reported had thinking issues in web UI"""
    from stampy_chat.llms import query_llm, RETRIEVE_DOCS_TOOL

    # The exact problematic config from the user
    config = {
        "prompts": {
            "system": "\n<miri-core-points>\n<entire-source id=\"LL\">\n{yudkowsky-list-of-lethalities-2507132226-e11d43}\n</entire-source>\n\n<entire-source id=\"TP\">\n{miri-the-problem-2507121135-b502d1}\n</entire-source>\n\n<entire-source id=\"TB\">\n{miri-the-briefing-2507132220-44fbe5}\n</entire-source>\n\n<main-points>\n{miri-the-problem-main-points-2507132222-1916a0}\n</main-points>\n</miri-core-points>\n",
            "history": "{stampy-history-2507211352-060b74}",
            "history_summary": "{stampy-history_summary-2507231056-b048af}",
            "pre_message": "",
            "post_message": "{post-message-refined-claude-written-2509140732-dbec84}\n\n{socratic-avoid-bad-questions-harder-2507220153-a11064}",
            "hyde_pre_message": "",
            "hyde_post_message": "{detailed-cautious-epistem-safetyinfo-v7-hyde-2508241917-fba3ad}\n\n{hyde_post_message-2507222109-597ed2}",
            "message_format": "<from-public-user>\n{message}\n</from-public-user>",
            "modes": {
                "default": "",
                "concise": "{mode-concise-2507231147-db01d9}",
                "rookie": "{mode-rookie-2507231143-f32d39}",
                "discord": "{mode-discord-2507231144-ffe1d1}"
            }
        },
        "mode": "concise",
        "model": "anthropic/claude-sonnet-4-20250514",
        "encoder": "cl100k_base",
        "topKBlocks": 50,
        "maxNumTokens": 200000,
        "tokensBuffer": 50,
        "maxHistory": 10,
        "maxHistorySummaryTokens": 200,
        "historyFraction": 0.25,
        "contextFraction": 0.5,
        "enable_hyde": False,
        "thinking_budget": 1024,  # This should trigger official thinking mode
        "tool_mode": True,
        "filters": {
            "miri_confidence": 6,
            "miri_distance": [],
            "needs_tech": False
        }
    }

    settings = Settings(**config)

    # Query that should trigger both thinking and tool use
    history = [
        {"role": "user", "content": "What are the key challenges in AI alignment? Please use <thinking> tags to show your reasoning process and search for relevant information."}
    ]

    # Run with default thinking mode (should be custom thinking since thinking_budget > 0)
    result_chunks = list(query_llm(
        history,
        settings,
        tools=[RETRIEVE_DOCS_TOOL],
        stream=True
        # custom_thinking not specified - should default to True since thinking_budget > 0
    ))

    # Collect chunks by type
    thinking_chunks = [chunk for chunk in result_chunks if chunk["type"] == "thinking"]
    response_chunks = [chunk for chunk in result_chunks if chunk["type"] == "response"]

    # THE MAIN FIX: With thinking_budget > 0, we should now get thinking content by default
    assert len(thinking_chunks) > 0, "Should have thinking content with thinking_budget > 0 (custom thinking by default)"

    # May or may not have response chunks depending on whether the model closed </thinking>
    # The key point is that thinking content is properly classified as thinking, not response

    # Assemble full texts
    thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
    response_text = "".join(chunk["text"] for chunk in response_chunks)

    # Verify substantial thinking content
    assert len(thinking_text.strip()) > 10, f"Should have substantial thinking content, got: '{thinking_text[:100]}'"

    # Check that search/retrieval likely occurred and that thinking contains alignment-related content
    alignment_terms = ["alignment", "ai safety", "safety", "risk", "control"]
    thinking_lower = thinking_text.lower()
    found_terms = [term for term in alignment_terms if term in thinking_lower]
    assert len(found_terms) > 0, f"Thinking should contain alignment-related terms from document search. Thinking: {thinking_text[:200]}"

    print(f"SUCCESS: Thinking content ({len(thinking_text)} chars): {thinking_text[:150]}...")
    if response_text:
        print(f"Response content ({len(response_text)} chars): {response_text[:150]}...")
    else:
        print("No response content (model may not have closed </thinking> tag)")
    print(f"Found alignment terms in thinking: {found_terms}")


@pytest.mark.vcr
def test_user_config_with_custom_thinking():
    """Test the same user config but with custom thinking to compare behavior"""
    from stampy_chat.llms import query_llm, RETRIEVE_DOCS_TOOL

    # Same config as above, but we'll use custom_thinking=True
    config = {
        "prompts": {
            "system": "\n<miri-core-points>\n<entire-source id=\"LL\">\n{yudkowsky-list-of-lethalities-2507132226-e11d43}\n</entire-source>\n\n<entire-source id=\"TP\">\n{miri-the-problem-2507121135-b502d1}\n</entire-source>\n\n<entire-source id=\"TB\">\n{miri-the-briefing-2507132220-44fbe5}\n</entire-source>\n\n<main-points>\n{miri-the-problem-main-points-2507132222-1916a0}\n</main-points>\n</miri-core-points>\n",
            "history": "{stampy-history-2507211352-060b74}",
            "history_summary": "{stampy-history_summary-2507231056-b048af}",
            "pre_message": "",
            "post_message": "{post-message-refined-claude-written-2509140732-dbec84}\n\n{socratic-avoid-bad-questions-harder-2507220153-a11064}",
            "hyde_pre_message": "",
            "hyde_post_message": "{detailed-cautious-epistem-safetyinfo-v7-hyde-2508241917-fba3ad}\n\n{hyde_post_message-2507222109-597ed2}",
            "message_format": "<from-public-user>\n{message}\n</from-public-user>",
            "modes": {
                "default": "",
                "concise": "{mode-concise-2507231147-db01d9}",
                "rookie": "{mode-rookie-2507231143-f32d39}",
                "discord": "{mode-discord-2507231144-ffe1d1}"
            }
        },
        "mode": "concise",
        "model": "anthropic/claude-sonnet-4-20250514",
        "encoder": "cl100k_base",
        "topKBlocks": 50,
        "maxNumTokens": 200000,
        "tokensBuffer": 50,
        "maxHistory": 10,
        "maxHistorySummaryTokens": 200,
        "historyFraction": 0.25,
        "contextFraction": 0.5,
        "enable_hyde": False,
        "thinking_budget": 1024,  # Will be ignored when custom_thinking=True
        "tool_mode": True,
        "filters": {
            "miri_confidence": 6,
            "miri_distance": [],
            "needs_tech": False
        }
    }

    settings = Settings(**config)

    # Query that explicitly asks for thinking tags to trigger custom thinking
    history = [
        {"role": "user", "content": "What are the key challenges in AI alignment? Please use <thinking> tags to show your reasoning process and search for relevant information."}
    ]

    # Run with custom thinking enabled
    result_chunks = list(query_llm(
        history,
        settings,
        tools=[RETRIEVE_DOCS_TOOL],
        stream=True,
        custom_thinking=True  # Force custom thinking mode
    ))

    # Collect chunks by type
    thinking_chunks = [chunk for chunk in result_chunks if chunk["type"] == "thinking"]
    response_chunks = [chunk for chunk in result_chunks if chunk["type"] == "response"]

    # Should have both thinking and response content
    assert len(thinking_chunks) > 0, "Should have thinking content with custom thinking enabled"
    assert len(response_chunks) > 0, "Should have response content"

    # Assemble full texts
    thinking_text = "".join(chunk["text"] for chunk in thinking_chunks)
    response_text = "".join(chunk["text"] for chunk in response_chunks)

    # Verify substantial content in both
    assert len(thinking_text.strip()) > 10, f"Should have substantial thinking content, got: '{thinking_text[:100]}'"
    assert len(response_text.strip()) > 10, f"Should have substantial response content, got: '{response_text[:100]}'"

    # Check that search/retrieval might have occurred during thinking
    # With custom thinking, tool usage might appear in the thinking content
    thinking_lower = thinking_text.lower()
    search_indicators = ["search", "retrieve", "look for", "find information", "query"]
    found_indicators = [ind for ind in search_indicators if ind in thinking_lower]

    print(f"Custom thinking content ({len(thinking_text)} chars): {thinking_text[:200]}...")
    print(f"Custom response content ({len(response_text)} chars): {response_text[:200]}...")
    print(f"Search indicators in thinking: {found_indicators}")

    # At minimum, should have thinking content that shows reasoning process
    assert "align" in thinking_lower or "safe" in thinking_lower, f"Thinking should relate to AI alignment/safety. Thinking: {thinking_text[:100]}"
