import pytest
import re
from unittest.mock import patch

from stampy_chat.chat import run_query
from stampy_chat.settings import Settings, ANTHROPIC
from stampy_chat.llms import RETRIEVE_DOCS_TOOL


def test_chat_end_to_end_with_tool_mode():
    """End-to-end test for chat functionality with tool mode enabled"""

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

    # Create settings object
    settings = Settings(**test_config)

    # Verify settings were created correctly
    assert settings.tool_mode == True
    assert settings.thinking_budget == 1024
    assert settings.mode == "concise"
    assert settings.model == "anthropic/claude-sonnet-4-20250514"
    assert settings.model_provider == ANTHROPIC
    assert settings.topKBlocks == 50
    assert settings.maxNumTokens == 200000

    # Run the chat query end-to-end (this will use real tools and record everything)
    result = run_query(
        session_id="test_session",
        query=test_query,
        history=test_history,
        settings=settings,
        callback=None,
        followups=False  # Disable followups to simplify test
    )

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

    result = run_query(
        session_id="test_thinking",
        query="Explain alignment problems in AI",
        history=[],
        settings=settings,
        callback=None,
        followups=False
    )

    # Response should exist (thinking budget should be used if model supports it)
    assert "response" in result
    assert len(result["response"].strip()) > 0


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