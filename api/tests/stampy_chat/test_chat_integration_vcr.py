"""
Comprehensive VCR integration test for the full chat.py endpoint.

This test validates end-to-end functionality with all three external APIs:
- Voyage AI (for embeddings in retrieve_docs)
- Pinecone (for vector search in retrieve_docs)
- Anthropic (for LLM response generation)

Uses pytest-recording to record API interactions for fast, reliable replays.

RECORDING INSTRUCTIONS:
-----------------------
To record a new cassette (requires real API keys and network access):

1. Ensure you have all three API keys in your environment:
   export ANTHROPIC_API_KEY_DEV="your-key"
   export PINECONE_API_KEY="your-key"
   export VOYAGEAI_API_KEY="your-key"

2. Run the test in record mode:
   cd api
   pipenv run pytest tests/stampy_chat/test_chat_integration_vcr.py -v --no-cov --record-mode=rewrite

3. The cassette will be created at:
   tests/cassettes/test_stampy_chat/test_chat_integration_vcr/test_run_query_end_to_end_with_vcr.yaml

4. Verify the cassette was created and all API keys are REDACTED

5. Test replay without API keys:
   ANTHROPIC_API_KEY_DEV="" PINECONE_API_KEY="" VOYAGEAI_API_KEY="" \
     pipenv run pytest tests/stampy_chat/test_chat_integration_vcr.py -v --no-cov

REPLAY MODE:
------------
Once the cassette is recorded, the test can be run without API keys or network access.
VCR will replay the recorded interactions from the cassette file.
"""
import os
import sys
import pytest

# Store real API keys for later
real_pinecone_key = os.environ.get("PINECONE_API_KEY")
real_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_DEV")
real_voyage_key = os.environ.get("VOYAGEAI_API_KEY") or os.environ.get("VOYAGE_API_KEY")

# TEMPORARILY unset PINECONE_API_KEY during import to prevent initialization
# in env.py (which checks `if PINECONE_API_KEY:` before initializing)
if "PINECONE_API_KEY" in os.environ:
    del os.environ["PINECONE_API_KEY"]

# Set minimal keys for other services
os.environ.setdefault("ANTHROPIC_API_KEY", real_anthropic_key or "sk-ant-dummy")
os.environ.setdefault("VOYAGEAI_API_KEY", real_voyage_key or "dummy-voyage")

# Now import the modules - Pinecone won't initialize
from stampy_chat.chat import run_query
from stampy_chat.settings import Settings

# Restore PINECONE_API_KEY after import for actual test usage
if real_pinecone_key:
    os.environ["PINECONE_API_KEY"] = real_pinecone_key
else:
    os.environ["PINECONE_API_KEY"] = "dummy-pinecone-key"


@pytest.mark.vcr
def test_run_query_end_to_end_with_vcr():
    """Test the full run_query function end-to-end with all three APIs.

    This test validates:
    1. Voyage API call for query embedding
    2. Pinecone API call for vector search
    3. Anthropic API call for LLM response generation
    4. All callback events are fired in the correct order
    5. Result structure matches what the webclient expects
    6. All event types match the webclient's expectations

    We use minimal settings to keep the test fast and the cassette small:
    - enable_hyde = False (skip HyDE generation to reduce API calls)
    - topKBlocks = 1 (minimize Pinecone results)
    - thinking_budget = 0 (no thinking to keep response short)
    - max_tokens = 100 (short response)

    The cassette will be created at:
    tests/cassettes/test_stampy_chat/test_chat_integration_vcr/test_run_query_end_to_end_with_vcr.yaml
    """
    # Create minimal settings to keep test fast and cassette small
    settings = Settings(
        completions="anthropic/claude-sonnet-4-20250514",
        enable_hyde=False,  # Skip HyDE to reduce API calls
        topKBlocks=1,  # Minimize Pinecone results
        thinking_budget=0,  # No thinking for simplicity
        min_response_tokens=10,
        maxNumTokens=8000,
    )

    # Collect all callback events
    events = []

    def callback(event):
        if event is not None:
            events.append(event)

    # Use a very simple query
    query = "What is AI safety?"
    history = []  # Empty history to keep it simple

    # Run the full query
    result = run_query(
        session_id="test-vcr-session",
        query=query,
        history=history,
        settings=settings,
        callback=callback,
        followups=False,  # Disable followups to keep test simple
    )

    # ========================================
    # Validate the result structure
    # ========================================
    assert "response" in result, "Result should have 'response' key"
    assert isinstance(result["response"], str), "Response should be a string"
    assert len(result["response"]) > 0, "Response should not be empty"

    assert "followups" in result, "Result should have 'followups' key"
    assert isinstance(result["followups"], list), "Followups should be a list"
    assert len(result["followups"]) == 0, "Followups should be empty since followups=False"

    # ========================================
    # Validate callback events
    # ========================================
    # Extract event states for easier assertion
    event_states = [e.get("state") for e in events if "state" in e]

    # We should have these events (in order):
    # 1. citations (from on_citations_retrieved)
    # 2. loading/prompt (from on_citations_retrieved)
    # 3. prompt (from on_prompt)
    # 4. loading/llm (from on_llm_start)
    # 5. streaming (one or more, from on_response)
    # 6. done (at the end)

    assert "citations" in event_states, "Should have 'citations' event"
    assert "prompt" in event_states, "Should have 'prompt' event"
    assert "loading" in event_states, "Should have 'loading' event(s)"
    assert "streaming" in event_states, "Should have 'streaming' event(s)"
    assert "done" in event_states, "Should have 'done' event"

    # Verify we got citations
    citations_events = [e for e in events if e.get("state") == "citations"]
    assert len(citations_events) > 0, "Should have at least one citations event"
    assert "citations" in citations_events[0], "Citations event should have 'citations' key"
    assert isinstance(citations_events[0]["citations"], list), "Citations should be a list"
    assert len(citations_events[0]["citations"]) > 0, "Should have at least one citation"

    # Verify we got the prompt
    prompt_events = [e for e in events if e.get("state") == "prompt"]
    assert len(prompt_events) > 0, "Should have at least one prompt event"
    assert "promptedHistory" in prompt_events[0], "Prompt event should have 'promptedHistory' key"
    assert isinstance(prompt_events[0]["promptedHistory"], list), "Prompted history should be a list"

    # Verify we got streaming content chunks
    streaming_events = [e for e in events if e.get("state") == "streaming"]
    assert len(streaming_events) > 0, "Should have at least one streaming event"

    # Verify each streaming event has content and it's a string
    for event in streaming_events:
        assert "content" in event, "Streaming event should have 'content' key"
        assert isinstance(event["content"], str), "Streaming content should be a string"
        assert len(event["content"]) > 0, "Streaming content should not be empty"

    # Verify we got done event
    done_events = [e for e in events if e.get("state") == "done"]
    assert len(done_events) == 1, "Should have exactly one 'done' event"

    # Verify the full response matches what was streamed
    streamed_text = "".join(e["content"] for e in streaming_events)
    assert streamed_text in result["response"], "Streamed text should be part of final response"

    # ========================================
    # Validate event ordering
    # ========================================
    # Find first occurrence of each state
    citations_idx = next(i for i, e in enumerate(events) if e.get("state") == "citations")
    prompt_idx = next(i for i, e in enumerate(events) if e.get("state") == "prompt")
    first_streaming_idx = next(i for i, e in enumerate(events) if e.get("state") == "streaming")
    done_idx = next(i for i, e in enumerate(events) if e.get("state") == "done")

    # Verify order: citations -> prompt -> streaming -> done
    assert citations_idx < prompt_idx, "Citations should come before prompt"
    assert prompt_idx < first_streaming_idx, "Prompt should come before streaming"
    assert first_streaming_idx < done_idx, "Streaming should come before done"

    # ========================================
    # Validate response content
    # ========================================
    # The response should be a meaningful answer about AI safety
    # (being flexible since we're using a simple prompt and short max_tokens)
    response_lower = result["response"].lower()
    # Should mention something related to AI, safety, risk, alignment, etc.
    ai_related_keywords = [
        "ai", "artificial intelligence", "safety", "risk", "alignment",
        "control", "human", "technology", "system", "future"
    ]
    assert any(keyword in response_lower for keyword in ai_related_keywords), \
        f"Response should contain AI safety related content. Got: {result['response'][:200]}"

    print(f"\n✓ Test passed!")
    print(f"  - Got {len(events)} total callback events")
    print(f"  - Got {len(streaming_events)} streaming chunks")
    print(f"  - Got {len(citations_events[0]['citations'])} citations")
    print(f"  - Response length: {len(result['response'])} characters")
    print(f"  - Response preview: {result['response'][:100]}...")
