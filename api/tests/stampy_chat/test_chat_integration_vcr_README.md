# VCR Integration Test for chat.py

## Overview

`test_chat_integration_vcr.py` is a comprehensive end-to-end integration test for the full `chat.py::run_query` function. It tests all three external APIs used by the Stampy chat system:

1. **Voyage AI** - For query embeddings in `retrieve_docs`
2. **Pinecone** - For vector search in `retrieve_docs`
3. **Anthropic** - For LLM response generation

The test uses VCR (pytest-recording) to record real API interactions into a "cassette" file, which can then be replayed without API keys or network access.

## Test Features

### What It Tests

- ✅ End-to-end query flow from user question to final response
- ✅ All callback events fired in correct order
- ✅ Correct event types for webclient compatibility
- ✅ Citations retrieved from Pinecone/Voyage
- ✅ Prompt construction with retrieved context
- ✅ Streaming LLM response generation
- ✅ Response structure matches webclient expectations

### What It Validates

**Result Structure:**
- `result["response"]` is a non-empty string
- `result["followups"]` is a list (empty when followups=False)

**Callback Events (in order):**
1. `{"state": "citations", "citations": [...]}`
2. `{"state": "loading", "phase": "prompt"}`
3. `{"state": "prompt", "promptedHistory": [...]}`
4. `{"state": "loading", "phase": "llm"}`
5. Multiple `{"state": "streaming", "content": "..."}` events
6. `{"state": "done"}`

## Recording a Cassette

### Prerequisites

You need all three API keys:
```bash
export ANTHROPIC_API_KEY_DEV="sk-ant-..."
export PINECONE_API_KEY="..."
export VOYAGEAI_API_KEY="..."
```

### Recording Steps

1. **Navigate to api directory:**
   ```bash
   cd api
   ```

2. **Run in record mode:**
   ```bash
   pipenv run pytest tests/stampy_chat/test_chat_integration_vcr.py -v --no-cov --record-mode=rewrite
   ```

3. **Verify cassette created:**
   ```bash
   ls -lh tests/cassettes/test_stampy_chat/test_chat_integration_vcr/test_run_query_end_to_end_with_vcr.yaml
   ```

4. **Verify API keys are redacted:**
   ```bash
   grep -E "(REDACTED|sk-ant-|voyage-|pinecone)" tests/cassettes/test_stampy_chat/test_chat_integration_vcr/test_run_query_end_to_end_with_vcr.yaml
   ```

   You should see "REDACTED" but NO actual API keys!

## Running in Replay Mode

Once the cassette is recorded, the test can run without any API keys:

```bash
ANTHROPIC_API_KEY_DEV="" PINECONE_API_KEY="" VOYAGEAI_API_KEY="" \
  pipenv run pytest tests/stampy_chat/test_chat_integration_vcr.py -v --no-cov
```

This should be much faster than recording (no real API calls) and proves VCR replay works.

## Test Configuration

The test uses minimal settings to keep the cassette small and fast:

```python
Settings(
    completions="anthropic/claude-sonnet-4-20250514",
    enable_hyde=False,        # Skip HyDE to reduce API calls
    topKBlocks=1,             # Only 1 citation to minimize size
    thinking_budget=0,        # No thinking for simplicity
    min_response_tokens=10,   # Short response
    maxNumTokens=8000,
)
```

## Troubleshooting

### Import Error: "Temporary failure in name resolution"

This happens if Pinecone tries to initialize during module import before VCR is set up. The test handles this by temporarily unsetting `PINECONE_API_KEY` during import.

### Test Fails: "Unknown model"

Make sure the model name in the test matches one defined in `settings.py::MODELS`. Currently using `anthropic/claude-sonnet-4-20250514`.

### Cassette Contains Real API Keys

Check `conftest.py::vcr_config` to ensure it filters:
- `x-api-key` (Anthropic)
- `api-key` (Pinecone, Voyage)
- `authorization` (general)

All should be replaced with "REDACTED".

### VCR Not Intercepting Calls

Make sure:
1. Test function has `@pytest.mark.vcr` decorator
2. `conftest.py` has `vcr_config` fixture
3. `pytest-recording` is installed

## Expected Cassette Structure

A complete cassette should contain approximately:

1. **1-2 Voyage AI requests** (embedding the query)
2. **1-2 Pinecone requests** (describe index, vector search)
3. **1 Anthropic streaming request** (with multiple response chunks)

Total cassette size: ~10-50 KB depending on response length.

## CI/CD Integration

Once recorded, this test can run in CI without any API keys, making it:
- ✅ Fast (no real API calls)
- ✅ Free (no API usage costs)
- ✅ Reliable (no network flakiness)
- ✅ Secure (no API keys in CI)

Just commit the cassette file along with the test!
