# Tests for Stampy Chat API

## Overview

This directory contains comprehensive tests for the Stampy Chat API, with a focus on testing the LLM integration layer.

## Test Files

- `stampy_chat/test_llms.py` - Tests for the LLM provider interface (`llms.py`)
- `stampy_chat/test_logging.py` - Tests for logging functionality
- `stampy_chat/test_callbacks.py` - Tests for callback functionality

## LLM Tests (`test_llms.py`)

The LLM tests use pytest-recording to create VCR cassettes for API requests, ensuring tests don't make actual API calls but can record and replay interactions when needed.

### Test Structure

1. **TestUtilityFunctions** - Tests utility functions like `split_system`, `execute_tool`
2. **TestAnthropicProvider** - Tests Anthropic API integration with mocked responses
3. **TestOpenAIProvider** - Tests OpenAI API integration
4. **TestGoogleProvider** - Tests Google Gemini API integration
5. **TestOpenRouterProvider** - Tests OpenRouter API integration
6. **TestQueryLLM** - Tests the main `query_llm` function

### Recording Markers

All potentially API-using tests are marked with `@pytest.mark.recording` to ensure they use VCR cassettes rather than making live API calls.

### Mocking Strategy

Tests use extensive mocking of API clients and responses to:
- Avoid actual API calls during testing
- Test error handling and fallback behavior
- Verify correct parameter passing to API clients
- Test tool use functionality

## Running Tests

### Without Coverage (Recommended for Development)
```bash
cd api && pipenv run pytest tests/stampy_chat/test_llms.py --no-cov -v
```

### With Coverage
```bash
cd api && pipenv run pytest tests/stampy_chat/test_llms.py -v
```

### Run Specific Test Groups
```bash
# Utility functions only
pipenv run pytest tests/stampy_chat/test_llms.py::TestUtilityFunctions --no-cov -v

# Single provider tests
pipenv run pytest tests/stampy_chat/test_llms.py::TestAnthropicProvider --no-cov -v
```

## Configuration

- `pytest.ini` - Main pytest configuration with markers for recording tests
- `conftest.py` - Test fixtures and configuration including API key mocking

## Safety Features

- All API keys are mocked in tests via the `mock_api_keys` fixture
- Tests use recording markers to prevent accidental API calls
- Resource usage is minimized by running without coverage during development