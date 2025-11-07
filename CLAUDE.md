in api/, use pipenv to run python commands. otherwise, `python` doesn't exist, `python3` isn't in venv.

tests are `cd api && pipenv run pytest`.

dev server are in readme or mprocs.yaml, but it's not convenient for claude to run them since they don't halt unless interrupted.

Code style:
- prefer putting sufficiently-short single-line if statements on the same line:

    if x: a()
    else: b()
    if condition(long_parameter=foo.bar().baz() + foo.bar(depth=-1).baz(herp="derp")):
        a()
    else:
        b()

- judiciously use short names to ease human typing, unless readability suffers. one or two acronym names per file is ok. Prefer single word names where possible.
- Don't try/except unless the error needs handling. For errors that will stop the program, just let it crash. Rare, world-stopping-anyway error handling isn't usually worth the readability cost.

## VCR Testing with pytest-recording

This project uses `pytest-recording` (a pytest plugin wrapping vcrpy) to record and replay API interactions for testing. This allows tests to run without making actual API calls after the initial recording.

### How to Use VCR Tests

**Mark a test with `@pytest.mark.vcr`:**
```python
import pytest

@pytest.mark.vcr
def test_my_api_call():
    # Make API calls here
    # First run: records to cassette
    # Subsequent runs: replays from cassette
    pass
```

**Cassette Location:**
- pytest-recording automatically creates cassettes at: `tests/cassettes/<test_module>/<test_function>.yaml`
- For example: `test_foo.py::test_bar()` creates `tests/cassettes/test_foo/test_bar.yaml`

**Recording New/Updated Cassettes:**
```bash
cd api
# Ensure API keys are set in environment (ANTHROPIC_API_KEY_DEV, etc.)
pipenv run pytest tests/stampy_chat/test_my_test.py --record-mode=rewrite
```

**Replaying from Cassettes (no API keys needed):**
```bash
cd api
# Unset API keys if you want to verify replay works
ANTHROPIC_API_KEY_DEV="" pipenv run pytest tests/stampy_chat/test_my_test.py
# Or just run normally - pytest-recording will use cassettes by default
pipenv run pytest tests/stampy_chat/test_my_test.py
```

**Block Network Option:**
- `pytest.ini` includes `--block-network` which prevents accidental API calls
- Tests marked with `@pytest.mark.vcr` are automatically allowed through
- This ensures tests fail if they try to make unmocked network calls

### Common pytest-recording Gotchas

1. **API Key Validation**: Some SDKs (like Anthropic) validate API keys before making HTTP requests. Use a fallback dummy key:
   ```python
   api_key = os.getenv("ANTHROPIC_API_KEY_DEV") or "sk-ant-dummy-key-for-vcr-replay"
   ```

2. **Record Modes**:
   - `once` (default): Record if cassette doesn't exist, otherwise replay
   - `rewrite`: Always record, overwrite existing cassettes
   - `none`: Never record, only replay (useful in CI)
   - `new_episodes`: Add new interactions to existing cassette

3. **Sensitive Data**: pytest-recording automatically filters common sensitive headers. For additional filtering, create a `tests/conftest.py`:
   ```python
   @pytest.fixture(scope="module")
   def vcr_config():
       return {
           "filter_headers": ["authorization", "x-api-key"],
           "filter_post_data_parameters": ["api_key"],
       }
   ```

4. **Debugging**: To see what's being recorded, check the cassette YAML files in `tests/cassettes/`

### Example Test Structure

```python
import os
import pytest
import anthropic

@pytest.mark.vcr
def test_anthropic_basic():
    api_key = os.getenv("ANTHROPIC_API_KEY_DEV") or "sk-ant-dummy-key-for-vcr-replay"
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert message.content[0].text
```

First run (with API key set) records the interaction. Subsequent runs replay from the cassette without needing the API key.
