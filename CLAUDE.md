in api/, use pipenv to run python commands. otherwise, `python` doesn't exist, `python3` isn't in venv.

tests are `cd api && pipenv run pytest`.

dev server are in readme or mprocs.yaml, but it's not convenient for claude to run them since they don't halt unless interrupted.

## VCR Testing (vcrpy)

VCR (vcrpy) is used to record and replay HTTP interactions with external APIs (like Anthropic). This makes tests faster, more reliable, and runnable without API keys.

**How VCR works:**
- First run: Records real API interactions to YAML "cassette" files
- Subsequent runs: Replays recorded interactions without hitting the API

**Recording new cassettes:**
- Ensure ANTHROPIC_API_KEY_DEV environment variable is set
- Run the test: `pipenv run pytest tests/stampy_chat/test_vcr_anthropic_basic.py -v`
- Cassette files are stored in `tests/cassettes/`

**Replaying cassettes:**
- No API key needed! Tests use dummy key when replaying
- Run normally: `pipenv run pytest tests/stampy_chat/test_vcr_anthropic_basic.py -v`
- Much faster than recording (no network calls)

**Re-recording cassettes:**
- Delete the cassette file in `tests/cassettes/`
- Run the test with API key set (will record fresh)

**Configuration gotchas:**
- VCR filters sensitive headers (authorization, x-api-key) from cassettes
- Anthropic client requires a valid-looking API key, so tests use a fallback dummy key for replay: `os.getenv("ANTHROPIC_API_KEY_DEV") or "sk-ant-dummy-key-for-vcr-replay"`
- VCR mode is "once" - records once, then always replays (prevents accidental overwrites)
- Match criteria: method, scheme, host, port, path, query

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
