# Tool Use Testing

This directory contains tests for the Anthropic API integration with tool use, thinking, and streaming.

## Test Files

- `test_tool_execution.py` - Unit tests for tool execution logic
- `test_tool_state_machine.py` - Mock-based tests for the streaming state machine
- `test_llms_with_tools.py` - Integration tests with VCR cassettes (requires API key)

## Running Tests

Run all tests:
```bash
pipenv run pytest
```

Run specific test files:
```bash
pipenv run pytest tests/stampy_chat/test_tool_execution.py -v
pipenv run pytest tests/stampy_chat/test_tool_state_machine.py -v
```

## VCR Integration Tests

The `test_llms_with_tools.py` file contains integration tests that use VCR.py to record and replay API interactions.

### Recording New Cassettes

To record new VCR cassettes (requires ANTHROPIC_API_KEY):

1. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. Delete existing cassettes (if any):
   ```bash
   rm tests/stampy_chat/fixtures/vcr_cassettes/*.yaml
   ```

3. Run the tests:
   ```bash
   pipenv run pytest tests/stampy_chat/test_llms_with_tools.py -v
   ```

VCR will record the API interactions and save them as cassettes. Subsequent test runs will use the recorded cassettes instead of making real API calls.

### VCR Configuration

The VCR configuration in `test_llms_with_tools.py`:
- Records cassettes in `fixtures/vcr_cassettes/`
- Uses `record_mode="once"` - only records if cassette doesn't exist
- Filters sensitive headers (authorization, x-api-key)
- Matches requests on method, host, path, query, and body

## Tool Implementation

The current implementation includes:

### Available Tools

1. **calculator** - Evaluates mathematical expressions
   - Input: `expression` (string)
   - Output: Result as string
   - Example: `{"expression": "123 * 456"}` → `"56088"`

### Adding New Tools

To add a new tool:

1. Add the tool definition to `TOOL_DEFINITIONS` in `llms.py`:
   ```python
   {
       "name": "tool_name",
       "description": "Tool description",
       "input_schema": {
           "type": "object",
           "properties": {
               "param": {"type": "string", "description": "..."}
           },
           "required": ["param"]
       }
   }
   ```

2. Implement the executor function:
   ```python
   def execute_my_tool(param: str) -> str:
       # Tool logic here
       return result
   ```

3. Register in `TOOL_EXECUTORS`:
   ```python
   TOOL_EXECUTORS = {
       "tool_name": lambda input: execute_my_tool(input["param"])
   }
   ```

4. Add tests in `test_tool_execution.py`

## State Machine Flow

The tool use state machine in `anthropic_stream_with_tools`:

1. Make streaming API request with tools enabled
2. Accumulate content blocks (thinking, text, tool_use)
3. Stream chunks to caller in real-time
4. If tool_use blocks present:
   - Execute tools
   - Yield tool_use and tool_result chunks
   - Add to conversation history
   - Loop back to step 1
5. If no tool_use blocks, exit loop

This allows for multiple sequential tool uses in a single conversation turn.
