"""
Basic VCR test for Anthropic API interactions.

This test validates that vcrpy correctly records and replays Anthropic API calls.
"""
import os
import pytest
import vcr
import anthropic


# Configure VCR to filter sensitive data
vcr_config = vcr.VCR(
    cassette_library_dir="tests/cassettes",
    filter_headers=["authorization", "x-api-key"],
    filter_post_data_parameters=["api_key"],
    record_mode="once",  # Record once, then replay
    match_on=["method", "scheme", "host", "port", "path", "query"],
)


@pytest.mark.vcr
@vcr_config.use_cassette("test_anthropic_basic.yaml")
def test_anthropic_basic_message():
    """Test basic Anthropic API call with VCR recording/replay."""
    # Get API key from environment, or use dummy key for replay
    # VCR will intercept the HTTP call, so the dummy key won't be used
    api_key = os.getenv("ANTHROPIC_API_KEY_DEV") or "sk-ant-dummy-key-for-vcr-replay"

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Make a simple API call with a short prompt
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": "In one token, do you see this message? Respond only with 'Yes'"
            }
        ]
    )

    # Verify response structure
    assert message is not None
    assert hasattr(message, "content")
    assert len(message.content) > 0

    # Verify response contains expected text
    response_text = message.content[0].text
    assert response_text is not None
    assert len(response_text) > 0

    # The response should contain "Yes" or similar affirmation
    # (being flexible since we're constraining tokens)
    assert any(word.lower() in response_text.lower() for word in ["yes", "y"])
