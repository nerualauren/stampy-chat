"""
Basic VCR test for Anthropic API interactions using pytest-recording.

This test validates that pytest-recording correctly records and replays Anthropic API calls.
"""
import os
import pytest
import anthropic


@pytest.mark.vcr
def test_anthropic_basic_message():
    """Test basic Anthropic API call with VCR recording/replay.

    This test will:
    - On first run (with ANTHROPIC_API_KEY_DEV set): Record the API interaction to a cassette
    - On subsequent runs: Replay from the cassette without making real API calls

    The cassette will be automatically created at:
    tests/cassettes/test_stampy_chat/test_vcr_anthropic_basic/test_anthropic_basic_message.yaml
    """
    # Get API key from environment, or use dummy key for replay
    # pytest-recording will intercept the HTTP call, so the dummy key won't actually be sent
    api_key = os.getenv("ANTHROPIC_API_KEY_DEV") or "sk-ant-dummy-key-for-vcr-replay"

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Make a simple API call with a short prompt to minimize API usage
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
