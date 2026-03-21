import json
from unittest.mock import MagicMock, patch

from cards import InsufficientContentError, generate_cards


def _mock_response(text: str):
    """Build a minimal Anthropic response mock returning the given text."""
    content = MagicMock()
    content.text = text
    message = MagicMock()
    message.content = [content]
    client = MagicMock()
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# Insufficient content
# ---------------------------------------------------------------------------

def test_hello_world_raises_insufficient_content():
    """'hello world' input — Claude signals insufficient content, should raise."""
    client = _mock_response('{"error": "insufficient_content"}')
    with patch("cards.anthropic.Anthropic", return_value=client):
        try:
            generate_cards("hello world", "fake_key", "claude-sonnet-4-20250514")
            assert False, "Expected InsufficientContentError"
        except InsufficientContentError:
            pass


def test_placeholder_text_raises_insufficient_content():
    """Any input where Claude returns the insufficient_content signal should raise."""
    client = _mock_response('{"error": "insufficient_content"}')
    with patch("cards.anthropic.Anthropic", return_value=client):
        try:
            generate_cards("testing 123", "fake_key", "claude-sonnet-4-20250514")
            assert False, "Expected InsufficientContentError"
        except InsufficientContentError:
            pass


# ---------------------------------------------------------------------------
# Normal card generation
# ---------------------------------------------------------------------------

def test_valid_notes_returns_cards():
    """Well-formed notes should return a list of card dicts."""
    cards_json = json.dumps([
        {"front": "What is photosynthesis?", "back": "The process plants use to convert light into energy."},
        {"front": "What molecule does photosynthesis produce?", "back": "Glucose"},
    ])
    client = _mock_response(cards_json)
    with patch("cards.anthropic.Anthropic", return_value=client):
        cards = generate_cards("Photosynthesis notes...", "fake_key", "claude-sonnet-4-20250514")

    assert len(cards) == 2
    assert cards[0]["front"] == "What is photosynthesis?"
    assert cards[0]["selected"] is True


def test_cards_include_model_attribution():
    """Each card's back field should include the model name."""
    model = "claude-sonnet-4-20250514"
    cards_json = json.dumps([{"front": "Q", "back": "A"}])
    client = _mock_response(cards_json)
    with patch("cards.anthropic.Anthropic", return_value=client):
        cards = generate_cards("some notes", "fake_key", model)

    assert model in cards[0]["back"]


def test_json_in_markdown_code_fence_is_parsed():
    """Claude sometimes wraps JSON in markdown fences — should still parse."""
    cards_json = json.dumps([{"front": "Q", "back": "A"}])
    fenced = f"```json\n{cards_json}\n```"
    client = _mock_response(fenced)
    with patch("cards.anthropic.Anthropic", return_value=client):
        cards = generate_cards("some notes", "fake_key", "claude-sonnet-4-20250514")

    assert len(cards) == 1
