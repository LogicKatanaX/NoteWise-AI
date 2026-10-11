"""
Unit tests for NoteWise AI — RAGAS eval score parsing.
Run with: pytest tests/ -v
"""
import json


def parse_eval_scores(raw_text: str) -> dict:
    """Mirror of eval JSON parsing logic in streamlit_app.py."""
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            scores = json.loads(raw_text[start:end])
            return {
                "faithfulness": max(0.0, min(1.0, float(scores.get("faithfulness", 0.5)))),
                "relevance": max(0.0, min(1.0, float(scores.get("relevance", 0.5)))),
                "completeness": max(0.0, min(1.0, float(scores.get("completeness", 0.5)))),
            }
    except Exception:
        pass
    return {}


def score_badge_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.45:
        return "medium"
    return "low"


def test_valid_json_parsed_correctly():
    """Valid JSON scores should parse to correct floats."""
    raw = '{"faithfulness": 0.9, "relevance": 0.85, "completeness": 0.7}'
    result = parse_eval_scores(raw)
    assert result["faithfulness"] == 0.9
    assert result["relevance"] == 0.85
    assert result["completeness"] == 0.7


def test_json_embedded_in_text():
    """JSON embedded in prose should still be extracted."""
    raw = 'Here are the scores: {"faithfulness": 0.8, "relevance": 0.6, "completeness": 0.5} End.'
    result = parse_eval_scores(raw)
    assert result["faithfulness"] == 0.8


def test_score_clamped_to_0_1():
    """Scores outside [0, 1] must be clamped."""
    raw = '{"faithfulness": 1.5, "relevance": -0.2, "completeness": 0.6}'
    result = parse_eval_scores(raw)
    assert result["faithfulness"] == 1.0
    assert result["relevance"] == 0.0
    assert result["completeness"] == 0.6


def test_missing_keys_default_to_0_5():
    """Missing score keys should default to 0.5."""
    raw = '{"faithfulness": 0.8}'
    result = parse_eval_scores(raw)
    assert result["relevance"] == 0.5
    assert result["completeness"] == 0.5


def test_invalid_json_returns_empty():
    """Malformed JSON should return empty dict, not raise."""
    raw = "Sorry, I cannot evaluate this."
    result = parse_eval_scores(raw)
    assert result == {}


def test_badge_class_high():
    assert score_badge_class(0.9) == "high"
    assert score_badge_class(0.75) == "high"


def test_badge_class_medium():
    assert score_badge_class(0.6) == "medium"
    assert score_badge_class(0.45) == "medium"


def test_badge_class_low():
    assert score_badge_class(0.3) == "low"
    assert score_badge_class(0.0) == "low"

