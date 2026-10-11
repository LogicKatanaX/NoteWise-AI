"""
Unit tests for NoteWise AI — Chat export functionality.
Run with: pytest tests/ -v
"""
import time


def generate_markdown_export(messages: list) -> str:
    """Mirror of the export function in streamlit_app.py for testing."""
    lines = [
        "# NoteWise AI — Session Export",
        f"*Exported on {time.strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]
    for m in messages:
        role_label = "**You**" if m["role"] == "user" else "**NoteWise**"
        meta = ""
        if m["role"] == "assistant":
            if m.get("model_used"):
                meta += f" ·  `{m['model_used']}`"
            if m.get("retrieval_ms") is not None:
                meta += f" · ⚡ {m['retrieval_ms']}ms retrieval"
        lines.append(f"### {role_label}{meta}")
        lines.append(m["content"])
        if m.get("sources"):
            lines.append("\n<details><summary>📚 Source Citations</summary>\n")
            for idx, src in enumerate(m["sources"], 1):
                score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                lines.append(f"**Source #{idx}** · Relevance: {score_str}")
                lines.append(f"> {src['text']}\n")
            lines.append("</details>")
        lines.append("\n---\n")
    return "\n".join(lines)


SAMPLE_MESSAGES = [
    {"role": "user", "content": "What is my GPA?"},
    {
        "role": "assistant",
        "content": "Your GPA is 8.5 based on the uploaded document.",
        "model_used": "gemini-3.8-flash",
        "retrieval_ms": 42,
        "sources": [{"score": 0.912, "text": "GPA: 8.5/10 MAKAUT 2027"}],
    },
]


def test_export_contains_header():
    """Export must start with NoteWise header."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert "# NoteWise AI — Session Export" in output


def test_export_contains_user_message():
    """User messages must appear in export."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert "What is my GPA?" in output


def test_export_contains_assistant_message():
    """Assistant messages must appear in export."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert "Your GPA is 8.5" in output


def test_export_contains_model_badge():
    """Model name should be included in export metadata."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert "gemini-3.8-flash" in output


def test_export_contains_sources():
    """Source citations should appear when available."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert "Source Citations" in output
    assert "GPA: 8.5/10 MAKAUT 2027" in output


def test_export_empty_session():
    """Empty session should still produce valid header."""
    output = generate_markdown_export([])
    assert "# NoteWise AI — Session Export" in output


def test_export_is_string():
    """Export function must return a string."""
    output = generate_markdown_export(SAMPLE_MESSAGES)
    assert isinstance(output, str)

