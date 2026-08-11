"""
components.py — Reusable Gradio UI helper functions.

Formats raw HACE prediction dicts into display-ready strings and HTML.
"""

from __future__ import annotations


SENTIMENT_EMOJI = {"positive": "🟢", "neutral": "🟡", "negative": "🔴", "unavailable": "⚪"}

EXPERT_LABELS = {
    "fiqa":         "FiQA Expert",
    "phrasebank":   "PhraseBank Expert",
    "twitter":      "Twitter Expert",
    "finance_news": "Finance News Expert",
    "general":      "General FinBERT",
}


def format_sentiment_badge(sentiment: str, confidence: float) -> str:
    emoji = SENTIMENT_EMOJI.get(sentiment.lower(), "⚪")
    return f"{emoji} {sentiment.upper()}  (confidence: {confidence:.2%})"


def format_hedge_summary(hedge_prob: float, hedge_count: int, hedge_words: list[str]) -> str:
    if hedge_count == 0:
        return "✅ No hedging language detected."
    words_str = ", ".join(f'"{w}"' for w in hedge_words)
    return (
        f"⚠️ Hedge detected (probability: {hedge_prob:.2%})\n"
        f"   Cues: {words_str}"
    )


def format_expert_table(experts: dict) -> str:
    """Format expert predictions as a markdown table."""
    lines = ["| Expert | Sentiment | Confidence | Neg | Neu | Pos |",
             "|---|---|---|---|---|---|"]
    for key, pred in experts.items():
        label = EXPERT_LABELS.get(key, key)
        p = pred.get("probabilities", {})
        lines.append(
            f"| {label} | {pred['sentiment'].capitalize()} "
            f"| {pred['confidence']:.2%} "
            f"| {p.get('negative', 0):.2%} "
            f"| {p.get('neutral', 0):.2%} "
            f"| {p.get('positive', 0):.2%} |"
        )
    return "\n".join(lines)


def format_comparison(base: dict, hace: dict) -> str:
    """Side-by-side Base Ensemble vs HACE comparison."""
    base_sent = base.get("sentiment", "unavailable")
    hace_sent = hace.get("sentiment", "unavailable")
    base_conf = base.get("confidence", 0.0)
    hace_conf = hace.get("confidence", 0.0)
    return (
        f"**Base Ensemble:** {SENTIMENT_EMOJI.get(base_sent, '⚪')} "
        f"{base_sent.upper()} ({base_conf:.2%})\n\n"
        f"**HACE:**          {SENTIMENT_EMOJI.get(hace_sent, '⚪')} "
        f"{hace_sent.upper()} ({hace_conf:.2%})"
    )


def format_explanation(items: list[str]) -> str:
    return "\n".join(f"• {item}" for item in items)
