"""
app.py — Gradio demonstration interface for HACE.

Tabs:
  1. Prediction — full HACE inference with expert breakdown and hedge display
  2. Examples   — pre-loaded research demonstration examples

Run:
  python frontend/app.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr

from backend.service import HACEService
from frontend.components import (
    format_comparison,
    format_expert_table,
    format_explanation,
    format_hedge_summary,
    format_sentiment_badge,
)

# ── Load service ──────────────────────────────────────────────────────────────

service = HACEService()
service.load()

# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["Revenue increased by 18%."],
    ["Revenue may increase by 18%."],
    ["The company may report lower earnings next quarter."],
    ["The board approved a 10% dividend increase."],
    ["Profits could fall depending on market conditions."],
]

# ── Inference function ────────────────────────────────────────────────────────

def analyze(text: str):
    """Run HACE inference and return formatted outputs for Gradio components."""
    if not text or not text.strip():
        empty = "Please enter some financial text."
        return empty, empty, empty, empty, empty, empty

    try:
        result = service.predict(text)
    except Exception as e:
        msg = f"Error: {e}"
        return msg, msg, msg, msg, msg, msg

    sentiment_display = format_sentiment_badge(result["sentiment"], result["confidence"])
    hedge_display = format_hedge_summary(
        result["hedge_probability"],
        result["hedge_count"],
        result["hedge_words"],
    )
    expert_display = format_expert_table(result["experts"])
    agreement_display = (
        f"Expert agreement: {result['expert_agreement']:.0%} "
        f"({int(result['expert_agreement'] * 5)}/5 experts agree)"
    )
    comparison_display = format_comparison(result["base_ensemble"], result)
    explanation_display = format_explanation(result["explanation"])

    return (
        sentiment_display,
        hedge_display,
        expert_display,
        agreement_display,
        comparison_display,
        explanation_display,
    )

# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="HACE — Financial Sentiment Analysis") as demo:
    gr.Markdown(
        "# HACE — Hedging-Aware Cross-Domain Ensemble\n"
        "**Financial Sentiment Analysis with Uncertainty Awareness**\n\n"
        "Combines five domain-specialized FinBERT experts with hedge detection "
        "to classify financial text as Negative / Neutral / Positive."
    )

    with gr.Tabs():
        # ── Tab 1: Prediction ──────────────────────────────────────────────
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Financial Text",
                        placeholder="Enter financial text here...",
                        lines=4,
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                sentiment_out = gr.Markdown(label="Sentiment")
                hedge_out = gr.Markdown(label="Hedge Detection")

            with gr.Row():
                agreement_out = gr.Markdown(label="Expert Agreement")

            expert_out = gr.Markdown(label="Expert Predictions")
            comparison_out = gr.Markdown(label="Base Ensemble vs HACE")
            explanation_out = gr.Markdown(label="Explanation")

            analyze_btn.click(
                fn=analyze,
                inputs=text_input,
                outputs=[
                    sentiment_out,
                    hedge_out,
                    expert_out,
                    agreement_out,
                    comparison_out,
                    explanation_out,
                ],
            )

        # ── Tab 2: Examples ────────────────────────────────────────────────
        with gr.Tab("Examples"):
            gr.Markdown(
                "Click any example to populate the input and run analysis.\n\n"
                "Examples 1–2 demonstrate the effect of hedging on confidence.\n"
                "Example 3 demonstrates the full HACE pipeline."
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                outputs=[
                    sentiment_out,
                    hedge_out,
                    expert_out,
                    agreement_out,
                    comparison_out,
                    explanation_out,
                ],
                fn=analyze,
                cache_examples=False,
            )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", 7860))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(server_port=port, share=share)
