"""
app.py — Gradio frontend for HACE.

The frontend communicates with the FastAPI backend through HTTP.

Architecture:

Gradio Frontend (:7860)
        ↓
FastAPI Backend (:8000)
        ↓
HACEService
        ↓
5 FinBERT Experts + Hedge Detector + HACE Meta-Learner
"""

from __future__ import annotations

import os
import requests
import gradio as gr

from components import (
    format_comparison,
    format_expert_table,
    format_explanation,
    format_hedge_summary,
    format_sentiment_badge,
)


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = os.getenv(
    "HACE_BACKEND_URL",
    "http://127.0.0.1:8000"
)

PREDICT_URL = f"{BACKEND_URL}/predict"


# ============================================================
# EXAMPLES
# ============================================================

EXAMPLES = [
    ["Revenue increased by 18%."],
    ["Revenue may increase by 18%."],
    ["The company may report lower earnings next quarter."],
    ["The board approved a 10% dividend increase."],
    ["Profits could fall depending on market conditions."],
]


# ============================================================
# BACKEND HEALTH CHECK
# ============================================================

def check_backend() -> bool:
    """
    Check whether the FastAPI backend is running
    and the HACE models are loaded.
    """

    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=10,
        )

        if response.status_code != 200:
            return False

        data = response.json()

        return bool(data.get("models_loaded", False))

    except requests.RequestException:
        return False


# ============================================================
# INFERENCE
# ============================================================

def analyze(text: str):
    """
    Send text to the FastAPI backend and format
    the returned HACE prediction for Gradio.
    """

    if not text or not text.strip():

        empty = "Please enter some financial text."

        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    try:

        response = requests.post(
            PREDICT_URL,
            json={"text": text},
            timeout=120,
        )

        # ----------------------------------------------------
        # Backend error
        # ----------------------------------------------------

        if response.status_code != 200:

            try:
                error_detail = response.json().get(
                    "detail",
                    "Unknown backend error."
                )
            except Exception:
                error_detail = response.text

            msg = (
                f"❌ Backend error "
                f"(HTTP {response.status_code}): {error_detail}"
            )

            return (
                msg,
                msg,
                msg,
                msg,
                msg,
                msg,
            )

        # ----------------------------------------------------
        # Parse response
        # ----------------------------------------------------

        result = response.json()

        # ----------------------------------------------------
        # Format sentiment
        # ----------------------------------------------------

        sentiment_display = format_sentiment_badge(
            result["sentiment"],
            result["confidence"],
        )

        # ----------------------------------------------------
        # Format hedge information
        # ----------------------------------------------------

        hedge_display = format_hedge_summary(
            result["hedge_probability"],
            result["hedge_count"],
            result["hedge_words"],
        )

        # ----------------------------------------------------
        # Format expert predictions
        # ----------------------------------------------------

        expert_display = format_expert_table(
            result["experts"]
        )

        # ----------------------------------------------------
        # Expert agreement
        # ----------------------------------------------------

        agreement = result["expert_agreement"]

        agreement_display = (
            f"Expert agreement: {agreement:.0%} "
            f"({int(round(agreement * 5))}/5 experts agree)"
        )

        # ----------------------------------------------------
        # Base Ensemble vs HACE
        # ----------------------------------------------------

        comparison_display = format_comparison(
            result["base_ensemble"],
            result,
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation_display = format_explanation(
            result["explanation"]
        )

        return (
            sentiment_display,
            hedge_display,
            expert_display,
            agreement_display,
            comparison_display,
            explanation_display,
        )

    except requests.exceptions.Timeout:

        msg = (
            "❌ Backend request timed out. "
            "The HACE models may still be loading or inference is taking too long."
        )

        return (
            msg,
            msg,
            msg,
            msg,
            msg,
            msg,
        )

    except requests.exceptions.ConnectionError:

        msg = (
            "❌ Cannot connect to the HACE backend.\n\n"
            "Make sure FastAPI is running with:\n\n"
            "`uvicorn backend.main:app --reload`"
        )

        return (
            msg,
            msg,
            msg,
            msg,
            msg,
            msg,
        )

    except Exception as e:

        msg = f"❌ Frontend error: {type(e).__name__}: {e}"

        return (
            msg,
            msg,
            msg,
            msg,
            msg,
            msg,
        )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="HACE — Financial Sentiment Analysis"
) as demo:

    gr.Markdown(
        "# HACE — Hedging-Aware Cross-Domain Ensemble\n"
        "**Financial Sentiment Analysis with Uncertainty Awareness**\n\n"
        "Combines five domain-specialized FinBERT experts with "
        "hedge detection to classify financial text as "
        "Negative / Neutral / Positive."
    )

    # --------------------------------------------------------
    # Backend status
    # --------------------------------------------------------

    backend_status = (
        "🟢 Backend connected"
        if check_backend()
        else "🔴 Backend unavailable"
    )

    gr.Markdown(
        f"**Backend:** {backend_status}"
    )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    with gr.Tabs():

        # ====================================================
        # PREDICTION TAB
        # ====================================================

        with gr.Tab("Prediction"):

            with gr.Row():

                with gr.Column(scale=2):

                    text_input = gr.Textbox(
                        label="Financial Text",
                        placeholder=(
                            "Enter financial text here..."
                        ),
                        lines=4,
                    )

                    analyze_btn = gr.Button(
                        "Analyze",
                        variant="primary",
                    )

            # ------------------------------------------------
            # Sentiment + Hedge
            # ------------------------------------------------

            with gr.Row():

                sentiment_out = gr.Markdown(
                    label="Sentiment"
                )

                hedge_out = gr.Markdown(
                    label="Hedge Detection"
                )

            # ------------------------------------------------
            # Agreement
            # ------------------------------------------------

            with gr.Row():

                agreement_out = gr.Markdown(
                    label="Expert Agreement"
                )

            # ------------------------------------------------
            # Expert predictions
            # ------------------------------------------------

            expert_out = gr.Markdown(
                label="Expert Predictions"
            )

            # ------------------------------------------------
            # Ensemble comparison
            # ------------------------------------------------

            comparison_out = gr.Markdown(
                label="Base Ensemble vs HACE"
            )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            explanation_out = gr.Markdown(
                label="Explanation"
            )

            # ------------------------------------------------
            # Analyze button
            # ------------------------------------------------

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

        # ====================================================
        # EXAMPLES TAB
        # ====================================================

        with gr.Tab("Examples"):

            gr.Markdown(
                "Click any example to populate the input "
                "and run analysis.\n\n"
                "Examples demonstrate sentiment classification "
                "and the effect of hedging language."
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "GRADIO_PORT",
            7860,
        )
    )

    share = (
        os.getenv(
            "GRADIO_SHARE",
            "false",
        ).lower()
        == "true"
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
    )