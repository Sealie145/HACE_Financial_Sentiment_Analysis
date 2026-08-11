"""
detector.py — Hybrid hedge detector (lexicon + spaCy linguistic analysis).

Pipeline:
  1. Lexicon match: find candidate hedge tokens/phrases.
  2. spaCy validation: confirm modal verbs via POS/dependency tags.
  3. Feature computation: hedge_probability, hedge_count, hedge_density.

The detector does NOT modify sentiment probabilities.
It only produces features for the stacking meta-learner.
"""

from __future__ import annotations

import math

import spacy

from .features import HedgeFeatures
from .lexicon import HEDGE_LEXICON, ALL_HEDGE_TERMS


class HedgeDetector:
    """Hybrid financial hedge detector.

    Args:
        spacy_model: spaCy model name to load. Defaults to 'en_core_web_sm'.
        probability_method: How to compute hedge_probability.
            - 'density': sigmoid of hedge_density (default).
            - 'binary': 1.0 if hedge_count > 0 else 0.0.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        probability_method: str = "density",
    ) -> None:
        self._nlp = spacy.load(spacy_model)
        self._probability_method = probability_method

        # Separate multi-word phrases from single tokens for efficiency
        self._phrases = sorted(
            [t for t in ALL_HEDGE_TERMS if " " in t],
            key=len,
            reverse=True,  # match longest first
        )
        self._tokens = set(t for t in ALL_HEDGE_TERMS if " " not in t)

        # Reverse lookup: term → category
        self._term_to_category: dict[str, str] = {
            term: cat
            for cat, terms in HEDGE_LEXICON.items()
            for term in terms
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> HedgeFeatures:
        """Run hedge detection on a preprocessed text string.

        Args:
            text: Cleaned financial text (output of cleaner.clean_text).

        Returns:
            HedgeFeatures instance.
        """
        if not text or not text.strip():
            return HedgeFeatures()

        doc = self._nlp(text)
        token_count = len([t for t in doc if not t.is_space])

        matched_words: list[str] = []
        matched_categories: dict[str, list[str]] = {}

        # ── Step 1: Multi-word phrase matching ────────────────────────────
        lower_text = text.lower()
        for phrase in self._phrases:
            if phrase in lower_text:
                matched_words.append(phrase)
                cat = self._term_to_category.get(phrase, "other")
                matched_categories.setdefault(cat, []).append(phrase)

        # ── Step 2: Single-token matching with spaCy validation ───────────
        for token in doc:
            lower_tok = token.text.lower()
            if lower_tok not in self._tokens:
                continue

            # Modal verb validation: must be tagged as MD and act as auxiliary
            if lower_tok in {"may", "might", "could", "should", "would"}:
                if not self._is_valid_modal(token):
                    continue

            if lower_tok not in matched_words:
                matched_words.append(lower_tok)
                cat = self._term_to_category.get(lower_tok, "other")
                matched_categories.setdefault(cat, []).append(lower_tok)

        hedge_count = len(matched_words)
        hedge_density = hedge_count / token_count if token_count > 0 else 0.0
        hedge_probability = self._compute_probability(hedge_count, hedge_density)

        return HedgeFeatures(
            hedge_probability=round(hedge_probability, 4),
            hedge_count=hedge_count,
            hedge_density=round(hedge_density, 4),
            detected_hedge_words=matched_words,
            hedge_categories=matched_categories,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_modal(token: spacy.tokens.Token) -> bool:
        """Return True if the token is a modal auxiliary in context."""
        return token.tag_ == "MD" and token.dep_ in {"aux", "auxpass", "ROOT"}

    def _compute_probability(self, count: int, density: float) -> float:
        """Compute hedge_probability from count and density.

        Methods:
          'density' : sigmoid(10 * density) — smooth, bounded in (0, 1).
          'binary'  : 1.0 if count > 0 else 0.0.
        """
        if self._probability_method == "binary":
            return 1.0 if count > 0 else 0.0
        # Default: sigmoid-scaled density
        return 1.0 / (1.0 + math.exp(-10.0 * density))
