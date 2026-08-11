"""Lightweight, token-aware lexicon hedge detector.

Version 1 intentionally contains no learned model. ``hedge_probability`` is
a deterministic lexicon-derived confidence score, not a calibrated probability.
"""

from __future__ import annotations

import re

from .features import HedgeFeatures
from .lexicon import HEDGE_WEIGHTS, TERM_CATEGORIES


class HedgeDetector:
    """Detect financial hedging cues without external NLP-model dependencies."""

    _WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)

    def __init__(self) -> None:
        self._phrase_patterns = [
            (term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE))
            for term in sorted((term for term in HEDGE_WEIGHTS if " " in term), key=len, reverse=True)
        ]
        self._token_terms = {term for term in HEDGE_WEIGHTS if " " not in term}

    def detect(self, text: str) -> HedgeFeatures:
        """Return deterministic hedge features for *text*.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return HedgeFeatures()

        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        occupied: list[tuple[int, int]] = []
        matches: list[tuple[int, str]] = []

        for term, pattern in self._phrase_patterns:
            for match in pattern.finditer(normalized):
                occupied.append(match.span())
                matches.append((match.start(), term))

        for match in self._WORD_RE.finditer(normalized):
            term = match.group(0)
            if term not in self._token_terms:
                continue
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            matches.append((match.start(), term))

        matches.sort(key=lambda item: item[0])
        detected_terms = [term for _, term in matches]
        categories: dict[str, list[str]] = {}
        for term in detected_terms:
            categories.setdefault(TERM_CATEGORIES[term], []).append(term)

        token_count = len(self._WORD_RE.findall(normalized))
        cue_score = sum(HEDGE_WEIGHTS[term] for term in detected_terms)
        # Saturating deterministic transformation; bounded, interpretable, uncalibrated.
        hedge_probability = min(1.0, round(cue_score, 4))
        return HedgeFeatures(
            hedge_flag=bool(detected_terms),
            hedge_probability=hedge_probability,
            detected_terms=detected_terms,
            hedge_count=len(detected_terms),
            hedge_density=round(len(detected_terms) / token_count, 4) if token_count else 0.0,
            hedge_categories=categories,
        )
