"""Interpretable financial-language hedge lexicon for HACE.

Each cue is stored exactly once, with its category and contribution to the
lexicon-derived hedge confidence score.  The score is not calibrated and is
not a machine-learning probability.
"""

from __future__ import annotations

HEDGE_LEXICON: dict[str, dict[str, float]] = {
    "modal": {
        "may": 0.30, "might": 0.30, "could": 0.28, "can": 0.18,
        "would": 0.18,
    },
    "epistemic": {
        "suggest": 0.25, "suggests": 0.25, "suggested": 0.25,
        "appear": 0.25, "appears": 0.25, "appeared": 0.25,
        "seem": 0.25, "seems": 0.25, "seemed": 0.25,
        "indicate": 0.22, "indicates": 0.22, "indicated": 0.22,
        "believe": 0.20, "believes": 0.20,
        "expect": 0.20, "expects": 0.20, "expected": 0.20,
        "estimate": 0.20, "estimates": 0.20, "estimated": 0.20,
    },
    "uncertainty": {
        "possibly": 0.30, "potentially": 0.30, "perhaps": 0.25,
        "likely": 0.28, "unlikely": 0.28, "probable": 0.25,
    },
    "approximator": {
        "approximately": 0.16, "roughly": 0.16, "around": 0.14,
        "about": 0.12, "nearly": 0.14, "almost": 0.14,
        "relatively": 0.12,
    },
    "conditional": {
        "depending on": 0.25, "subject to": 0.25, "contingent on": 0.25,
        "based on": 0.15, "assuming that": 0.22, "assuming": 0.18,
        "depending": 0.18, "if": 0.10, "unless": 0.15,
    },
}

HEDGE_WEIGHTS: dict[str, float] = {
    term: weight for cues in HEDGE_LEXICON.values() for term, weight in cues.items()
}

TERM_CATEGORIES: dict[str, str] = {
    term: category for category, cues in HEDGE_LEXICON.items() for term in cues
}

ALL_HEDGE_TERMS: tuple[str, ...] = tuple(HEDGE_WEIGHTS)
