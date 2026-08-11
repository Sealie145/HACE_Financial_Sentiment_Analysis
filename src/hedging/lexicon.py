"""
lexicon.py — Financial hedge lexicon.

Organized by category to support optional hedge_category output.
Extend this lexicon based on corpus analysis during Phase 4.
All entries should be lowercase.
"""

from __future__ import annotations

# ── Lexicon by category ───────────────────────────────────────────────────────

MODAL_VERBS: list[str] = [
    "may", "might", "could", "should", "would",
]

EPISTEMIC_ADVERBS: list[str] = [
    "possibly", "potentially", "likely", "probably", "perhaps",
    "approximately", "roughly", "about", "around", "nearly",
    "almost", "seemingly", "ostensibly",
]

EPISTEMIC_VERBS: list[str] = [
    "appears", "appear", "suggests", "suggest", "seems", "seem",
    "expects", "expect", "anticipates", "anticipate",
    "estimates", "estimate", "believes", "believe",
    "indicates", "indicate",
]

CONDITIONAL_PHRASES: list[str] = [
    "subject to", "depending on", "contingent on",
    "assuming that", "in the event that", "provided that",
    "if approved", "pending",
]

APPROXIMATION_PHRASES: list[str] = [
    "in the range of", "up to", "close to", "as much as",
    "at least approximately", "no more than approximately",
]

# ── Flat set (used for fast membership checks) ────────────────────────────────

HEDGE_LEXICON: dict[str, list[str]] = {
    "modal": MODAL_VERBS,
    "epistemic_adverb": EPISTEMIC_ADVERBS,
    "epistemic_verb": EPISTEMIC_VERBS,
    "conditional": CONDITIONAL_PHRASES,
    "approximation": APPROXIMATION_PHRASES,
}

# Flat list of all terms — used for simple substring / token matching
ALL_HEDGE_TERMS: list[str] = [
    term
    for terms in HEDGE_LEXICON.values()
    for term in terms
]
