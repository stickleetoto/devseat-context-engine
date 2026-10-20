"""Transparent lexical relevance, not embedding-based semantic similarity."""

import re

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_HANGUL = re.compile(r"[가-힣]{2,}")


def lexical_features(text: str) -> frozenset[str]:
    """Case-folded words plus Hangul syllable bigrams for agglutinative matching.

    Works without optional models; scores indicate token overlap, not meaning.
    """
    features = set()
    for token in _WORD.findall(text.casefold()):
        features.add(token)
        for match in _HANGUL.finditer(token):
            word = match.group()
            features.update(word[i : i + 2] for i in range(len(word) - 1))
    return frozenset(features)


def lexical_relevance(query: str, text: str) -> float:
    """Fraction of query's lexical features appearing in text (0.0..1.0)."""
    terms = lexical_features(query)
    if not terms:
        return 0.0
    return len(terms & lexical_features(text)) / len(terms)
