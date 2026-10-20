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

def prepare_lexical_scorer(query: str):
    """Compile query features once and score many fragments without a global cache.

    Exact score-equivalent to lexical_relevance(query, text): words are
    casefolded and Korean Hangul sequences generate the same syllable bigrams.
    Temporary query features are scoped to the returned callable, avoiding a
    persistent cache of potentially private search queries.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    terms = lexical_features(query)
    denominator = len(terms)
    if denominator == 0:
        return lambda text: 0.0

    # Hangul bigrams cannot match a query without any 2+ Hangul features.
    # Skipping this regex for English-only queries is safe and measurable.
    needs_hangul = any(_HANGUL.search(term) for term in terms)

    def score(text: str) -> float:
        matched: set[str] = set()
        for word in _WORD.findall(text.casefold()):
            if word in terms:
                matched.add(word)
            if needs_hangul:
                for match in _HANGUL.finditer(word):
                    hangul = match.group()
                    for i in range(len(hangul) - 1):
                        bigram = hangul[i:i + 2]
                        if bigram in terms:
                            matched.add(bigram)
        return len(matched) / denominator

    return score
