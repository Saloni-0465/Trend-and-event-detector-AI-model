"""Simple frequency-based pseudo-topics (baseline)."""


def top_terms_by_frequency(
    token_lists: list[list[str]], k: int = 20
) -> list[tuple[str, int]]:
    """Return top-k tokens by count across documents."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for doc in token_lists:
        counts.update(doc)
    return counts.most_common(k)
