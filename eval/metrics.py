def recall_at_k(retrieved_titles: list[str], gold: set[str], k: int) -> float:
    top = set(retrieved_titles[:k])
    return len(top & gold) / len(gold) if gold else 0.0


def both_at_k(retrieved_titles: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold and gold <= set(retrieved_titles[:k]) else 0.0


def reciprocal_rank(retrieved_titles: list[str], gold: set[str]) -> float:
    for rank, t in enumerate(retrieved_titles, start=1):
        if t in gold:
            return 1.0 / rank
    return 0.0