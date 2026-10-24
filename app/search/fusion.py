import re
from dataclasses import dataclass, field

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been what which who whom "
    "whose when where why how did does do has have had that this these those as it its into than "
    "then there their they he she his her about after before during both either not no".split()
)


def keyword_terms(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    seen, out = set(), []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    ranks: dict = field(default_factory=dict)     # {"dense": 1, "sparse": 3}


def rrf_fuse(rankings: dict[str, list[Hit]], k: int = 60,
             weights: dict[str, float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion: რამდენიმე დალაგებულ სიას ერთ სიად ვაერთიანებთ."""
    weights = weights or {}
    merged: dict[str, Hit] = {}
    for name, hits in rankings.items():
        w = weights.get(name, 1.0)
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(
                h.chunk_id,
                Hit(h.chunk_id, h.doc_id, h.title, h.text),
            )
            m.score += w / (k + rank)
            m.ranks[name] = rank
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


def title_boost(hits: list[Hit], query: str, bonus: float = 0.01) -> list[Hit]:
    """Re-ranker: თუ chunk-ის სათაური კითხვაში სიტყვა-სიტყვით გვხვდება, ქულას ვუმატებთ."""
    q = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    for h in hits:
        t = " ".join(re.findall(r"[a-z0-9]+", h.title.lower()))
        if len(t) >= 3 and f" {t} " in f" {q} ":
            h.score += bonus
    return sorted(hits, key=lambda h: h.score, reverse=True)