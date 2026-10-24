import hashlib
import re
from dataclasses import dataclass

import numpy as np

from app.ingestion.models import Document, normalize_text

_MERSENNE = (1 << 61) - 1          # დიდი მარტივი რიცხვი hash-ისთვის
_MAX_HASH = (1 << 32) - 1


def shingles(text: str, k: int = 3) -> set[int]:
    """ტექსტი -> k-სიტყვიანი ჯგუფების 32-ბიტიანი hash-ების სიმრავლე."""
    words = re.findall(r"\w+", normalize_text(text).lower())
    if len(words) < k:
        words = words + [""] * (k - len(words))
    out = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i:i + k]).encode("utf-8")
        out.add(int.from_bytes(hashlib.md5(gram).digest()[:4], "little"))
    return out


class MinHasher:
    """თითო დოკუმენტს აქცევს num_perm რიცხვიან ხელმოწერად."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        # ვქმნით num_perm ცალი ხაზოვან hash ფუნქციას: (a*x + b) mod p
        self.a = rng.randint(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        x = np.fromiter(shingle_set, dtype=np.uint64)[:, None]      # (n, 1)
        hashed = (x * self.a + self.b) % _MERSENNE & _MAX_HASH      # (n, num_perm)
        return hashed.min(axis=0)                                   # თითო ფუნქციის მინიმუმი


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """ხელმოწერებში რამდენი პოზიცია ემთხვევა = Jaccard-ის შეფასება."""
    return float(np.mean(sig_a == sig_b))


@dataclass
class DedupResult:
    unique: list[Document]
    exact_duplicates: list[tuple[str, str]]     # (წაშლილი doc_id, დარჩენილი doc_id)
    near_duplicates: list[tuple[str, str, float]]


class Deduplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = 32) -> None:
        if num_perm % bands:
            raise ValueError("num_perm უნდა იყოფოდეს bands-ზე")
        self.threshold = threshold
        self.hasher = MinHasher(num_perm)
        self.bands = bands
        self.rows = num_perm // bands

    def run(self, docs: list[Document]) -> DedupResult:
        seen_hash: dict[str, str] = {}                      # hash -> doc_id
        buckets: list[dict[bytes, list[int]]] = [{} for _ in range(self.bands)]
        kept: list[Document] = []
        sigs: list[np.ndarray] = []
        exact, near = [], []

        for doc in docs:
            # დონე 1: ზუსტი დუბლიკატი -> ვშლით
            if doc.content_hash in seen_hash:
                exact.append((doc.doc_id, seen_hash[doc.content_hash]))
                continue

            # დონე 2: თითქმის დუბლიკატი -> ვინახავთ, ოღონდ ვნიშნავთ
            sig = self.hasher.signature(shingles(doc.text))
            match = self._find_match(sig, buckets, sigs)
            if match is not None:
                idx, score = match
                near.append((doc.doc_id, kept[idx].doc_id, score))
                doc.metadata["near_duplicate_of"] = kept[idx].doc_id
                doc.metadata["similarity"] = round(score, 3)

            seen_hash[doc.content_hash] = doc.doc_id
            idx = len(kept)
            kept.append(doc)
            sigs.append(sig)
            for b in range(self.bands):
                key = sig[b * self.rows:(b + 1) * self.rows].tobytes()
                buckets[b].setdefault(key, []).append(idx)

        return DedupResult(kept, exact, near)

    def _find_match(self, sig, buckets, sigs):
        candidates: set[int] = set()
        for b in range(self.bands):
            key = sig[b * self.rows:(b + 1) * self.rows].tobytes()
            candidates.update(buckets[b].get(key, []))
        best = None
        for idx in candidates:
            score = estimate_jaccard(sig, sigs[idx])
            if score >= self.threshold and (best is None or score > best[1]):
                best = (idx, score)
        return best