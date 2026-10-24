from app.ingestion.dedup import Deduplicator, estimate_jaccard, MinHasher, shingles
from app.ingestion.models import Document


def _doc(i, text):
    return Document(doc_id=str(i), title=str(i), text=text, source="t", format="txt")


BASE = ("The Eiffel Tower is a wrought iron lattice tower on the Champ de Mars in Paris "
        "France it is named after the engineer Gustave Eiffel whose company designed "
        "and built the tower from 1887 to 1889 as the entrance to the 1889 World's Fair")


def test_exact_duplicate_detected():
    res = Deduplicator().run([_doc(1, BASE), _doc(2, BASE + "  ")])
    assert len(res.unique) == 1 and len(res.exact_duplicates) == 1


def test_near_duplicate_flagged_not_removed():
    changed = BASE.replace("1887 to 1889", "1887 to 1890")
    res = Deduplicator(threshold=0.7).run([_doc(1, BASE), _doc(2, changed)])
    assert len(res.unique) == 2
    assert len(res.near_duplicates) == 1
    assert res.unique[1].metadata["near_duplicate_of"] == "1"

def test_different_documents_kept():
    other = "Python is a high level programming language created by Guido van Rossum " * 3
    res = Deduplicator().run([_doc(1, BASE), _doc(2, other)])
    assert len(res.unique) == 2


def test_minhash_approximates_jaccard():
    a, b = shingles(BASE), shingles(BASE.replace("1887 to 1889", "1887 to 1890"))
    true_j = len(a & b) / len(a | b)
    h = MinHasher(num_perm=256)
    est = estimate_jaccard(h.signature(a), h.signature(b))
    assert abs(true_j - est) < 0.15