from app.search.bm25 import BM25Index

ROWS = [
    ("a::0", "a", "Dwarka", "Dwarka is a city on the Okhamandal Peninsula in Gujarat"),
    ("b::0", "b", "Railway", "The station is a railway station and the station has platforms"),
    ("c::0", "c", "Radar", "A radar station on the western shore"),
]


def _index():
    i = BM25Index()
    i.build(ROWS)
    return i


def test_rare_term_beats_common_term():
    res = _index().search("station Okhamandal")
    assert res[0].chunk_id == "a::0"


def test_idf_rare_greater_than_common():
    i = _index()
    assert i.idf("okhamandal") > i.idf("station")


def test_unknown_query_returns_empty():
    assert _index().search("zzzz") == []