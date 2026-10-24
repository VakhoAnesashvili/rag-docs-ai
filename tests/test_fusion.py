from app.search.fusion import Hit, keyword_terms, rrf_fuse, title_boost


def h(i, title="t"):
    return Hit(chunk_id=str(i), doc_id=str(i), title=title, text="")


def test_rrf_prefers_docs_in_both_lists():
    res = rrf_fuse({"dense": [h(1), h(2), h(3)], "sparse": [h(3), h(4), h(1)]})
    assert [x.chunk_id for x in res[:2]] == ["1", "3"]


def test_rrf_scores_match_formula():
    res = rrf_fuse({"dense": [h(1)], "sparse": [h(1)]}, k=60)
    assert abs(res[0].score - 2 / 61) < 1e-9


def test_keyword_terms_drops_stopwords_and_dupes():
    assert keyword_terms("What was the Eiffel Tower, the tower?") == ["eiffel", "tower"]


def test_title_boost_moves_matching_title_up():
    hits = [h(1, "Paris"), h(2, "Iqbal F. Qadir")]
    hits[0].score, hits[1].score = 0.02, 0.019
    out = title_boost(hits, "What was Iqbal F. Qadir on?")
    assert out[0].chunk_id == "2"