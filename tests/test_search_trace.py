import pytest

from main.core.search_trace import (
    NULL_COLLECTION_TRACE,
    NULL_TRACE,
    SCHEMA_VERSION,
    CollectionTrace,
    NullCollectionTrace,
    NullSearchTrace,
    SearchTrace,
    create_trace,
)


class TestCreateTrace:
    def test_enabled_returns_real_trace(self):
        t = create_trace(True)
        assert isinstance(t, SearchTrace)
        assert t.enabled is True

    def test_disabled_returns_shared_null(self):
        t = create_trace(False)
        assert t is NULL_TRACE
        assert t.enabled is False


class TestSearchTraceQueryFields:
    def test_raw_defaults_expanded(self):
        t = SearchTrace()
        t.set_query_raw("hva er LA_BUC_02")
        d = t.to_dict()
        assert d["query"]["raw"] == "hva er LA_BUC_02"
        assert d["query"]["expanded"] == "hva er LA_BUC_02"

    def test_expansion_overrides_expanded(self):
        t = SearchTrace()
        t.set_query_raw("hva er LA_BUC_02")
        t.set_expansion("hva er LA_BUC_02 A003 A004", ["A003", "A004"])
        d = t.to_dict()
        assert d["query"]["expanded"] == "hva er LA_BUC_02 A003 A004"
        assert d["query"]["expansionTerms"] == ["A003", "A004"]

    def test_detected_entities_collected(self):
        t = SearchTrace()
        t.add_detected_entity("BUC:LA_BUC_02", "BUC", "LA_BUC_02", "LA_BUC_02")
        t.add_detected_entity("SED:A003", "SED", "A003", "A003")
        d = t.to_dict()
        assert len(d["query"]["detectedEntities"]) == 2
        assert d["query"]["detectedEntities"][0]["matchedSpan"] == "LA_BUC_02"

    def test_graph_answered_and_rerank_skipped(self):
        t = SearchTrace()
        t.set_graph_answered(True)
        t.set_reranker_skipped(True, reason="brief")
        d = t.to_dict()
        assert d["query"]["graphAnswered"] is True
        assert d["query"]["rerankerSkipped"] is True
        assert d["query"]["rerankerSkipReason"] == "brief"

    def test_schema_version_stamped(self):
        assert SearchTrace().to_dict()["schemaVersion"] == SCHEMA_VERSION


class TestCollectionTraceStages:
    def test_record_stage_creates_candidate(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.record_stage("faiss", chunk_id=42, rank=3, score=0.412)
        d = c.to_dict()
        assert d["candidates"][0] == {
            "chunkId": 42,
            "documentId": None,
            "docTitle": None,
            "headings": None,
            "stages": {"faiss": {"rank": 3, "score": pytest.approx(0.412)}},
            "kept": True,
            "dropReason": None,
        }

    def test_multiple_stages_same_chunk(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.record_stage("faiss", chunk_id=42, rank=3, score=0.412)
        c.record_stage("bm25", chunk_id=42, rank=1, score=8.91)
        c.record_stage("rrf", chunk_id=42, rank=1, score=-0.0331)
        c.record_stage("ce", chunk_id=42, rank=2, score=4.21)
        c.record_stage("final", chunk_id=42, rank=1, score=-0.20)
        stages = c.to_dict()["candidates"][0]["stages"]
        assert set(stages.keys()) == {"faiss", "bm25", "rrf", "ce", "final"}
        assert stages["rrf"]["score"] == pytest.approx(-0.0331)

    def test_unknown_stage_rejected(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        with pytest.raises(ValueError, match="unknown stage"):
            c.record_stage("nonsense", chunk_id=1, rank=1, score=0.0)

    def test_annotate_candidate(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.record_stage("faiss", chunk_id=42, rank=1, score=0.5)
        c.annotate_candidate(42, document_id="doc-7", doc_title="My Doc",
                             headings=["Section A", "Subsection"])
        cand = c.to_dict()["candidates"][0]
        assert cand["documentId"] == "doc-7"
        assert cand["docTitle"] == "My Doc"
        assert cand["headings"] == ["Section A", "Subsection"]


class TestCollectionTraceTitleBoost:
    def test_boost_attached_to_candidates_of_same_doc(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.record_stage("final", chunk_id=1, rank=1, score=-0.2)
        c.record_stage("final", chunk_id=2, rank=2, score=-0.1)
        c.annotate_candidate(1, document_id="doc-A")
        c.annotate_candidate(2, document_id="doc-B")
        c.record_title_boost("doc-A", delta=-0.18)

        candidates = {cand["chunkId"]: cand for cand in c.to_dict()["candidates"]}
        assert candidates[1]["stages"]["titleBoost"] == {"applied": True, "delta": pytest.approx(-0.18)}
        assert "titleBoost" not in candidates[2]["stages"]


class TestCollectionTraceDropReason:
    def test_mark_dropped_sets_kept_false(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.record_stage("rrf", chunk_id=99, rank=10, score=-0.001)
        c.mark_dropped(99, reason="dedup")
        cand = c.to_dict()["candidates"][0]
        assert cand["kept"] is False
        assert cand["dropReason"] == "dedup"

    def test_unknown_reason_rejected(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        with pytest.raises(ValueError, match="unknown drop reason"):
            c.mark_dropped(1, reason="because")


class TestCollectionTraceConfidenceAndTimings:
    def test_confidence_block(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.set_confidence(low_confidence=False, best_score=-0.20,
                         low_confidence_threshold=-0.10, noise_threshold=-0.01,
                         filtered_count=2)
        conf = c.to_dict()["confidence"]
        assert conf == {
            "lowConfidence": False,
            "bestScore": pytest.approx(-0.20),
            "lowConfidenceThreshold": pytest.approx(-0.10),
            "noiseThreshold": pytest.approx(-0.01),
            "filteredCount": 2,
        }

    def test_confidence_with_no_results(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.set_confidence(low_confidence=True, best_score=None,
                         low_confidence_threshold=-0.10, noise_threshold=-0.01,
                         filtered_count=0)
        assert c.to_dict()["confidence"]["bestScore"] is None
        assert c.to_dict()["confidence"]["lowConfidence"] is True

    def test_timings_coerced_to_int(self):
        c = CollectionTrace(name="x", indexer="hybrid", fetch_k=10)
        c.set_timings(indexFetch=14.7, chunkLoad=3, rerank=41.2, total=63.9)
        t = c.to_dict()["timingsMs"]
        assert t == {"indexFetch": 14, "chunkLoad": 3, "rerank": 41, "total": 63}


class TestSearchTraceCollectionWiring:
    def test_start_collection_attaches(self):
        t = SearchTrace()
        coll = t.start_collection(name="melosys-confluence-v3", indexer="hybrid", fetch_k=22)
        coll.record_stage("faiss", chunk_id=1, rank=1, score=0.5)
        d = t.to_dict()
        assert len(d["collections"]) == 1
        assert d["collections"][0]["name"] == "melosys-confluence-v3"
        assert d["collections"][0]["fetchK"] == 22
        assert d["collections"][0]["candidates"][0]["chunkId"] == 1

    def test_total_ms_present(self):
        t = SearchTrace()
        t.set_query_raw("q")
        d = t.to_dict()
        assert isinstance(d["totalMs"], int)
        assert d["totalMs"] >= 0


class TestNullVariants:
    def test_null_trace_methods_noop(self):
        # Just verify nothing raises.
        NULL_TRACE.set_query_raw("q")
        NULL_TRACE.set_expansion("q x", ["x"])
        NULL_TRACE.add_detected_entity("e", "t", "l", "s")
        NULL_TRACE.set_graph_answered(True)
        NULL_TRACE.set_reranker_skipped(True, "brief")
        assert NULL_TRACE.to_dict() is None
        assert NULL_TRACE.enabled is False

    def test_null_start_collection_returns_null_collection(self):
        coll = NULL_TRACE.start_collection("c", "hybrid", 10)
        assert coll is NULL_COLLECTION_TRACE
        assert coll.enabled is False

    def test_null_collection_methods_noop(self):
        NULL_COLLECTION_TRACE.record_stage("faiss", 1, 1, 0.5)
        NULL_COLLECTION_TRACE.annotate_candidate(1, document_id="d")
        NULL_COLLECTION_TRACE.record_title_boost("d", -0.1)
        NULL_COLLECTION_TRACE.mark_dropped(1, "noise")
        NULL_COLLECTION_TRACE.set_confidence(False, 0.0, -0.1, -0.01, 0)
        NULL_COLLECTION_TRACE.set_timings(total=10)
        assert NULL_COLLECTION_TRACE.to_dict() is None

    def test_disabled_factory_is_null_singleton(self):
        a = create_trace(False)
        b = create_trace(False)
        assert a is b is NULL_TRACE
