import json

from aqar_intel.config import DOCS_DIR
from aqar_intel.llm import FakeLLM, MockLLM
from aqar_intel.rag.assistant import Assistant, detect_project, heuristic_route
from aqar_intel.rag.chunking import chunk_directory
from aqar_intel.rag.embeddings import HashingEmbeddings
from aqar_intel.rag.retriever import HybridRetriever
from aqar_intel.rag.sql_agent import SQLAgent, UnsafeSQL, validate_sql
import pytest


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever.build(DOCS_DIR, HashingEmbeddings())


def test_chunks_carry_metadata():
    chunks = chunk_directory(DOCS_DIR)
    assert len(chunks) > 50
    c = next(ch for ch in chunks if ch.doc_id.startswith("payment_NRJ_ar"))
    assert c.metadata["lang"] == "ar" and c.metadata["project_id"] == "NRJ"


def test_retrieval_finds_booking_fee_chunk(retriever):
    hits = retriever.search("ما هي رسوم الحجز؟", k=3, lang="ar")
    assert any("رسوم الحجز" in h.chunk.text for h in hits)
    assert all(h.chunk.metadata["lang"] == "ar" for h in hits)


def test_retrieval_english_warranty(retriever):
    hits = retriever.search("How long is the structural warranty?", k=3, lang="en")
    assert any("Structural warranty" in h.chunk.text for h in hits)


def test_index_roundtrip(tmp_path, retriever):
    retriever.save(tmp_path)
    r2 = HybridRetriever.load(tmp_path, HashingEmbeddings())
    assert len(r2.chunks) == len(retriever.chunks)


@pytest.mark.parametrize("bad", ["DROP TABLE units", "SELECT 1; DELETE FROM units", "PRAGMA table_info(units)",
                                 "UPDATE units SET status='x'", "ATTACH DATABASE 'x' AS y"])
def test_sql_guardrails_block(bad):
    with pytest.raises(UnsafeSQL):
        validate_sql(bad)


def test_sql_limit_is_enforced():
    assert validate_sql("select * from units").endswith("LIMIT 50")
    assert "LIMIT 50" in validate_sql("select * from units limit 5000")


def test_sql_agent_executes_and_repairs():
    good = json.dumps({"sql": "SELECT COUNT(*) AS n FROM units WHERE status='available'", "explanation": "count"})
    res = SQLAgent(FakeLLM([good])).ask("how many available units")
    assert res.rows[0][0] > 0 and res.attempts == 1
    res2 = SQLAgent(FakeLLM([json.dumps({"sql": "SELECT nope FROM units"}), good])).ask("x")
    assert res2.attempts == 2 and res2.error is None


def test_routing_and_project_detection():
    assert heuristic_route("كم عدد الفلل المتاحة في واحة النرجس؟", "ar") == "inventory"
    assert heuristic_route("ما هي سياسة الإلغاء؟", "ar") == "docs"
    assert heuristic_route("How many apartments are available and what is the payment plan?", "en") == "both"
    assert detect_project("prices in Coast Towers") == "CST"


def test_assistant_end_to_end_offline(retriever):
    sql_llm = FakeLLM([json.dumps({"sql": "SELECT COUNT(*) AS n FROM units WHERE unit_type='villa' AND status='available'"})])
    a = Assistant(MockLLM(), retriever, SQLAgent(sql_llm))
    ans = a.ask("كم عدد الفلل المتاحة؟")
    assert ans.route == "inventory" and ans.sql_result.rows[0][0] > 0
    ans2 = a.ask("What documents are required to sign the contract?")
    assert ans2.route == "docs" and ans2.sources
