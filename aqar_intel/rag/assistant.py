"""Bilingual (Arabic/English) assistant over project documents + live inventory.

Flow:

1. **Route** the question to ``docs`` (policies, brochures, payment plans),
   ``inventory`` (counts, prices, availability from the database), or ``both``.
   Routing is done by the LLM with a keyword heuristic as fallback, so the
   assistant degrades gracefully when the model returns something odd.
2. **Retrieve** – hybrid search for docs, guarded text-to-SQL for inventory.
3. **Answer** in the language of the question, grounded in the retrieved
   context, citing document sources so sales staff can verify claims.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..arabic import detect_language
from ..llm import LLM, extract_json, get_llm
from .retriever import Hit, HybridRetriever
from .sql_agent import SQLAgent, SQLResult

log = logging.getLogger(__name__)

_INVENTORY_HINTS_AR = ["كم عدد", "كم وحدة", "متاح", "متاحة", "المتاحة", "متوسط", "أقل سعر", "أرخص", "أغلى", "أعلى سعر", "مبيعات", "بيعت", "مباعة", "غرف", "مساحة", "الدور", "إطلالة", "أبيع", "بيع"]
_INVENTORY_HINTS_EN = ["how many", "available", "average", "cheapest", "most expensive", "sold", "sales", "bedroom", "sqm", "floor", "view", "list of", "units under", "price of unit", "which units"]
_DOC_HINTS_AR = ["رسوم", "سياسة", "ضمان", "ضريبة", "خطة السداد", "الدفعة", "إلغاء", "تسليم", "مرافق", "تمويل", "سكني", "وافي", "مستندات", "حجز", "خصم", "تواصل", "هاتف", "عن الشركة"]
_DOC_HINTS_EN = ["fee", "policy", "warranty", "tax", "payment plan", "instalment", "installment", "cancel", "handover", "amenit", "financ", "sakani", "wafi", "document", "booking", "discount", "contact", "phone", "about the company"]
_PROJECT_PATTERNS = {
    "NRJ": ["النرجس", "narjis"], "CST": ["الشاطئ", "coast"], "RWB": ["الروابي", "rawabi"],
    "YSM": ["الياسمين", "yasmin"], "KHM": ["الخبر", "khobar"], "AQQ": ["العقيق", "aqiq"],
}


@dataclass
class Answer:
    question: str
    language: str
    route: str
    answer: str
    doc_hits: list[Hit] = field(default_factory=list)
    sql_result: SQLResult | None = None

    @property
    def sources(self) -> list[str]:
        seen, out = set(), []
        for h in self.doc_hits:
            s = h.chunk.metadata.get("source", h.chunk.doc_id)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out


def heuristic_route(question: str, lang: str) -> str:
    q = question.lower()
    inv = any(h in q for h in (_INVENTORY_HINTS_AR + _INVENTORY_HINTS_EN))
    doc = any(h in q for h in (_DOC_HINTS_AR + _DOC_HINTS_EN))
    if inv and doc:
        return "both"
    if inv:
        return "inventory"
    return "docs"


def detect_project(question: str) -> str | None:
    q = question.lower()
    for pid, pats in _PROJECT_PATTERNS.items():
        if any(p in q for p in pats):
            return pid
    return None


class Assistant:
    def __init__(self, llm: LLM | None = None, retriever: HybridRetriever | None = None, sql_agent: SQLAgent | None = None,
                 *, use_llm_router: bool = True):
        self.llm = llm or get_llm()
        self.retriever = retriever or HybridRetriever.load_or_build()
        self.sql_agent = sql_agent or SQLAgent(self.llm)
        self.use_llm_router = use_llm_router

    # ------------------------------------------------------------------ #
    def route(self, question: str, lang: str) -> str:
        fallback = heuristic_route(question, lang)
        if not self.use_llm_router or getattr(self.llm, "name", "") == "mock":
            return fallback
        system = (
            "Classify a customer/sales question for a real-estate developer.\n"
            "Return ONLY JSON: {\"route\": \"docs\" | \"inventory\" | \"both\"}.\n"
            "- docs: policies, fees, warranties, payment plans, amenities, taxes, contacts, general project descriptions.\n"
            "- inventory: counts, prices, availability, specific units, sales figures (requires querying the units database).\n"
            "- both: needs a database figure AND a policy/plan explanation."
        )
        try:
            data = extract_json(self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": question}], json_mode=True, max_tokens=50))
            r = str(data.get("route", "")).lower()
            if r in ("docs", "inventory", "both"):
                return r
        except Exception as e:  # noqa: BLE001
            log.warning("LLM router failed (%s); using heuristic", e)
        return fallback

    # ------------------------------------------------------------------ #
    def ask(self, question: str, *, k: int = 5, history: list[dict] | None = None) -> Answer:
        lang = detect_language(question)
        route = self.route(question, lang)
        project_id = detect_project(question)

        doc_hits: list[Hit] = []
        sql_result: SQLResult | None = None
        if route in ("docs", "both"):
            doc_hits = self.retriever.search(question, k=k, lang=lang, project_id=project_id)
        if route in ("inventory", "both"):
            sql_result = self.sql_agent.ask(question)

        answer = self._compose(question, lang, route, doc_hits, sql_result, history or [])
        return Answer(question, lang, route, answer, doc_hits, sql_result)

    # ------------------------------------------------------------------ #
    def _compose(self, question: str, lang: str, route: str, doc_hits: list[Hit], sql_result: SQLResult | None,
                 history: list[dict]) -> str:
        lang_name = "Arabic" if lang == "ar" else "English"
        context_parts = []
        if doc_hits:
            ctx = "\n\n".join(f"[{i + 1}] source={h.chunk.metadata.get('source')}\n{h.chunk.text}" for i, h in enumerate(doc_hits))
            context_parts.append(f"DOCUMENT EXCERPTS:\n{ctx}")
        if sql_result is not None:
            if sql_result.error:
                context_parts.append(f"DATABASE: the inventory query failed ({sql_result.error}). Say the figure is unavailable; do not guess.")
            else:
                context_parts.append(
                    f"DATABASE RESULT (SQL: {sql_result.sql}):\n{sql_result.as_markdown(max_rows=30)}"
                )
        system = (
            f"You are a sales-support assistant for a Saudi real-estate developer. Answer in {lang_name}.\n"
            "Use ONLY the provided context. If the context does not contain the answer, say so briefly and suggest "
            "contacting the sales team. Quote figures exactly (currency SAR / ريال). When you use a document excerpt, "
            "cite it as [n]. Be concise: a short paragraph or a compact bullet list; no preamble."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": f"{question}\n\nCONTEXT:\n" + "\n\n".join(context_parts)})
        return self.llm.chat(messages, temperature=0.1, max_tokens=700).strip()


def format_answer(a: Answer) -> str:
    """Plain-text rendering for CLI/demo output."""
    out = [a.answer]
    if a.sources:
        out.append(("المصادر: " if a.language == "ar" else "Sources: ") + ", ".join(a.sources))
    if a.sql_result is not None and not a.sql_result.error:
        out.append(f"SQL: {a.sql_result.sql}")
    return "\n".join(out)
