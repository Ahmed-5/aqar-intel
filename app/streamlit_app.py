"""AqarIntel – Streamlit demo UI.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from aqar_intel.config import CONTRACTS_DIR, REPORTS_DIR, get_settings  # noqa: E402
from aqar_intel.contracts.extractor import LLMExtractor, RuleBasedExtractor  # noqa: E402
from aqar_intel.llm import get_llm  # noqa: E402

st.set_page_config(page_title="AqarIntel", page_icon="🏗️", layout="wide")
settings = get_settings()

st.title("🏗️ AqarIntel — AI for Real Estate Development")
mode = f"OpenRouter · {settings.chat_model}" if settings.use_openrouter else "offline mock (set OPENROUTER_API_KEY)"
st.caption(f"LLM: {mode} · all data is simulated")


@st.cache_resource(show_spinner="Loading models and index…")
def _assistant():
    from aqar_intel.rag.assistant import Assistant

    return Assistant()


@st.cache_resource
def _llm():
    return get_llm()


tab_chat, tab_docs, tab_analytics = st.tabs(["💬 Sales assistant", "📄 Contract intelligence", "📈 Sales analytics"])

# --------------------------------------------------------------------------- #
with tab_chat:
    st.markdown("Ask in **Arabic or English** about policies, payment plans, or live inventory (counts, prices, availability).")
    examples = [
        "ما هي رسوم الحجز وهل تُسترد؟",
        "كم عدد الفلل المتاحة في واحة النرجس وما أقل سعر؟",
        "What is the structural warranty period?",
        "Which project has the most available apartments under SAR 1.5M?",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state["pending_q"] = ex

    if "history" not in st.session_state:
        st.session_state["history"] = []
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                with st.expander("Sources / SQL"):
                    st.markdown(turn["meta"])

    q = st.chat_input("اكتب سؤالك هنا / Type your question") or st.session_state.pop("pending_q", None)
    if q:
        st.session_state["history"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("…"):
                hist = [{"role": t["role"], "content": t["content"]} for t in st.session_state["history"][:-1]]
                ans = _assistant().ask(q, history=hist)
            st.markdown(ans.answer)
            meta = []
            if ans.sources:
                meta.append("**Sources:** " + ", ".join(f"`{s}`" for s in ans.sources))
            if ans.sql_result is not None:
                meta.append(f"**SQL:** `{ans.sql_result.sql}`\n\n" + ans.sql_result.as_markdown())
            if meta:
                with st.expander("Sources / SQL"):
                    st.markdown("\n\n".join(meta))
            st.session_state["history"].append({"role": "assistant", "content": ans.answer, "meta": "\n\n".join(meta)})

# --------------------------------------------------------------------------- #
with tab_docs:
    st.markdown("Extract structured fields from a sale contract, validate them, and flag records for human review.")
    left, right = st.columns([1, 1])
    with left:
        samples = sorted(p.name for p in CONTRACTS_DIR.glob("*.txt"))
        choice = st.selectbox("Sample contract", samples, index=0)
        uploaded = st.file_uploader("…or upload your own (.txt / .md / .pdf)", type=["txt", "md", "pdf"])
        extractor_name = st.radio("Extractor", ["LLM", "Rule-based baseline"], horizontal=True)
        if uploaded is not None:
            if uploaded.name.lower().endswith(".pdf"):
                from pypdf import PdfReader

                text = "\n".join((p.extract_text() or "") for p in PdfReader(uploaded).pages)
            else:
                text = uploaded.read().decode("utf-8")
            source = uploaded.name
        else:
            text = (CONTRACTS_DIR / choice).read_text(encoding="utf-8")
            source = choice
        st.text_area("Contract text", text, height=420)
    with right:
        if st.button("Extract fields", type="primary"):
            extractor = LLMExtractor(_llm()) if extractor_name == "LLM" else RuleBasedExtractor()
            with st.spinner("Extracting…"):
                ex = extractor.extract(text, source=source)
            if ex.needs_review:
                st.warning("⚠️ Needs human review: " + "; ".join(ex.issues + ([ex.error] if ex.error else [])))
            else:
                st.success("✅ Record is internally consistent")
            st.json(ex.record.model_dump())
            gt_path = CONTRACTS_DIR / "ground_truth.json"
            if uploaded is None and gt_path.exists():
                truth = json.loads(gt_path.read_text(encoding="utf-8")).get(choice)
                if truth:
                    from aqar_intel.contracts.evaluate import FIELDS, field_match

                    pred = ex.record.model_dump()
                    rows = [(f, pred.get(f), truth.get(f), "✅" if field_match(f, pred.get(f), truth.get(f)) else "❌") for f in FIELDS]
                    st.markdown("**Against ground truth**")
                    st.dataframe({"field": [r[0] for r in rows], "extracted": [str(r[1]) for r in rows],
                                  "truth": [str(r[2]) for r in rows], "ok": [r[3] for r in rows]},
                                 use_container_width=True, hide_index=True, height=35 * (len(rows) + 1) + 3)
        eval_path = REPORTS_DIR / "extraction_eval.md"
        if eval_path.exists():
            with st.expander("Latest batch evaluation (scripts/run_demo.py evaluate)"):
                st.markdown(eval_path.read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- #
with tab_analytics:
    from aqar_intel.analytics.report import kpi_snapshot

    snap = kpi_snapshot()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Units sold", f"{snap['units']['sold']:,}", f"{snap['units']['sold_pct']}% of portfolio")
    k2.metric("Available", f"{snap['units']['available']:,}")
    k3.metric("Revenue (SAR)", f"{snap['revenue_sar_total'] / 1e6:,.0f}M")
    delta = snap["last_3_months"]["units_sold"] - snap["last_3_months"]["units_sold_previous_3_months"]
    k4.metric("Sold last 3 months", snap["last_3_months"]["units_sold"], f"{delta:+d} vs prior 3 months")

    st.dataframe(snap["by_project"], use_container_width=True, hide_index=True)

    if st.button("Train models, forecast and write executive report", type="primary"):
        from aqar_intel.analytics.report import build_report

        with st.spinner("Training pricing model, backtesting forecasts, writing report…"):
            out = build_report(REPORTS_DIR, llm=_llm())
        st.success("Done")

    charts = [REPORTS_DIR / n for n in ("absorption.png", "price_per_sqm.png", "inventory_status.png")]
    if all(c.exists() for c in charts):
        c1, c2 = st.columns(2)
        c1.image(str(charts[0]), use_container_width=True)
        c2.image(str(charts[1]), use_container_width=True)
        st.image(str(charts[2]), width=600)
    for lang, label in (("ar", "التقرير التنفيذي (عربي)"), ("en", "Executive report (English)")):
        p = REPORTS_DIR / f"sales_report_{lang}.md"
        if p.exists():
            with st.expander(label, expanded=(lang == "ar")):
                text = p.read_text(encoding="utf-8")
                # Show only the LLM summary; the charts/metrics appendix is rendered above. Models often emit
                # "---" rules inside the summary, so split on the explicit marker written by build_report.
                st.markdown(text.split("<!-- appendix -->")[0] if "<!-- appendix -->" in text else text)

    st.markdown("#### Price a unit")
    with st.form("price_form"):
        c = st.columns(6)
        project_id = c[0].selectbox("Project", ["NRJ", "CST", "RWB", "YSM", "KHM", "AQQ"])
        unit_type = c[1].selectbox("Type", ["apartment", "townhouse", "duplex", "villa"])
        area = c[2].number_input("Area (sqm)", 60, 800, 250)
        beds = c[3].number_input("Bedrooms", 1, 7, 4)
        view = c[4].selectbox("View", ["garden", "street", "city", "sea", "park"])
        floor = c[5].number_input("Floor (0 = n/a)", 0, 40, 0)
        if st.form_submit_button("Estimate price"):
            from aqar_intel.analytics.pricing import load_pricing_model, predict_price

            city = {"NRJ": "Riyadh", "YSM": "Riyadh", "CST": "Jeddah", "RWB": "Dammam", "KHM": "Al Khobar", "AQQ": "Madinah"}[project_id]
            try:
                model = load_pricing_model()
                price = predict_price(model, dict(project_id=project_id, city=city, unit_type=unit_type, view=view, bedrooms=beds,
                                                  bathrooms=max(1, beds - 1), area_sqm=area, floor=floor or None, parking_spaces=2))
                st.metric("Estimated sale price", f"SAR {price:,.0f}", f"≈ SAR {price / area:,.0f} / sqm")
            except FileNotFoundError:
                st.info("Train the model first (button above).")
