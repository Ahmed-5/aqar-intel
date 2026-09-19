"""AqarIntel REST API.

    uvicorn api.main:app --reload --port 8000

Endpoints
---------
GET  /health
POST /ask                {"question": "...", "history": [...]}
POST /contracts/extract  {"text": "...", "extractor": "llm"|"rules"}
POST /contracts/extract-file   multipart file upload (.txt/.md/.pdf)
GET  /analytics/kpis
POST /analytics/price    {"project_id": "NRJ", "unit_type": "villa", "area_sqm": 400, ...}
POST /analytics/report   -> builds charts + bilingual executive report
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from aqar_intel.analytics.pricing import load_pricing_model, predict_price  # noqa: E402
from aqar_intel.analytics.report import build_report, kpi_snapshot  # noqa: E402
from aqar_intel.config import REPORTS_DIR, get_settings  # noqa: E402
from aqar_intel.contracts.extractor import LLMExtractor, RuleBasedExtractor  # noqa: E402
from aqar_intel.llm import get_llm  # noqa: E402

app = FastAPI(title="AqarIntel API", version="0.1.0", description="AI services for real-estate development (demo, simulated data).")


@lru_cache
def _assistant():
    from aqar_intel.rag.assistant import Assistant

    return Assistant()


@lru_cache
def _llm():
    return get_llm()


# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    history: list[dict] = Field(default_factory=list, description="Prior turns as [{role, content}]")


class AskResponse(BaseModel):
    answer: str
    language: str
    route: str
    sources: list[str]
    sql: str | None = None
    rows: list[list] | None = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=20)
    extractor: Literal["llm", "rules"] = "llm"


class PriceRequest(BaseModel):
    project_id: str
    city: str | None = None
    unit_type: Literal["apartment", "townhouse", "duplex", "villa"]
    area_sqm: float = Field(..., gt=30, lt=2000)
    bedrooms: int = 3
    bathrooms: int | None = None
    view: Literal["garden", "street", "city", "sea", "park"] = "street"
    floor: int | None = None
    parking_spaces: int = 2


_CITY = {"NRJ": "Riyadh", "YSM": "Riyadh", "CST": "Jeddah", "RWB": "Dammam", "KHM": "Al Khobar", "AQQ": "Madinah"}


# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm": s.chat_model if s.use_openrouter else "mock", "embeddings": s.embedding_model if s.use_openrouter else "hashing"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    a = _assistant().ask(req.question, history=req.history)
    sql = rows = None
    if a.sql_result is not None and not a.sql_result.error:
        sql, rows = a.sql_result.sql, [list(r) for r in a.sql_result.rows[:50]]
    return AskResponse(answer=a.answer, language=a.language, route=a.route, sources=a.sources, sql=sql, rows=rows)


@app.post("/contracts/extract")
def extract(req: ExtractRequest):
    ex = (LLMExtractor(_llm()) if req.extractor == "llm" else RuleBasedExtractor()).extract(req.text)
    return ex.to_dict()


@app.post("/contracts/extract-file")
async def extract_file_upload(file: UploadFile = File(...), extractor: Literal["llm", "rules"] = "llm"):
    data = await file.read()
    if file.filename and file.filename.lower().endswith(".pdf"):
        import io

        from pypdf import PdfReader

        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text.strip()) < 20:
        raise HTTPException(400, "could not read text from file (scanned PDF? run OCR first)")
    ex = (LLMExtractor(_llm()) if extractor == "llm" else RuleBasedExtractor()).extract(text, source=file.filename or "upload")
    return ex.to_dict()


@app.get("/analytics/kpis")
def kpis():
    return kpi_snapshot()


@app.post("/analytics/price")
def price(req: PriceRequest):
    try:
        model = load_pricing_model()
    except FileNotFoundError:
        raise HTTPException(409, "pricing model not trained yet; call POST /analytics/report first")
    spec = req.model_dump()
    spec["city"] = spec["city"] or _CITY.get(req.project_id, "Riyadh")
    spec["bathrooms"] = spec["bathrooms"] or max(1, req.bedrooms - 1)
    p = predict_price(model, spec)
    return {"estimated_price_sar": round(p), "price_per_sqm": round(p / req.area_sqm)}


@app.post("/analytics/report")
def report():
    out = build_report(REPORTS_DIR, llm=_llm())
    return {"reports": out["reports"], "charts": out["charts"], "pricing_metrics": out["pricing"]["model"]}
