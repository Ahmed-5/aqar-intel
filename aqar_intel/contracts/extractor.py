"""Contract field extraction.

Two extractors share one output schema so they can be evaluated head-to-head:

* ``LLMExtractor`` – prompts the model with the schema and the raw contract
  text and parses the JSON reply through ``ContractRecord``.
* ``RuleBasedExtractor`` – regex parsers for the known templates. It is the
  baseline every LLM pipeline should be measured against: cheap, fast, and
  very precise on layouts it knows, useless on anything else.

Input can be plain text, Markdown or a text-layer PDF. Scanned PDFs would go
through OCR first (e.g. Tesseract with the ``ara`` model, or a vision LLM);
that stage is a drop-in addition in ``load_contract_text``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..arabic import to_western_digits
from ..llm import LLM, extract_json, get_llm
from .schema import ContractRecord, validate_business_rules

log = logging.getLogger(__name__)


@dataclass
class Extraction:
    source: str
    record: ContractRecord
    issues: list[str] = field(default_factory=list)
    raw: str | None = None
    extractor: str = ""
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.issues) or self.error is not None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "extractor": self.extractor,
            "fields": self.record.model_dump(),
            "issues": self.issues,
            "needs_review": self.needs_review,
            "error": self.error,
        }


def load_contract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #
class LLMExtractor:
    name = "llm"

    SYSTEM = (
        "You are a document-intelligence engine for a Saudi real-estate developer. Extract the fields below from the "
        "off-plan sale contract. Return ONLY a JSON object with exactly these keys (use null when absent).\n"
        "Rules:\n"
        "- Numbers must be plain numbers (no thousands separators, no currency). Convert Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western digits.\n"
        "- Dates must be ISO YYYY-MM-DD.\n"
        "- unit_type must be one of: apartment, townhouse, duplex, villa (شقة=apartment, تاون هاوس=townhouse, دوبلكس=duplex, فيلا=villa).\n"
        "- payment_method must be one of: installments, bank_financing, cash. For cash deals set down_payment_sar = total price, "
        "installments_count = 0, installment_amount_sar = 0. For bank financing set installments_count = 1 and "
        "installment_amount_sar = the financed balance.\n"
        "- Keep names, project name and city in the language of the document.\n"
        "- Do not invent values.\n\n"
        "FIELDS:\n" + json.dumps(ContractRecord.json_schema_for_prompt(), ensure_ascii=False, indent=1)
    )

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or get_llm()

    def extract(self, text: str, source: str = "<text>") -> Extraction:
        raw = None
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": f"CONTRACT:\n{text}"}],
                json_mode=True,
                max_tokens=800,
            )
            data = extract_json(raw)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            record = ContractRecord(**{k: data.get(k) for k in ContractRecord.model_fields})
            return Extraction(source, record, validate_business_rules(record), raw, self.name)
        except (ValueError, ValidationError, RuntimeError) as e:
            log.warning("extraction failed for %s: %s", source, e)
            return Extraction(source, ContractRecord(), ["extraction failed"], raw, self.name, error=str(e))


# --------------------------------------------------------------------------- #
# Rule-based baseline
# --------------------------------------------------------------------------- #
def _g(m: re.Match | None, i: int = 1):
    return m.group(i).strip() if m else None


def _payment_ar(t: str) -> dict:
    if m := re.search(r"كامل الثمن وقدره ([\d,]+) ريال", t):
        return dict(payment_method="cash", down_payment_sar=m.group(1), installments_count=0, installment_amount_sar=0, bank_name=None)
    if m := re.search(r"دفعة مقدمة قدرها ([\d,]+) ريال عند التوقيع، ويُسدد المبلغ المتبقي وقدره ([\d,]+) ريال عن طريق التمويل العقاري من (.+?)\.", t):
        return dict(payment_method="bank_financing", down_payment_sar=m.group(1), installments_count=1, installment_amount_sar=m.group(2), bank_name=m.group(3))
    if m := re.search(r"دفعة مقدمة قدرها ([\d,]+) ريال عند التوقيع، ويُسدد المبلغ المتبقي على (\d+) أقساط متساوية قيمة كل قسط ([\d,]+) ريال", t):
        return dict(payment_method="installments", down_payment_sar=m.group(1), installments_count=m.group(2), installment_amount_sar=m.group(3), bank_name=None)
    return {}


def _parse_a(t: str) -> dict:
    d = dict(
        contract_no=_g(re.search(r"رقم العقد:\s*(\S+)", t)),
        contract_date=_g(re.search(r"تاريخ العقد:\s*([\d/]+)", t)),
    )
    if m := re.search(r"السيد/\s*(.+?)، هوية وطنية رقم (\d+)، جوال (\d+)", t):
        d.update(buyer_name=m.group(1), buyer_national_id=m.group(2), buyer_phone=m.group(3))
    if m := re.search(r"الوحدة رقم (\S+) من نوع (.+?) في مشروع (.+?) بمدينة (.+?)، بمساحة إجمالية قدرها ([\d.]+) متر", t):
        d.update(unit_no=m.group(1), unit_type=m.group(2), project_name=m.group(3), city=m.group(4), area_sqm=m.group(5))
    d["total_price_sar"] = _g(re.search(r"الثمن الإجمالي للوحدة هو ([\d,]+) ريال", t))
    d.update(_payment_ar(t))
    d["delivery_date"] = _g(re.search(r"موعد أقصاه ([\d/]+)", t))
    return d


def _parse_b(t: str) -> dict:
    d = dict(
        contract_no=_g(re.search(r"مرجع العقد:\s*(\S+)", t)),
        contract_date=_g(re.search(r"إنه في يوم (\d{1,2} \S+ \d{4})", t)),
        city=_g(re.search(r"الكائن في (.+?)،", t)),
        total_price_sar=_g(re.search(r"قيمة البيع الإجمالية ([\d,]+) ريال", t)),
        delivery_date=_g(re.search(r"موعد تسليم الوحدة:\s*(\d{1,2} \S+ \d{4})", t)),
    )
    if m := re.search(r"ثانياً:\s*(.+?)، (?:سعودي الجنسية|مقيم)، بموجب (?:الهوية|الإقامة) رقم (\d+)، هاتف (\d+)", t):
        d.update(buyer_name=m.group(1), buyer_national_id=m.group(2), buyer_phone=m.group(3))
    if m := re.search(r"يبيع المطور للمشتري (.+?) رقم (\S+) بمساحة ([\d.]+) م2 في مشروع (.+?)\.", t):
        d.update(unit_type=m.group(1), unit_no=m.group(2), area_sqm=m.group(3), project_name=m.group(4))
    d.update(_payment_ar(t))
    return d


_C_KEYS = {
    "رقم العقد": "contract_no", "تاريخ التوقيع": "contract_date", "اسم المشتري": "buyer_name", "رقم الهوية": "buyer_national_id",
    "رقم الجوال": "buyer_phone", "المشروع": "project", "رقم الوحدة": "unit_no", "نوع الوحدة": "unit_type", "المساحة": "area_sqm",
    "إجمالي الثمن": "total_price_sar", "طريقة السداد": "payment_method", "الدفعة المقدمة": "down_payment_sar",
    "عدد الأقساط": "installments_count", "قيمة القسط": "installment_amount_sar", "جهة التمويل": "bank_name", "تاريخ التسليم": "delivery_date",
}


def _parse_c(t: str) -> dict:
    d: dict = {}
    for line in t.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = _C_KEYS.get(k.strip())
        if not key:
            continue
        v = v.strip()
        if key == "project":
            parts = [x.strip() for x in v.split("–")]
            d["project_name"] = parts[0]
            if len(parts) > 1:
                d["city"] = parts[1]
        elif key in ("total_price_sar", "down_payment_sar", "installment_amount_sar"):
            d[key] = v.replace("ر.س", "").strip()
        elif key == "area_sqm":
            d[key] = v.replace("م²", "").strip()
        else:
            d[key] = v
    return d


def _parse_d(t: str) -> dict:
    d = dict(
        contract_no=_g(re.search(r"Agreement No\.:\s*(\S+)", t)),
        contract_date=_g(re.search(r"^Date:\s*(.+)$", t, re.MULTILINE)),
        total_price_sar=_g(re.search(r"total Purchase Price is SAR ([\d,]+)", t)),
        delivery_date=_g(re.search(r"no later than (\d{1,2} \w+ \d{4})", t)),
    )
    if m := re.search(r"Mr\./Ms\. (.+?), National ID No\. (\d+), Mobile (\d+)", t):
        d.update(buyer_name=m.group(1), buyer_national_id=m.group(2), buyer_phone=m.group(3))
    if m := re.search(r"sells to the Purchaser (.+?) No\. (\S+) in the (.+?) project, (.+?), with a gross area of ([\d.]+)", t):
        d.update(unit_type=m.group(1), unit_no=m.group(2), project_name=m.group(3), city=m.group(4), area_sqm=m.group(5))
    if m := re.search(r"full Purchase Price of SAR ([\d,]+) in cash", t):
        d.update(payment_method="cash", down_payment_sar=m.group(1), installments_count=0, installment_amount_sar=0, bank_name=None)
    elif m := re.search(r"down payment of SAR ([\d,]+) upon signing, and the balance of SAR ([\d,]+) through mortgage financing provided by (.+?)\.", t):
        d.update(payment_method="bank_financing", down_payment_sar=m.group(1), installments_count=1, installment_amount_sar=m.group(2), bank_name=m.group(3))
    elif m := re.search(r"down payment of SAR ([\d,]+) upon signing, and the balance in (\d+) equal instalments of SAR ([\d,]+)", t):
        d.update(payment_method="installments", down_payment_sar=m.group(1), installments_count=m.group(2), installment_amount_sar=m.group(3), bank_name=None)
    return d


class RuleBasedExtractor:
    """Template-specific regex baseline. Tries every known template and keeps the most complete parse."""

    name = "rules"
    PARSERS = (_parse_a, _parse_b, _parse_c, _parse_d)

    def extract(self, text: str, source: str = "<text>") -> Extraction:
        t = to_western_digits(text)
        best: dict = {}
        for parser in self.PARSERS:
            try:
                d = {k: v for k, v in parser(t).items() if v not in (None, "")}
            except Exception:  # noqa: BLE001 - a parser failing on a foreign layout is expected
                d = {}
            if len(d) > len(best):
                best = d
        try:
            record = ContractRecord(**{k: best.get(k) for k in ContractRecord.model_fields})
        except ValidationError as e:
            return Extraction(source, ContractRecord(), ["extraction failed"], None, self.name, error=str(e))
        return Extraction(source, record, validate_business_rules(record), None, self.name)


def extract_file(path: Path, extractor: LLMExtractor | RuleBasedExtractor) -> Extraction:
    return extractor.extract(load_contract_text(path), source=path.name)
