"""Contract extraction schema and business-rule validation.

The Pydantic model is the *contract* between the LLM and the rest of the
system: it normalises types (Arabic-Indic digits → numbers, dd/mm/yyyy →
ISO), and ``validate_business_rules`` flags records that are internally
inconsistent (e.g. instalments that do not sum to the price) so a human
reviews them instead of silently loading bad data into the ERP.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..arabic import to_western_digits

UnitType = Literal["apartment", "townhouse", "duplex", "villa"]
PaymentMethod = Literal["installments", "bank_financing", "cash"]

_UNIT_TYPE_ALIASES = {
    "شقة": "apartment", "شقه": "apartment", "apartment": "apartment", "flat": "apartment",
    "تاون هاوس": "townhouse", "تاونهاوس": "townhouse", "townhouse": "townhouse", "town house": "townhouse",
    "دوبلكس": "duplex", "duplex": "duplex",
    "فيلا": "villa", "فيلة": "villa", "villa": "villa",
}
_PAYMENT_ALIASES = {
    "installments": "installments", "instalments": "installments", "أقساط": "installments", "اقساط": "installments",
    "bank_financing": "bank_financing", "bank financing": "bank_financing", "mortgage": "bank_financing",
    "تمويل بنكي": "bank_financing", "تمويل عقاري": "bank_financing", "تمويل": "bank_financing",
    "cash": "cash", "نقدي": "cash", "نقداً": "cash", "نقدا": "cash",
}
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
_MONTHS.update({m: i for i, m in enumerate(
    ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"], 1)})
_MONTHS.update({"ابريل": 4, "اغسطس": 8})


def parse_number(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = to_western_digits(str(v))
    s = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    return float(s) if s not in ("", "-", ".") else None


def parse_date(v) -> str | None:
    """Accept ISO, dd/mm/yyyy, '14 March 2026', '١٤ مارس ٢٠٢٦' → 'YYYY-MM-DD'."""
    if v is None or v == "":
        return None
    s = to_western_digits(str(v)).strip().rstrip("م").strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d).isoformat()
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        return date(y, mo, d).isoformat()
    m = re.match(r"^(\d{1,2})\s+([^\s\d]+)\s+(\d{4})", s)
    if m and m.group(2).lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1).lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))).isoformat()
    return s  # leave as-is; validation will flag it


class ContractRecord(BaseModel):
    contract_no: str | None = Field(None, description="Contract/agreement reference number, e.g. AUF-2026-0143")
    contract_date: str | None = Field(None, description="Signing date, ISO YYYY-MM-DD")
    buyer_name: str | None = Field(None, description="Buyer full name as written")
    buyer_national_id: str | None = Field(None, description="10-digit national ID / Iqama number")
    buyer_phone: str | None = Field(None, description="Buyer mobile number")
    project_name: str | None = Field(None, description="Project name as written in the contract")
    city: str | None = Field(None, description="City of the project as written")
    unit_no: str | None = Field(None, description="Unit number, e.g. NRJ-012")
    unit_type: UnitType | None = Field(None, description="apartment | townhouse | duplex | villa")
    area_sqm: float | None = Field(None, description="Gross area in square metres")
    total_price_sar: float | None = Field(None, description="Total price in SAR (number)")
    payment_method: PaymentMethod | None = Field(None, description="installments | bank_financing | cash")
    down_payment_sar: float | None = Field(None, description="Down payment in SAR; equals total price for cash deals")
    installments_count: int | None = Field(None, description="Number of instalments (0 for cash, 1 for bank financing balance)")
    installment_amount_sar: float | None = Field(None, description="Amount of each instalment in SAR (0 for cash)")
    bank_name: str | None = Field(None, description="Financing bank if any, else null")
    delivery_date: str | None = Field(None, description="Handover date, ISO YYYY-MM-DD")

    # --- normalisers ---------------------------------------------------- #
    @field_validator("area_sqm", "total_price_sar", "down_payment_sar", "installment_amount_sar", mode="before")
    @classmethod
    def _num(cls, v):
        return parse_number(v)

    @field_validator("installments_count", mode="before")
    @classmethod
    def _int(cls, v):
        n = parse_number(v)
        return int(n) if n is not None else None

    @field_validator("contract_date", "delivery_date", mode="before")
    @classmethod
    def _date(cls, v):
        return parse_date(v)

    @field_validator("buyer_national_id", "buyer_phone", "contract_no", "unit_no", mode="before")
    @classmethod
    def _ids(cls, v):
        if v is None:
            return None
        return to_western_digits(str(v)).strip()

    @field_validator("unit_type", mode="before")
    @classmethod
    def _unit_type(cls, v):
        if v is None:
            return None
        key = str(v).strip().lower()
        return _UNIT_TYPE_ALIASES.get(key, key)

    @field_validator("payment_method", mode="before")
    @classmethod
    def _pm(cls, v):
        if v is None:
            return None
        key = str(v).strip().lower()
        return _PAYMENT_ALIASES.get(key, key)

    @field_validator("bank_name", mode="before")
    @classmethod
    def _bank(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return None if s.lower() in ("", "null", "none", "n/a", "لا ينطبق", "-") else s

    @classmethod
    def json_schema_for_prompt(cls) -> dict:
        return {name: f.description for name, f in cls.model_fields.items()}


def validate_business_rules(r: ContractRecord) -> list[str]:
    """Return a list of human-readable issues (empty list == consistent record)."""
    issues: list[str] = []
    if r.buyer_national_id and not re.fullmatch(r"[12]\d{9}", r.buyer_national_id):
        issues.append("national ID should be 10 digits starting with 1 (citizen) or 2 (resident)")
    if r.buyer_phone and not re.fullmatch(r"(05\d{8}|\+9665\d{8})", r.buyer_phone.replace(" ", "")):
        issues.append("phone should be a Saudi mobile (05XXXXXXXX)")
    if r.total_price_sar is not None and r.total_price_sar <= 0:
        issues.append("total price must be positive")
    if r.area_sqm is not None and not (40 <= r.area_sqm <= 2000):
        issues.append("area outside plausible range (40–2000 sqm)")
    if r.payment_method == "cash":
        if r.down_payment_sar is not None and r.total_price_sar is not None and abs(r.down_payment_sar - r.total_price_sar) > 1:
            issues.append("cash deal but down payment != total price")
    elif r.payment_method in ("installments", "bank_financing"):
        if None not in (r.total_price_sar, r.down_payment_sar, r.installments_count, r.installment_amount_sar):
            expected = r.down_payment_sar + r.installments_count * r.installment_amount_sar
            if abs(expected - r.total_price_sar) > max(1.0, 0.001 * r.total_price_sar):
                issues.append(f"payments do not sum to price (expected {r.total_price_sar:,.0f}, got {expected:,.0f})")
        if r.payment_method == "bank_financing" and not r.bank_name:
            issues.append("bank financing without a bank name")
    for fld in ("contract_date", "delivery_date"):
        val = getattr(r, fld)
        if val and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
            issues.append(f"{fld} is not a valid ISO date: {val}")
    if r.contract_date and r.delivery_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.contract_date) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.delivery_date):
        if r.delivery_date < r.contract_date:
            issues.append("delivery date precedes contract date")
    required = ["contract_no", "buyer_name", "unit_no", "total_price_sar"]
    missing = [f for f in required if getattr(r, f) in (None, "")]
    if missing:
        issues.append("missing required fields: " + ", ".join(missing))
    return issues
