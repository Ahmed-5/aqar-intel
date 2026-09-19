import json

import pytest

from aqar_intel.config import CONTRACTS_DIR
from aqar_intel.contracts.evaluate import evaluate
from aqar_intel.contracts.extractor import LLMExtractor, RuleBasedExtractor, extract_file
from aqar_intel.contracts.schema import ContractRecord, parse_date, parse_number, validate_business_rules
from aqar_intel.llm import FakeLLM


def test_parsers():
    assert parse_date("١٤ مارس ٢٠٢٦") == "2026-03-14"
    assert parse_date("14/03/2026م") == "2026-03-14"
    assert parse_date("3 October 2025") == "2025-10-03"
    assert parse_number("١,١٥١,٠٠٠ ريال") == 1151000.0


def test_schema_normalises_aliases():
    r = ContractRecord(unit_type="شقة", payment_method="تمويل بنكي", bank_name="لا ينطبق", total_price_sar="1,000,000")
    assert r.unit_type == "apartment" and r.payment_method == "bank_financing" and r.bank_name is None
    assert r.total_price_sar == 1_000_000


def test_business_rules_catch_inconsistencies():
    r = ContractRecord(contract_no="X", buyer_name="a", unit_no="U1", total_price_sar=1_000_000, payment_method="installments",
                       down_payment_sar=100_000, installments_count=4, installment_amount_sar=200_000,
                       buyer_national_id="123", contract_date="2026-05-01", delivery_date="2026-01-01")
    issues = validate_business_rules(r)
    assert any("do not sum" in i for i in issues)
    assert any("national ID" in i for i in issues)
    assert any("precedes" in i for i in issues)
    assert validate_business_rules(ContractRecord(contract_no="X", buyer_name="a", unit_no="U1", total_price_sar=5.0,
                                                  payment_method="cash", down_payment_sar=5.0)) == []


def test_rule_based_baseline_on_known_templates():
    rep = evaluate(RuleBasedExtractor(), CONTRACTS_DIR)
    for t in "ABCD":
        assert rep.by_template[t] == 1.0
    assert rep.by_template["E"] < 0.2  # held-out layout: baseline should fail, LLM should not
    assert rep.planted_issues_caught == rep.planted_issues_total == 3


def test_llm_extractor_parses_and_validates():
    truth = json.loads((CONTRACTS_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    name = next(n for n, v in truth.items() if v["template"] == "B")
    payload = {k: v for k, v in truth[name].items() if k in ContractRecord.model_fields}
    payload["total_price_sar"] = "١,٢٣٤,٠٠٠"  # model returned Arabic-Indic digits: schema must normalise
    fake = FakeLLM(["```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"])
    ex = extract_file(CONTRACTS_DIR / name, LLMExtractor(fake))
    assert ex.record.total_price_sar == 1_234_000
    assert ex.extractor == "llm" and ex.error is None


def test_llm_extractor_survives_garbage():
    ex = LLMExtractor(FakeLLM(["I cannot do that"])).extract("some contract text")
    assert ex.error and ex.needs_review
