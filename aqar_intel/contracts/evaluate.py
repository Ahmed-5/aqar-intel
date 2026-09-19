"""Field-level evaluation of contract extraction against ground truth.

Reports exact-match accuracy per field (after normalisation), overall
accuracy, the share of contracts extracted perfectly, and a breakdown by
template/language so weak spots (e.g. Arabic-Indic digits) are visible
rather than averaged away.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..arabic import normalize_for_match
from ..config import CONTRACTS_DIR
from .extractor import Extraction, LLMExtractor, RuleBasedExtractor, extract_file
from .schema import ContractRecord

FIELDS = [f for f in ContractRecord.model_fields]
NUMERIC = {"area_sqm", "total_price_sar", "down_payment_sar", "installments_count", "installment_amount_sar"}


def field_match(name: str, pred, truth) -> bool:
    if name in NUMERIC:
        if pred is None or truth is None:
            return pred is None and truth is None
        return abs(float(pred) - float(truth)) <= max(0.5, 0.001 * abs(float(truth)))
    return normalize_for_match(pred) == normalize_for_match(truth)


@dataclass
class EvalReport:
    extractor: str
    n_contracts: int
    per_field: dict[str, float]
    overall: float
    perfect_contracts: float
    by_template: dict[str, float]
    flagged_for_review: int
    seconds: float
    planted_issues_total: int = 0
    planted_issues_caught: int = 0
    mismatches: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"### Extractor: `{self.extractor}` — {self.n_contracts} contracts, {self.seconds:.1f}s",
            "",
            f"- Overall field accuracy: **{self.overall:.1%}**",
            f"- Contracts with all fields correct: **{self.perfect_contracts:.1%}**",
            f"- Flagged for human review (validation issues/errors): {self.flagged_for_review}",
            f"- Deliberately inconsistent contracts caught by validation: **{self.planted_issues_caught}/{self.planted_issues_total}**",
            "",
            "| Field | Accuracy |", "|---|---|",
        ]
        lines += [f"| {k} | {v:.1%} |" for k, v in self.per_field.items()]
        lines += ["", "| Template | Accuracy |", "|---|---|"]
        lines += [f"| {k} | {v:.1%} |" for k, v in sorted(self.by_template.items())]
        return "\n".join(lines)


def evaluate(extractor: LLMExtractor | RuleBasedExtractor, contracts_dir: Path = CONTRACTS_DIR,
             limit: int | None = None, keep_mismatches: int = 30) -> EvalReport:
    truth = json.loads((contracts_dir / "ground_truth.json").read_text(encoding="utf-8"))
    names = sorted(truth)[:limit] if limit else sorted(truth)
    correct = defaultdict(int)
    tmpl_hits, tmpl_total = defaultdict(int), defaultdict(int)
    perfect, flagged, mismatches = 0, 0, []
    planted_total, planted_caught = 0, 0
    t0 = time.perf_counter()
    for name in names:
        ex: Extraction = extract_file(contracts_dir / name, extractor)
        gt = truth[name]
        pred = ex.record.model_dump()
        all_ok = True
        for f in FIELDS:
            ok = field_match(f, pred.get(f), gt.get(f))
            correct[f] += ok
            tmpl_hits[gt["template"]] += ok
            tmpl_total[gt["template"]] += 1
            if not ok:
                all_ok = False
                if len(mismatches) < keep_mismatches:
                    mismatches.append({"contract": name, "field": f, "pred": pred.get(f), "truth": gt.get(f)})
        perfect += all_ok
        flagged += ex.needs_review
        if gt.get("planted_issue"):
            planted_total += 1
            planted_caught += bool(ex.issues) and ex.error is None
    n = len(names)
    per_field = {f: correct[f] / n for f in FIELDS}
    return EvalReport(
        extractor=extractor.name,
        n_contracts=n,
        per_field=per_field,
        overall=sum(correct.values()) / (n * len(FIELDS)),
        perfect_contracts=perfect / n,
        by_template={t: tmpl_hits[t] / tmpl_total[t] for t in tmpl_total},
        flagged_for_review=flagged,
        seconds=time.perf_counter() - t0,
        planted_issues_total=planted_total,
        planted_issues_caught=planted_caught,
        mismatches=mismatches,
    )
