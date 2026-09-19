"""Command-line entry point for the three modules.

    python scripts/run_demo.py ask "ما هي رسوم الحجز؟"
    python scripts/run_demo.py ask "How many 4-bedroom villas are available in Narjis Oasis?"
    python scripts/run_demo.py extract data/contracts/contract_001_A.txt
    python scripts/run_demo.py evaluate [--extractor llm|rules|both] [--limit 10]
    python scripts/run_demo.py report
    python scripts/run_demo.py all        # runs a short version of everything
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from aqar_intel.config import CONTRACTS_DIR, REPORTS_DIR, get_settings  # noqa: E402
from aqar_intel.contracts.evaluate import evaluate  # noqa: E402
from aqar_intel.contracts.extractor import LLMExtractor, RuleBasedExtractor, extract_file  # noqa: E402
from aqar_intel.llm import get_llm  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SAMPLE_QUESTIONS = [
    "ما هي رسوم الحجز وهل تُسترد؟",
    "كم عدد الفلل المتاحة في واحة النرجس وما أقل سعر؟",
    "What is the structural warranty period?",
    "Which project has the most available apartments under SAR 1.5M?",
]


def cmd_ask(args) -> None:
    from aqar_intel.rag.assistant import Assistant, format_answer

    assistant = Assistant()
    questions = [args.question] if args.question else SAMPLE_QUESTIONS
    for q in questions:
        a = assistant.ask(q)
        print(f"\n❓ {q}\n   route={a.route} lang={a.language}")
        print(format_answer(a))


def cmd_extract(args) -> None:
    extractor = RuleBasedExtractor() if args.extractor == "rules" else LLMExtractor()
    ex = extract_file(Path(args.path), extractor)
    print(json.dumps(ex.to_dict(), ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    extractors = []
    if args.extractor in ("rules", "both"):
        extractors.append(RuleBasedExtractor())
    if args.extractor in ("llm", "both"):
        extractors.append(LLMExtractor())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ex in extractors:
        rep = evaluate(ex, CONTRACTS_DIR, limit=args.limit)
        print(rep.to_markdown())
        if rep.mismatches:
            print("\nFirst mismatches:")
            for m in rep.mismatches[:8]:
                print(f"  {m['contract']} {m['field']}: pred={m['pred']!r} truth={m['truth']!r}")
        out.append(rep.to_markdown())
    (REPORTS_DIR / "extraction_eval.md").write_text("\n\n".join(out), encoding="utf-8")
    print(f"\nSaved -> {REPORTS_DIR / 'extraction_eval.md'}")


def cmd_report(args) -> None:
    from aqar_intel.analytics.report import build_report

    out = build_report(REPORTS_DIR)
    print("Reports:", json.dumps(out["reports"], indent=2))
    print("Charts:", out["charts"])
    print("\nPricing model:", json.dumps(out["pricing"]["model"], indent=2))


def cmd_all(args) -> None:
    settings = get_settings()
    print(f"LLM mode: {'OpenRouter ' + settings.chat_model if settings.use_openrouter else 'mock (offline)'}")
    cmd_ask(argparse.Namespace(question=None))
    cmd_evaluate(argparse.Namespace(extractor="both" if settings.use_openrouter else "rules", limit=10))
    cmd_report(argparse.Namespace())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question", nargs="?"); a.set_defaults(fn=cmd_ask)
    e = sub.add_parser("extract"); e.add_argument("path"); e.add_argument("--extractor", default="llm", choices=["llm", "rules"]); e.set_defaults(fn=cmd_extract)
    v = sub.add_parser("evaluate"); v.add_argument("--extractor", default="both", choices=["llm", "rules", "both"]); v.add_argument("--limit", type=int); v.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("report"); r.set_defaults(fn=cmd_report)
    x = sub.add_parser("all"); x.set_defaults(fn=cmd_all)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
