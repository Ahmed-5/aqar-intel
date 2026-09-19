"""Regenerate every simulated dataset (inventory DB, documents, contracts).

    python scripts/generate_data.py [--seed 42] [--contracts 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqar_intel.data_gen.contracts import generate_contracts  # noqa: E402
from aqar_intel.data_gen.documents import generate_documents  # noqa: E402
from aqar_intel.data_gen.inventory import generate_inventory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contracts", type=int, default=40)
    args = ap.parse_args()

    out = {
        "inventory": generate_inventory(seed=args.seed),
        "documents": generate_documents(),
        "contracts": generate_contracts(n=args.contracts, seed=args.seed),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nNext: python scripts/build_index.py")


if __name__ == "__main__":
    main()
