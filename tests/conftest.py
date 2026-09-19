import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["AQAR_LLM_MODE"] = "mock"  # never hit the network in tests


@pytest.fixture(scope="session", autouse=True)
def generated_data():
    """Make sure simulated data exists (cheap, deterministic)."""
    from aqar_intel.config import CONTRACTS_DIR, DB_PATH, DOCS_DIR

    if not (DB_PATH.exists() and DOCS_DIR.exists() and (CONTRACTS_DIR / "ground_truth.json").exists()):
        from aqar_intel.data_gen.contracts import generate_contracts
        from aqar_intel.data_gen.documents import generate_documents
        from aqar_intel.data_gen.inventory import generate_inventory

        generate_inventory()
        generate_documents()
        generate_contracts()
