"""Sale-price model for residential units.

Design choices worth noting for a production setting:

* **Time-based split.** Sales are ordered by date; the last N months are the
  test set. A random split would leak future price levels into training and
  overstate accuracy (classic problem with trending markets).
* **Baselines first.** A price-per-sqm-by-project baseline is reported next
  to the model so the lift is visible. If the model does not beat the
  baseline, ship the baseline.
* **Log target.** Prices span 0.5–5M SAR; modelling ``log(price)`` keeps
  errors proportional and MAPE meaningful.
* **Permutation importance on the test set** rather than impurity
  importance, which is biased toward high-cardinality features.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..config import DB_PATH, MODELS_DIR

log = logging.getLogger(__name__)

CATEGORICAL = ["project_id", "city", "unit_type", "view", "payment_method"]
NUMERIC = ["bedrooms", "bathrooms", "area_sqm", "floor", "parking_spaces", "month_index"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "sale_price_sar"


def load_sales(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT s.sale_id, s.sale_date, s.sale_price_sar, s.discount_pct, s.payment_method,
                  u.unit_type, u.bedrooms, u.bathrooms, u.area_sqm, u.floor, u.view, u.parking_spaces, u.list_price_sar,
                  p.project_id, p.name_en AS project_name, p.city_en AS city
           FROM sales s JOIN units u ON u.unit_id = s.unit_id JOIN projects p ON p.project_id = s.project_id
           ORDER BY s.sale_date""",
        con,
    )
    con.close()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["month"] = df["sale_date"].dt.to_period("M")
    df["month_index"] = (df["sale_date"].dt.year - 2024) * 12 + df["sale_date"].dt.month - 1
    df["floor"] = df["floor"].fillna(0)
    return df


@dataclass
class PricingReport:
    n_train: int
    n_test: int
    test_start: str
    metrics_model: dict
    metrics_baseline: dict
    importance: dict[str, float]
    model_path: str | None = None
    by_project_mape: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train, "n_test": self.n_test, "test_start": self.test_start,
            "model": self.metrics_model, "baseline_ppsqm": self.metrics_baseline,
            "permutation_importance": self.importance, "by_project_mape": self.by_project_mape,
            "model_path": self.model_path,
        }

    def to_markdown(self) -> str:
        m, b = self.metrics_model, self.metrics_baseline
        lines = [
            f"### Pricing model — train {self.n_train} / test {self.n_test} sales (test from {self.test_start})",
            "",
            "| Metric | Gradient boosting | Baseline (price/sqm by project & type) |",
            "|---|---|---|",
            f"| MAE (SAR) | {m['mae']:,.0f} | {b['mae']:,.0f} |",
            f"| MAPE | {m['mape']:.2%} | {b['mape']:.2%} |",
            f"| R² | {m['r2']:.3f} | {b['r2']:.3f} |",
            "",
            "Permutation importance (test set, Δ MAPE):",
            "",
        ] + [f"- {k}: {v:.4f}" for k, v in self.importance.items()]
        return "\n".join(lines)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_pipeline(seed: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
         ("num", "passthrough", NUMERIC)]
    )
    n_cat = len(CATEGORICAL)
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=0.1,
        categorical_features=list(range(n_cat)), random_state=seed,
    )
    return Pipeline([("pre", pre), ("gbr", model)])


def train_pricing_model(df: pd.DataFrame | None = None, *, test_months: int = 6, seed: int = 42,
                        save_to: Path | None = MODELS_DIR / "pricing_model.joblib") -> tuple[Pipeline, PricingReport]:
    df = load_sales() if df is None else df.copy()
    cutoff = df["month"].max() - (test_months - 1)
    train, test = df[df["month"] < cutoff], df[df["month"] >= cutoff]
    if len(test) < 20:
        raise ValueError("test set too small; reduce test_months")

    pipe = build_pipeline(seed)
    pipe.fit(train[FEATURES], np.log(train[TARGET]))
    pred = np.exp(pipe.predict(test[FEATURES]))

    # Baseline: median price per sqm by (project, unit_type) from training data, × area
    ppsqm = (train[TARGET] / train["area_sqm"]).groupby([train["project_id"], train["unit_type"]]).median()
    global_ppsqm = float((train[TARGET] / train["area_sqm"]).median())
    base = test.apply(lambda r: ppsqm.get((r["project_id"], r["unit_type"]), global_ppsqm) * r["area_sqm"], axis=1).to_numpy()

    # Permutation importance in log space (MAPE-like)
    pi = permutation_importance(pipe, test[FEATURES], np.log(test[TARGET]), scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=seed)
    importance = {f: float(v) for f, v in sorted(zip(FEATURES, pi.importances_mean), key=lambda kv: -kv[1])}

    by_proj = {}
    for pid, grp in test.assign(pred=pred).groupby("project_name"):
        by_proj[pid] = float(mean_absolute_percentage_error(grp[TARGET], grp["pred"]))

    report = PricingReport(
        n_train=len(train), n_test=len(test), test_start=str(cutoff),
        metrics_model=_metrics(test[TARGET].to_numpy(), pred),
        metrics_baseline=_metrics(test[TARGET].to_numpy(), base),
        importance=importance, by_project_mape=by_proj,
    )
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        # Refit on all data for deployment; keep the honest test metrics from above.
        final = build_pipeline(seed).fit(df[FEATURES], np.log(df[TARGET]))
        joblib.dump(final, save_to)
        (save_to.with_suffix(".metrics.json")).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        report.model_path = str(save_to)
        return final, report
    return pipe, report


def predict_price(model: Pipeline, unit: dict) -> float:
    """Predict a sale price for a single unit spec (dict with FEATURES keys)."""
    row = {f: unit.get(f) for f in FEATURES}
    row["floor"] = row.get("floor") or 0
    row["payment_method"] = row.get("payment_method") or "bank_financing"
    if row.get("month_index") is None:
        today = pd.Timestamp.today()
        row["month_index"] = (today.year - 2024) * 12 + today.month - 1
    X = pd.DataFrame([row])
    return float(np.exp(model.predict(X))[0])


def load_pricing_model(path: Path = MODELS_DIR / "pricing_model.joblib") -> Pipeline:
    return joblib.load(path)
