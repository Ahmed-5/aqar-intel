"""Absorption forecasting: how many units each project will sell per month.

Deliberately simple and honest. With 12–36 months of history per project a
heavy model is not justified, so we compare three transparent forecasters
on a rolling backtest and pick the best per project:

* ``naive_last``   – repeat last month;
* ``mean_3``       – mean of the last three months;
* ``damped_trend`` – Holt's linear trend with damping (implemented inline;
  no statsmodels dependency), floored at zero.

Also derives "months of inventory" (remaining units ÷ forecast monthly
absorption), the number sales managers actually plan around.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DB_PATH


def monthly_absorption(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return a wide table: index = month (Period), columns = project_id, values = units sold."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT project_id, sale_date FROM sales", con)
    launches = pd.read_sql_query("SELECT project_id, launch_month FROM projects", con).set_index("project_id")["launch_month"]
    con.close()
    df["month"] = pd.to_datetime(df["sale_date"]).dt.to_period("M")
    wide = df.groupby(["month", "project_id"]).size().unstack(fill_value=0)
    full_idx = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    wide = wide.reindex(full_idx, fill_value=0)
    # months before a project's launch are not "zero sales", they are not applicable -> NaN
    for pid, launch in launches.items():
        if pid in wide.columns:
            wide.loc[wide.index < pd.Period(launch, freq="M"), pid] = np.nan
    return wide


# --------------------------------------------------------------------------- #
# Forecasters (each: history -> horizon-length forecast)
# --------------------------------------------------------------------------- #
def naive_last(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(max(y[-1], 0.0), h)


def mean_3(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(max(float(np.mean(y[-3:])), 0.0), h)


def damped_trend(y: np.ndarray, h: int, alpha: float = 0.5, beta: float = 0.2, phi: float = 0.85) -> np.ndarray:
    level, trend = float(y[0]), float(y[1] - y[0]) if len(y) > 1 else 0.0
    for t in range(1, len(y)):
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (prev_level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
    steps = np.array([sum(phi ** i for i in range(1, k + 1)) for k in range(1, h + 1)])
    return np.maximum(level + steps * trend, 0.0)


FORECASTERS = {"naive_last": naive_last, "mean_3": mean_3, "damped_trend": damped_trend}


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@dataclass
class ProjectForecast:
    project_id: str
    history: pd.Series
    method: str
    forecast: pd.Series
    backtest_mae: dict[str, float]
    remaining_units: int
    months_of_inventory: float | None

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "method": self.method,
            "backtest_mae": self.backtest_mae,
            "last_3_months": [int(v) for v in self.history.tail(3).tolist()],
            "forecast": {str(k): round(float(v), 1) for k, v in self.forecast.items()},
            "remaining_units": self.remaining_units,
            "months_of_inventory": None if self.months_of_inventory is None else round(self.months_of_inventory, 1),
        }


@dataclass
class AbsorptionReport:
    horizon: int
    backtest_window: int
    projects: list[ProjectForecast] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"horizon_months": self.horizon, "backtest_window_months": self.backtest_window,
                "projects": [p.to_dict() for p in self.projects]}

    def to_markdown(self) -> str:
        lines = [f"### Absorption forecast — next {self.horizon} months (backtest on last {self.backtest_window} months)", "",
                 "| Project | Best method | Backtest MAE (best / naive) | Last 3 months | Forecast next 3 | Remaining | Months of inventory |",
                 "|---|---|---|---|---|---|---|"]
        for p in self.projects:
            f3 = ", ".join(f"{v:.0f}" for v in p.forecast.head(3))
            l3 = ", ".join(f"{int(v)}" for v in p.history.tail(3))
            moi = "—" if p.months_of_inventory is None else f"{p.months_of_inventory:.1f}"
            lines.append(f"| {p.project_id} | {p.method} | {p.backtest_mae[p.method]:.2f} / {p.backtest_mae['naive_last']:.2f} | {l3} | {f3} | {p.remaining_units} | {moi} |")
        return "\n".join(lines)


def forecast_absorption(db_path: Path = DB_PATH, horizon: int = 6, backtest_window: int = 6) -> AbsorptionReport:
    wide = monthly_absorption(db_path)
    con = sqlite3.connect(db_path)
    remaining = dict(con.execute("SELECT project_id, SUM(status='available') FROM units GROUP BY project_id").fetchall())
    con.close()

    report = AbsorptionReport(horizon=horizon, backtest_window=backtest_window)
    for pid in wide.columns:
        hist = wide[pid].dropna()
        y = hist.to_numpy(dtype=float)
        if len(y) < backtest_window + 3:
            continue
        # rolling-origin backtest: forecast 1 step ahead over the last `backtest_window` months
        errors = {name: [] for name in FORECASTERS}
        for cut in range(len(y) - backtest_window, len(y)):
            for name, fn in FORECASTERS.items():
                errors[name].append(abs(fn(y[:cut], 1)[0] - y[cut]))
        mae = {name: float(np.mean(v)) for name, v in errors.items()}
        best = min(mae, key=mae.get)
        fc = FORECASTERS[best](y, horizon)
        idx = pd.period_range(hist.index[-1] + 1, periods=horizon, freq="M")
        fc_series = pd.Series(fc, index=idx)
        rem = int(remaining.get(pid, 0))
        avg_fc = float(np.mean(fc))
        moi = (rem / avg_fc) if avg_fc > 0.05 else None
        report.projects.append(ProjectForecast(pid, hist, best, fc_series, mae, rem, moi))
    return report
