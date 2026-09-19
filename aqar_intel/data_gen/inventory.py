"""Generate a simulated unit inventory + sales history into SQLite.

The generator encodes a few realistic dynamics so the analytics module has
something worth modelling:

* list prices scale with area, unit type, view and floor, plus a ~6%/yr trend;
* absorption follows a launch spike, then decays, with month-of-year effects;
* sale prices carry a small negotiated discount off list price.

All values are fictional.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, DB_PATH
from .world import BANKS_AR, BANKS_EN, PROJECTS, STATUSES, UNIT_TYPES, VIEWS

DATA_START = (2024, 1)  # month index 0
DATA_END = (2026, 8)  # last month with sales data (inclusive)

SCHEMA = """
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS units;
DROP TABLE IF EXISTS projects;

CREATE TABLE projects (
    project_id      TEXT PRIMARY KEY,
    name_ar         TEXT NOT NULL,
    name_en         TEXT NOT NULL,
    city_ar         TEXT NOT NULL,
    city_en         TEXT NOT NULL,
    district_ar     TEXT NOT NULL,
    district_en     TEXT NOT NULL,
    launch_month    TEXT NOT NULL,   -- YYYY-MM
    delivery_month  TEXT NOT NULL,   -- YYYY-MM
    total_units     INTEGER NOT NULL
);

CREATE TABLE units (
    unit_id         TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(project_id),
    unit_no         TEXT NOT NULL,
    unit_type       TEXT NOT NULL,   -- apartment | townhouse | duplex | villa
    unit_type_ar    TEXT NOT NULL,
    bedrooms        INTEGER NOT NULL,
    bathrooms       INTEGER NOT NULL,
    area_sqm        REAL NOT NULL,
    floor           INTEGER,         -- NULL for villas/townhouses
    view            TEXT NOT NULL,   -- garden | street | city | sea | park
    view_ar         TEXT NOT NULL,
    parking_spaces  INTEGER NOT NULL,
    list_price_sar  REAL NOT NULL,
    status          TEXT NOT NULL,   -- available | reserved | sold
    status_ar       TEXT NOT NULL,
    delivery_month  TEXT NOT NULL
);

CREATE TABLE sales (
    sale_id         INTEGER PRIMARY KEY,
    unit_id         TEXT NOT NULL REFERENCES units(unit_id),
    project_id      TEXT NOT NULL REFERENCES projects(project_id),
    sale_date       TEXT NOT NULL,   -- YYYY-MM-DD
    sale_price_sar  REAL NOT NULL,
    discount_pct    REAL NOT NULL,
    payment_method  TEXT NOT NULL,   -- bank_financing | cash
    bank_name       TEXT,
    buyer_city      TEXT NOT NULL
);

CREATE INDEX idx_units_project ON units(project_id);
CREATE INDEX idx_units_status ON units(status);
CREATE INDEX idx_sales_project_date ON sales(project_id, sale_date);
"""


def month_index(year: int, month: int) -> int:
    return (year - DATA_START[0]) * 12 + (month - DATA_START[1])


def index_to_ym(idx: int) -> tuple[int, int]:
    y = DATA_START[0] + (DATA_START[1] - 1 + idx) // 12
    m = (DATA_START[1] - 1 + idx) % 12 + 1
    return y, m


def _ym_str(idx: int) -> str:
    y, m = index_to_ym(idx)
    return f"{y:04d}-{m:02d}"


# Month-of-year absorption multipliers (rough seasonality: strong Q1/Q4, softer summer)
SEASONALITY = {1: 1.10, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.80, 7: 0.75, 8: 0.85,
               9: 1.05, 10: 1.10, 11: 1.15, 12: 1.30}


def list_price(rng: np.random.Generator, base_sqm: float, area: float, unit_type: str,
               view: str, floor: int | None, month_idx: int) -> float:
    trend = (1 + 0.005) ** month_idx  # ~6% p.a.
    floor_mult = 1.0 + 0.004 * floor if floor is not None else 1.0
    noise = float(np.exp(rng.normal(0, 0.04)))
    price = base_sqm * area * UNIT_TYPES[unit_type]["mult"] * VIEWS[view]["mult"] * floor_mult * trend * noise
    return float(round(price / 1000) * 1000)


def generate_inventory(db_path: Path = DB_PATH, seed: int = 42, export_csv: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    end_idx = month_index(*DATA_END)
    units_rows, sales_rows = [], []
    sale_id = 1

    for p in PROJECTS:
        launch_idx = month_index(int(p.launch[:4]), int(p.launch[5:]))
        con.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.name_ar, p.name_en, p.city_ar, p.city_en, p.district_ar, p.district_en,
             p.launch, p.delivery, p.n_units),
        )

        # --- units ------------------------------------------------------- #
        views = [v for v in VIEWS if v != "sea"] if p.city_en not in ("Jeddah", "Al Khobar") else list(VIEWS)
        view_weights = np.array([VIEWS[v]["mult"] for v in views])
        view_weights = (1 / view_weights) / (1 / view_weights).sum()  # premium views rarer

        project_units = []
        for i in range(p.n_units):
            utype = p.unit_types[rng.integers(len(p.unit_types))]
            spec = UNIT_TYPES[utype]
            area = float(round(rng.uniform(*spec["area"]) / 5) * 5)
            lo, hi = spec["beds"]
            beds = int(np.clip(round(lo + (area - spec["area"][0]) / (spec["area"][1] - spec["area"][0]) * (hi - lo) + rng.normal(0, 0.5)), lo, hi))
            baths = beds if utype in ("villa", "duplex") else max(1, beds - 1)
            floor = int(rng.integers(1, 25)) if utype in ("apartment", "duplex") else None
            view = str(rng.choice(views, p=view_weights))
            parking = 2 if utype in ("villa", "duplex") else (1 if area < 150 else 2)
            price = list_price(rng, p.base_price_sqm, area, utype, view, floor, launch_idx)
            unit_no = f"{p.id}-{(floor or 0):02d}{i + 1:03d}" if floor else f"{p.id}-{i + 1:03d}"
            uid = f"{p.id}{i + 1:04d}"
            project_units.append(dict(
                unit_id=uid, project_id=p.id, unit_no=unit_no, unit_type=utype,
                unit_type_ar=spec["ar"], bedrooms=beds, bathrooms=baths, area_sqm=area,
                floor=floor, view=view, view_ar=VIEWS[view]["ar"], parking_spaces=parking,
                list_price_sar=price, status="available", status_ar=STATUSES["available"]["ar"],
                delivery_month=p.delivery,
            ))

        # --- sales / absorption ------------------------------------------- #
        n_months = end_idx - launch_idx + 1
        target_sold_frac = float(rng.uniform(0.55, 0.85))
        target_sold = int(p.n_units * target_sold_frac)
        # launch spike, slow decay towards a steady-state floor
        raw = np.array([0.35 + math.exp(-0.03 * t) * (1.0 + 1.2 * math.exp(-0.8 * t)) for t in range(n_months)])
        raw *= np.array([SEASONALITY[index_to_ym(launch_idx + t)[1]] for t in range(n_months)])
        raw *= np.exp(rng.normal(0, 0.15, size=n_months))
        monthly = np.floor(raw / raw.sum() * target_sold).astype(int)

        order = rng.permutation(len(project_units))
        cursor = 0
        for t, n_sold in enumerate(monthly):
            m_idx = launch_idx + t
            y, m = index_to_ym(m_idx)
            for _ in range(int(n_sold)):
                if cursor >= len(order):
                    break
                u = project_units[order[cursor]]
                cursor += 1
                day = int(rng.integers(1, 28))
                # sale price = list price at time of sale, less negotiated discount
                trend_adj = (1 + 0.005) ** (m_idx - launch_idx)
                discount = float(np.clip(rng.normal(0.02, 0.015), 0.0, 0.08))
                sale_price = round(u["list_price_sar"] * trend_adj * (1 - discount) / 1000) * 1000
                financed = rng.random() < 0.72
                bank_i = int(rng.integers(len(BANKS_AR)))
                u["status"], u["status_ar"] = "sold", STATUSES["sold"]["ar"]
                sales_rows.append((
                    sale_id, u["unit_id"], p.id, date(y, m, day).isoformat(), float(sale_price),
                    round(discount * 100, 2), "bank_financing" if financed else "cash",
                    BANKS_EN[bank_i] if financed else None,
                    str(rng.choice([p.city_en, p.city_en, p.city_en, "Riyadh", "Jeddah", "Dammam"])),
                ))
                sale_id += 1
        # a few reserved
        for j in range(cursor, min(cursor + int(0.05 * p.n_units), len(order))):
            u = project_units[order[j]]
            u["status"], u["status_ar"] = "reserved", STATUSES["reserved"]["ar"]

        units_rows.extend(project_units)

    con.executemany(
        "INSERT INTO units VALUES (:unit_id,:project_id,:unit_no,:unit_type,:unit_type_ar,:bedrooms,"
        ":bathrooms,:area_sqm,:floor,:view,:view_ar,:parking_spaces,:list_price_sar,:status,:status_ar,:delivery_month)",
        units_rows,
    )
    con.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?)", sales_rows)
    con.commit()

    summary = {
        "projects": len(PROJECTS),
        "units": len(units_rows),
        "sales": len(sales_rows),
        "db_path": str(db_path),
    }

    if export_csv:
        out = DATA_DIR / "sales.csv"
        cur = con.execute(
            """SELECT s.sale_id, s.sale_date, s.sale_price_sar, s.discount_pct, s.payment_method, s.bank_name,
                      s.buyer_city, u.unit_id, u.unit_no, u.unit_type, u.bedrooms, u.bathrooms, u.area_sqm,
                      u.floor, u.view, u.parking_spaces, u.list_price_sar, p.project_id, p.name_en AS project_name,
                      p.city_en AS city
               FROM sales s JOIN units u ON s.unit_id = u.unit_id JOIN projects p ON p.project_id = s.project_id
               ORDER BY s.sale_date"""
        )
        cols = [d[0] for d in cur.description]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(cur.fetchall())
        summary["sales_csv"] = str(out)

    con.close()
    return summary


if __name__ == "__main__":
    print(generate_inventory())
