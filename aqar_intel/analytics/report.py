"""Automated sales-performance report: charts + LLM-written executive summary.

The LLM only receives *computed* figures (JSON) and is instructed to comment
on them, not to invent numbers. Charts are produced deterministically with
matplotlib so the report is reproducible.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import DB_PATH, REPORTS_DIR  # noqa: E402
from ..llm import LLM, get_llm  # noqa: E402
from .absorption import AbsorptionReport, forecast_absorption, monthly_absorption  # noqa: E402
from .pricing import PricingReport, load_sales, train_pricing_model  # noqa: E402


def kpi_snapshot(db_path: Path = DB_PATH) -> dict:
    con = sqlite3.connect(db_path)
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    total, available, reserved, sold = q("SELECT COUNT(*), SUM(status='available'), SUM(status='reserved'), SUM(status='sold') FROM units")[0]
    revenue = q("SELECT ROUND(SUM(sale_price_sar)) FROM sales")[0][0]
    last3 = q("SELECT COUNT(*), ROUND(SUM(sale_price_sar)) FROM sales WHERE sale_date >= date((SELECT MAX(sale_date) FROM sales), '-3 months')")[0]
    prev3 = q("SELECT COUNT(*) FROM sales WHERE sale_date < date((SELECT MAX(sale_date) FROM sales), '-3 months') AND sale_date >= date((SELECT MAX(sale_date) FROM sales), '-6 months')")[0][0]
    by_project = q("""SELECT p.name_en, p.total_units, SUM(u.status='sold') AS sold, SUM(u.status='available') AS available,
                             ROUND(AVG(CASE WHEN u.status='sold' THEN s.sale_price_sar END)) AS avg_sale_price,
                             ROUND(AVG(CASE WHEN u.status='sold' THEN s.sale_price_sar/u.area_sqm END)) AS avg_price_sqm
                      FROM projects p JOIN units u ON u.project_id=p.project_id
                      LEFT JOIN sales s ON s.unit_id=u.unit_id
                      GROUP BY p.name_en ORDER BY sold DESC""")
    by_type = q("SELECT u.unit_type, COUNT(*), ROUND(AVG(s.sale_price_sar)) FROM sales s JOIN units u ON u.unit_id=s.unit_id GROUP BY u.unit_type ORDER BY 2 DESC")
    financing = q("SELECT payment_method, COUNT(*) FROM sales GROUP BY payment_method")
    avg_discount = q("SELECT ROUND(AVG(discount_pct),2) FROM sales")[0][0]
    last_month = q("SELECT MAX(sale_date) FROM sales")[0][0]
    con.close()
    return {
        "as_of": last_month,
        "units": {"total": total, "available": available, "reserved": reserved, "sold": sold, "sold_pct": round(100 * sold / total, 1)},
        "revenue_sar_total": revenue,
        "last_3_months": {"units_sold": last3[0], "revenue_sar": last3[1], "units_sold_previous_3_months": prev3},
        "avg_discount_pct": avg_discount,
        "by_project": [dict(zip(["project", "total_units", "sold", "available", "avg_sale_price_sar", "avg_price_per_sqm"], r)) for r in by_project],
        "by_unit_type": [dict(zip(["unit_type", "units_sold", "avg_sale_price_sar"], r)) for r in by_type],
        "financing_mix": {k: v for k, v in financing},
    }


def make_charts(out_dir: Path = REPORTS_DIR, db_path: Path = DB_PATH, absorption: AbsorptionReport | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    wide = monthly_absorption(db_path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = wide.index.to_timestamp()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fc_by_pid = {p.project_id: p for p in absorption.projects} if absorption else {}
    for i, pid in enumerate(wide.columns):
        c = colors[i % len(colors)]
        ax.plot(x, wide[pid].values, linewidth=1.8, color=c, label=pid)
        if pid in fc_by_pid:
            f = fc_by_pid[pid].forecast
            fx = [x[-1]] + list(f.index.to_timestamp())
            fy = [wide[pid].values[-1]] + list(f.values)
            ax.plot(fx, fy, linestyle="--", color=c, alpha=0.8)
    ax.set_title("Monthly absorption by project (dashed = forecast)")
    ax.set_ylabel("units sold")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "absorption.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    df = load_sales(db_path)
    df["ppsqm"] = df["sale_price_sar"] / df["area_sqm"]
    monthly = df.groupby([df["sale_date"].dt.to_period("Q"), "project_name"])["ppsqm"].median().unstack()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    monthly.index = monthly.index.to_timestamp()
    monthly.plot(ax=ax, marker="o", linewidth=1.5)
    ax.set_title("Median achieved price per sqm by project (quarterly)")
    ax.set_ylabel("SAR / sqm")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "price_per_sqm.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    snap = kpi_snapshot(db_path)
    bp = pd.DataFrame(snap["by_project"]).set_index("project")
    fig, ax = plt.subplots(figsize=(8, 4))
    bp[["sold", "available"]].plot(kind="barh", stacked=True, ax=ax, color=["#2a9d8f", "#e9c46a"])
    ax.set_title("Inventory status by project")
    ax.set_xlabel("units")
    ax.set_ylabel("")
    fig.tight_layout()
    p3 = out_dir / "inventory_status.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    paths.append(p3)
    return paths


def executive_summary(llm: LLM, snapshot: dict, pricing: PricingReport, absorption: AbsorptionReport, lang: str = "ar") -> str:
    lang_name = "Arabic" if lang == "ar" else "English"
    system = (
        f"You are a real-estate portfolio analyst writing for the executive committee of a Saudi developer. Write in {lang_name}.\n"
        "Using ONLY the figures provided, write a concise executive summary (max ~250 words) with: (1) headline KPIs, "
        "(2) project-level highlights and concerns (slow absorption, high remaining inventory, price trends), "
        "(3) the pricing-model accuracy in one sentence, (4) three concrete recommendations. "
        "Use markdown headings and bullets. Do not invent figures; round large numbers (e.g. 1.2 مليون ريال / SAR 1.2M)."
    )
    payload = {"kpis": snapshot, "pricing_model": pricing.to_dict(), "absorption": absorption.to_dict()}
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        temperature=0.2, max_tokens=900,
    ).strip()


def build_report(out_dir: Path = REPORTS_DIR, llm: LLM | None = None, langs: tuple[str, ...] = ("ar", "en"),
                 db_path: Path = DB_PATH) -> dict:
    """Train models, draw charts, write the bilingual markdown report. Returns paths and metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = llm or get_llm()
    snapshot = kpi_snapshot(db_path)
    _, pricing = train_pricing_model(load_sales(db_path))
    absorption = forecast_absorption(db_path)
    charts = make_charts(out_dir, db_path, absorption)
    (out_dir / "metrics.json").write_text(
        json.dumps({"kpis": snapshot, "pricing": pricing.to_dict(), "absorption": absorption.to_dict()}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    written = {}
    for lang in langs:
        summary = executive_summary(llm, snapshot, pricing, absorption, lang)
        title = "تقرير أداء المبيعات" if lang == "ar" else "Sales Performance Report"
        md = "\n".join([
            f"# {title} — {snapshot['as_of']}",
            "",
            summary,
            "",
            "---",
            "",
            "![absorption](absorption.png)",
            "",
            "![price per sqm](price_per_sqm.png)",
            "",
            "![inventory](inventory_status.png)",
            "",
            pricing.to_markdown(),
            "",
            absorption.to_markdown(),
            "",
            f"_Generated {date.today().isoformat()} by AqarIntel (LLM: {getattr(llm, 'name', '?')})._",
        ])
        path = out_dir / f"sales_report_{lang}.md"
        path.write_text(md, encoding="utf-8")
        written[lang] = str(path)
    return {"reports": written, "charts": [str(c) for c in charts], "snapshot": snapshot,
            "pricing": pricing.to_dict(), "absorption": absorption.to_dict()}
