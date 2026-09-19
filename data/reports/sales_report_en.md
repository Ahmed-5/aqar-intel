# Sales Performance Report — 2026-08-26

### Headline KPIs (as of August 26, 2026)
* **Portfolio Sales:** SAR 2.99B realized across 1,412 sold units (66.3% of 2,130 total portfolio; 612 available, 106 reserved).
* **Recent Momentum:** 100 units sold (SAR 219.8M) in the last 3 months, reflecting a slowdown from 140 units in the previous quarter.
* **Commercial Terms & Financing:** Average discount maintained at 2.06%; bank financing drove 73% of sales (1,031 units vs. 381 cash).

### Project Highlights & Concerns
* **High Inventory Overhang:** *Coast Towers* commands the highest rate (SAR 12,138/sqm) but carries 159 unsold units and an extended 34.1 months of inventory (MOI). *Al Rawabi District* faces the slowest absorption (2 units/month; 36.5 MOI, 73 units left).
* **Subdued Velocity:** *Narjis Oasis* reflects sluggish demand with 27.0 MOI (108 units available at SAR 7,607/sqm).
* **Top Performers:** *Khobar Marina* shows rapid absorption with only 7.1 MOI remaining (64 units at SAR 9,937/sqm), while *Yasmin Suburb* leads total volume (376 sold; 13.1 MOI).

### Pricing-Model Accuracy
The valuation model demonstrates high precision with an $R^2$ of 0.981, a MAPE of 4.32%, and an MAE of SAR 98.2K, significantly outperforming the baseline price-per-sqm benchmark.

### Strategic Recommendations
1. **Accelerate Lagging Projects:** Implement targeted marketing and structured payment plans for *Coast Towers* and *Al Rawabi District* to mitigate >34-month inventory overhangs.
2. **Optimize Khobar Marina Pricing:** Leverage high demand and low supply (7.1 MOI) by capturing additional margin on the remaining 64 units.
3. **Deepen Mortgage Partnerships:** Strengthen tie-ups with commercial banks, given bank financing fuels the vast majority (73%) of portfolio absorption.

<!-- appendix -->
---

![absorption](absorption.png)

![price per sqm](price_per_sqm.png)

![inventory](inventory_status.png)

### Pricing model — train 1177 / test 235 sales (test from 2026-03)

| Metric | Gradient boosting | Baseline (price/sqm by project & type) |
|---|---|---|
| MAE (SAR) | 98,159 | 163,216 |
| MAPE | 4.32% | 6.86% |
| R² | 0.981 | 0.948 |

Permutation importance (test set, Δ MAPE):

- area_sqm: 0.3064
- project_id: 0.2107
- unit_type: 0.0936
- bathrooms: 0.0146
- view: 0.0100
- floor: 0.0051
- city: 0.0008
- bedrooms: 0.0007
- payment_method: 0.0003
- parking_spaces: 0.0000
- month_index: 0.0000

### Absorption forecast — next 6 months (backtest on last 6 months)

| Project | Best method | Backtest MAE (best / naive) | Last 3 months | Forecast next 3 | Remaining | Months of inventory |
|---|---|---|---|---|---|---|
| AQQ | damped_trend | 1.97 / 2.17 | 5, 8, 6 | 6, 6, 6 | 90 | 15.9 |
| CST | mean_3 | 1.83 / 2.33 | 5, 4, 5 | 5, 5, 5 | 159 | 34.1 |
| KHM | naive_last | 1.50 / 1.50 | 6, 6, 9 | 9, 9, 9 | 64 | 7.1 |
| NRJ | mean_3 | 0.78 / 1.00 | 3, 5, 4 | 4, 4, 4 | 108 | 27.0 |
| RWB | naive_last | 0.50 / 0.50 | 2, 2, 2 | 2, 2, 2 | 73 | 36.5 |
| YSM | naive_last | 1.33 / 1.33 | 8, 8, 9 | 9, 9, 9 | 118 | 13.1 |

_Generated 2026-09-20 by AqarIntel (LLM: openrouter:google/gemini-3.7-flash)._