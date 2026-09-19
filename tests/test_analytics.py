import warnings

warnings.filterwarnings("ignore")

from aqar_intel.analytics.absorption import damped_trend, forecast_absorption, mean_3, naive_last  # noqa: E402
from aqar_intel.analytics.pricing import load_sales, predict_price, train_pricing_model  # noqa: E402
from aqar_intel.analytics.report import kpi_snapshot  # noqa: E402
import numpy as np  # noqa: E402


def test_time_split_and_beats_baseline():
    df = load_sales()
    model, rep = train_pricing_model(df, test_months=6, save_to=None)
    assert rep.n_train + rep.n_test == len(df)
    assert rep.metrics_model["mape"] < rep.metrics_baseline["mape"]
    assert rep.metrics_model["r2"] > 0.9
    p = predict_price(model, dict(project_id="NRJ", city="Riyadh", unit_type="villa", view="garden", bedrooms=5,
                                  bathrooms=5, area_sqm=400, floor=None, parking_spaces=2))
    assert 1_000_000 < p < 10_000_000


def test_forecasters_are_non_negative_and_sane():
    y = np.array([10, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    for fn in (naive_last, mean_3, damped_trend):
        f = fn(y, 6)
        assert len(f) == 6 and (f >= 0).all()


def test_absorption_report():
    rep = forecast_absorption(horizon=6, backtest_window=6)
    assert len(rep.projects) == 6
    for p in rep.projects:
        assert p.method in p.backtest_mae and len(p.forecast) == 6
        assert p.backtest_mae[p.method] <= p.backtest_mae["naive_last"] + 1e-9


def test_kpi_snapshot_consistency():
    s = kpi_snapshot()
    u = s["units"]
    assert u["total"] == u["available"] + u["reserved"] + u["sold"]
    assert sum(p["sold"] for p in s["by_project"]) == u["sold"]
