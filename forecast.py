"""Forecasting demo."""
import argparse
from data.synthetic_revenue import generate_revenue_series
from models.decomposition import decompose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--model", default="decompose")
    args = parser.parse_args()

    df = generate_revenue_series(n_months=48)
    series = df["revenue"]

    print(f"
Decomposing series (n={len(series)})...")
    result = decompose(series.values, period=12)
    print(f"Trend strength: {result.trend_strength}")
    print(f"Seasonal strength: {result.seasonal_strength}")

    if args.model in ("sarima", "ensemble"):
        try:
            from models.sarima import AutoSARIMA
            m = AutoSARIMA()
            m.fit(series)
            fc = m.predict(args.horizon)
            print(f"
Forecast (horizon={args.horizon}):")
            for i, (p, lo, hi) in enumerate(zip(fc.point, fc.lower, fc.upper)):
                print(f"  t+{i+1:02d}: {p:,.0f}  [{lo:,.0f}, {hi:,.0f}]")
        except ImportError:
            print("Install statsmodels for SARIMA: pip install statsmodels")

if __name__ == "__main__":
    main()
