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

    print("Decomposing series (n=" + str(len(series)) + ")...")
    result = decompose(series.values, period=12)
    print("Trend strength:", result.trend_strength)
    print("Seasonal strength:", result.seasonal_strength)

    if args.model in ("sarima", "ensemble"):
        try:
            from models.sarima import AutoSARIMA
            m = AutoSARIMA()
            m.fit(series)
            fc = m.predict(args.horizon)
            print("Forecast (horizon=" + str(args.horizon) + "):")
            for i, (p, lo, hi) in enumerate(zip(fc.point, fc.lower, fc.upper)):
                print("  t+" + str(i+1).zfill(2) + ": " + str(round(p)) + "  [" + str(round(lo)) + ", " + str(round(hi)) + "]")
        except ImportError:
            print("Install statsmodels for SARIMA: pip install statsmodels")


if __name__ == "__main__":
    main()
