import pandas as pd
from pathlib import Path


INPUT = Path("data/features/features.csv")
OUTPUT = Path("data/features/feature_quality_report.csv")


def main():

    df = pd.read_csv(INPUT)

    report = pd.DataFrame({
        "tic_id": df["tic_id"],
        "missing_values": df.isna().sum(axis=1),
        "extreme_variability": df["flux_std"] > 0.5,
        "large_flux_range": df["flux_range"] > 5,
        "negative_period": df["dominant_period"] < 0,
        "high_period_power": df["period_power"] > 0.5
    })

    report["quality_flags"] = (
        report.drop(columns="tic_id")
        .sum(axis=1)
    )

    report = report.sort_values(
        "quality_flags",
        ascending=False
    )

    OUTPUT.parent.mkdir(
        exist_ok=True
    )

    report.to_csv(
        OUTPUT,
        index=False
    )

    print(
        "Saved:",
        OUTPUT
    )

    print(report.head(10))


if __name__ == "__main__":
    main()