import pandas as pd


ERROR_FILE = "results/validation/vsx_errors.csv"
FEATURE_FILE = "data/features/features.csv"

OUTPUT = "results/validation/vsx_error_analysis.csv"


def main():

    errors = pd.read_csv(ERROR_FILE)

    features = pd.read_csv(
        FEATURE_FILE
    )


    merged = errors.merge(
        features,
        on="tic_id",
        how="left"
    )


    columns = [
        "tic_id",
        "prediction",
        "vsx_classification",
        "flux_std",
        "flux_range",
        "flux_mad",
        "dominant_period",
        "period_power",
        "num_periods",
        "flare_count",
        "largest_flare",
        "dip_count",
        "largest_dip_depth",
    ]


    merged = merged[
        columns
    ]


    merged.to_csv(
        OUTPUT,
        index=False
    )


    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()