import os
import pandas as pd


ERROR_PATH = "results/validation/vsx_errors.csv"
FEATURE_PATH = "data/features/features.csv"

OUTPUT_PATH = "results/validation/vsx_error_analysis.csv"
SUMMARY_PATH = "results/validation/error_summary.txt"


FEATURE_COLUMNS = [
    "flux_std",
    "flux_range",
    "flux_mad",
    "flux_skew",
    "flux_kurtosis",
    "dominant_period",
    "period_power",
    "num_periods",
    "flare_count",
    "largest_flare",
    "dip_count",
    "largest_dip_depth",
]


def load_data():

    errors = pd.read_csv(ERROR_PATH)

    features = pd.read_csv(
        FEATURE_PATH
    )

    merged = errors.merge(
        features,
        on="tic_id",
        how="left"
    )

    return merged


def classify_error(row):

    prediction = row["prediction"]
    truth = row["vsx_classification"]

    if prediction == "quiet_star" and truth == "variable_star":
        return "missed_variable"

    if prediction == "flare_variable" and truth == "variable_star":
        return "flare_confusion"

    if prediction == "quiet_star" and truth == "flare_variable":
        return "missed_flare"

    if prediction == "variable_star" and truth == "flare_variable":
        return "weak_flare_detection"

    return "other"


def analyze_features(df):

    df["error_type"] = df.apply(
        classify_error,
        axis=1
    )

    return df


def generate_summary(df):

    lines = []

    lines.append(
        "Stellar-Scope VSX Error Analysis\n"
    )

    lines.append(
        f"Total errors: {len(df)}\n"
    )

    lines.append(
        "\nError categories:\n"
    )

    categories = (
        df["error_type"]
        .value_counts()
    )

    for name, count in categories.items():

        lines.append(
            f"{name}: {count}\n"
        )


    lines.append(
        "\nFeature averages by error type:\n"
    )


    summary = (
        df.groupby("error_type")[FEATURE_COLUMNS]
        .mean()
    )

    lines.append(
        summary.to_string()
    )


    lines.append(
        "\n\nMost suspicious missed variables:\n"
    )

    missed = df[
        df["error_type"] == "missed_variable"
    ].copy()


    if len(missed) > 0:

        missed = missed.sort_values(
            "flux_range",
            ascending=False
        )

        lines.append(
            missed[
                [
                    "tic_id",
                    "flux_range",
                    "flux_std",
                    "dominant_period",
                    "period_power",
                    "flare_count",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    return "\n".join(lines)


def main():

    print(
        "Loading VSX errors..."
    )

    df = load_data()


    print(
        f"Loaded {len(df)} errors"
    )


    df = analyze_features(
        df
    )


    os.makedirs(
        "results/validation",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        f"Saved: {OUTPUT_PATH}"
    )


    summary = generate_summary(
        df
    )


    with open(
        SUMMARY_PATH,
        "w"
    ) as f:

        f.write(summary)


    print(
        f"Saved: {SUMMARY_PATH}"
    )


    print()
    print(summary)


if __name__ == "__main__":
    main()