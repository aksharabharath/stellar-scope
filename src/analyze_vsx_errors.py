import os
import pandas as pd


ERROR_FILE = "results/validation/vsx_errors.csv"

FEATURE_FILE = "data/features/features.csv"

OUTPUT_DIR = "results/validation"

OUTPUT_CSV = (
    "results/validation/vsx_error_analysis.csv"
)

OUTPUT_TXT = (
    "results/validation/vsx_error_summary.txt"
)


FEATURE_COLUMNS = [
    "flux_std",
    "flux_range",
    "relative_flux_range",
    "coefficient_of_variation",
    "flux_mad",
    "flux_skew",
    "flux_kurtosis",
    "dominant_period",
    "period_power",
    "period_confidence",
    "num_periods",
    "flare_count",
    "flare_frequency",
    "largest_flare",
    "flare_strength",
    "dip_count",
    "largest_dip_depth",
]


def load_data():

    errors = pd.read_csv(
        ERROR_FILE
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    return errors, features



def categorize_error(row):

    prediction = row["prediction"]

    truth = row["vsx_classification"]


    if (
        prediction == "flare_variable"
        and truth == "periodic_variable"
    ):
        return "flare_to_periodic"


    if (
        prediction == "periodic_variable"
        and truth == "flare_variable"
    ):
        return "periodic_to_flare"


    if truth == "variable":

        return "generic_variable"


    return "other"



def analyze():

    print(
        "Loading VSX errors..."
    )


    errors, features = load_data()


    print(
        f"Loaded {len(errors)} errors"
    )


    merged = errors.merge(
        features,
        on="tic_id",
        how="left"
    )


    merged["error_type"] = merged.apply(
        categorize_error,
        axis=1
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    merged.to_csv(
        OUTPUT_CSV,
        index=False
    )


    print(
        f"Saved: {OUTPUT_CSV}"
    )


    generate_summary(
        merged
    )



def generate_summary(df):

    lines = []


    lines.append(
        "Stellar-Scope VSX Error Analysis"
    )

    lines.append(
        "=" * 40
    )

    lines.append("")


    lines.append(
        f"Total errors: {len(df)}"
    )


    lines.append("")


    lines.append(
        "Error categories:"
    )


    lines.append(
        str(
            df["error_type"]
            .value_counts()
        )
    )


    lines.append("")


    lines.append(
        "Average feature values by error type:"
    )


    feature_summary = (
        df
        .groupby("error_type")
        [
            FEATURE_COLUMNS
        ]
        .mean()
    )


    lines.append(
        str(feature_summary)
    )


    lines.append("")


    high_confidence = df[
        df["confidence"] >= 0.9
    ]


    lines.append(
        "High confidence mistakes:"
    )


    lines.append(
        f"{len(high_confidence)}"
    )


    if len(high_confidence) > 0:

        lines.append("")

        lines.append(
            str(
                high_confidence[
                    [
                        "tic_id",
                        "prediction",
                        "vsx_classification",
                        "confidence"
                    ]
                ]
            )
        )


    with open(
        OUTPUT_TXT,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )


    print(
        f"Saved: {OUTPUT_TXT}"
    )



if __name__ == "__main__":

    analyze()