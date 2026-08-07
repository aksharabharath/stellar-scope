import os
import pandas as pd
import numpy as np


FINAL_DIR = "results/final"

RANKED_FILE = os.path.join(
    FINAL_DIR,
    "ranked_final_candidates.csv"
)

PERIOD_FILE = os.path.join(
    FINAL_DIR,
    "period_analysis.csv"
)

OUTPUT_FILE = os.path.join(
    FINAL_DIR,
    "final_candidates_with_periods.csv"
)

REPORT_FILE = os.path.join(
    FINAL_DIR,
    "final_period_analysis_report.md"
)


def load_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    return pd.read_csv(path)


def classify_period(period):

    if pd.isna(period):
        return (
            "unknown",
            0.0
        )

    if period < 0.5:
        return (
            "short_period_variable",
            0.5
        )

    if period < 10:
        return (
            "possible_rotation_or_pulsation",
            0.7
        )

    if period < 50:
        return (
            "long_period_variable",
            0.8
        )

    return (
        "very_long_period_variable",
        0.6
    )


def add_period_classification(df):

    classes = []
    confidence = []

    for _, row in df.iterrows():

        classification, score = classify_period(
            row.get(
                "period_days",
                np.nan
            )
        )

        classes.append(
            classification
        )

        confidence.append(
            score
        )

    df["period_behavior"] = classes

    df["period_confidence"] = confidence

    return df


def merge_period_data(
    ranked,
    periods
):

    if periods.empty:
        print(
            "No period analysis file found. Continuing without periods."
        )

        return ranked


    keep = [
        "tic_id",
        "period_days",
        "period_power"
    ]


    available = [
        c for c in keep
        if c in periods.columns
    ]


    if len(available) == 0:

        return ranked


    merged = ranked.merge(
        periods[available],
        on="tic_id",
        how="left",
        suffixes=(
            "",
            "_period"
        )
    )


    if (
        "period_days_period"
        in merged.columns
    ):

        merged["period_days"] = (
            merged["period_days_period"]
            .fillna(
                merged["period_days"]
            )
        )


        merged.drop(
            columns=[
                "period_days_period"
            ],
            inplace=True
        )


    return merged


def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "# Final Candidate Period Analysis Report\n\n"
        )

        f.write(
            "Period information was incorporated using "
            "Lomb-Scargle analysis of TESS light curves. "
            "Detected timescales were classified into "
            "possible physical variability regimes.\n\n"
        )


        f.write(
            "| Rank | TIC | Period (days) | Behavior | Classification |\n"
        )

        f.write(
            "|---|---|---|---|---|\n"
        )


        for _, row in df.iterrows():

            f.write(
                "| "
                f"{row.get('rank')} | "
                f"{row['tic_id']} | "
                f"{row.get('period_days', np.nan)} | "
                f"{row.get('period_behavior')} | "
                f"{row.get('refined_classification')} |"
                "\n"
            )


        f.write(
            "\n## Candidate Details\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## TIC {row['tic_id']}\n\n"
            )

            f.write(
                f"- Rank: {row.get('rank')}\n"
            )

            f.write(
                f"- Classification: "
                f"{row.get('refined_classification')}\n"
            )

            f.write(
                f"- Period: "
                f"{row.get('period_days', np.nan)} days\n"
            )

            f.write(
                f"- Period behavior: "
                f"{row.get('period_behavior')}\n"
            )

            f.write(
                f"- Period confidence: "
                f"{row.get('period_confidence')}\n\n"
            )


def main():

    print(
        "Loading ranked candidates..."
    )


    ranked = load_file(
        RANKED_FILE
    )


    print(
        f"Loaded {len(ranked)} candidates"
    )


    if os.path.exists(
        PERIOD_FILE
    ):

        periods = load_file(
            PERIOD_FILE
        )

        print(
            f"Loaded {len(periods)} period measurements"
        )

    else:

        periods = pd.DataFrame()

        print(
            "No period_analysis.csv found"
        )


    merged = merge_period_data(
        ranked,
        periods
    )


    merged = add_period_classification(
        merged
    )


    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )


    write_report(
        merged
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved CSV: {OUTPUT_FILE}"
    )

    print(
        f"Saved report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()