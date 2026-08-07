import os

import pandas as pd


RANKED_FILE = (
    "results/validation/"
    "ranked_variable_candidates.csv"
)

ANALYSIS_FILE = (
    "results/validation/"
    "candidate_analysis_refined.csv"
)

MORPHOLOGY_FILE = (
    "results/validation/"
    "candidate_morphology_classification.csv"
)

DIAGNOSTICS_DIR = (
    "results/validation/"
    "diagnostics"
)

OUTPUT_CSV = (
    "results/validation/"
    "candidate_validation_summary.csv"
)

OUTPUT_TXT = (
    "results/validation/"
    "candidate_validation_report.txt"
)


def load_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    return pd.read_csv(path)


def count_diagnostics():

    if not os.path.exists(
        DIAGNOSTICS_DIR
    ):

        return {}

    counts = {}

    for name in os.listdir(
        DIAGNOSTICS_DIR
    ):

        if name.startswith(
            "TIC_"
        ):

            tic = int(
                name.replace(
                    "TIC_",
                    ""
                )
            )

            counts[tic] = True

    return counts


def assign_validation_flag(row):

    morphology = row.get(
        "morphology",
        ""
    )

    confidence = row.get(
        "confidence",
        0
    )

    amplitude = row.get(
        "amplitude",
        0
    )

    period = row.get(
        "best_period_days",
        None
    )


    if (
        morphology
        ==
        "periodic_variable"
        and confidence >= 0.9
    ):

        return "strong_candidate"


    if (
        morphology
        ==
        "transient_event"
    ):

        return "needs_review"


    if (
        amplitude < 0.01
    ):

        return "weak_variability"


    if (
        period is not None
        and pd.notna(period)
    ):

        return "candidate_variable"


    return "uncertain"


def build_summary():

    print(
        "Loading validation results..."
    )


    ranked = load_file(
        RANKED_FILE
    )


    analysis = load_file(
        ANALYSIS_FILE
    )


    morphology = load_file(
        MORPHOLOGY_FILE
    )


    print(
        f"Loaded {len(ranked)} ranked candidates"
    )


    summary = ranked.merge(
        analysis,
        on="tic_id",
        how="left",
        suffixes=(
            "",
            "_analysis"
        )
    )


    summary = summary.merge(
        morphology,
        on="tic_id",
        how="left"
    )


    diagnostics = count_diagnostics()


    summary["diagnostics_generated"] = (
        summary["tic_id"]
        .map(diagnostics)
        .infer_objects(copy=False)
        .fillna(False)
    )


    summary["validation_flag"] = (
        summary.apply(
            assign_validation_flag,
            axis=1
        )
    )


    columns = [
        "tic_id",
        "prediction",
        "morphology",
        "confidence",
        "total_score",
        "amplitude",
        "rms",
        "mad",
        "robust_sigma",
        "max_sigma",
        "num_5sigma_outliers",
        "best_period_days",
        "time_span_days",
        "num_points",
        "num_sectors",
        "diagnostics_generated",
        "validation_flag"
    ]


    existing = [
        c
        for c in columns
        if c in summary.columns
    ]


    return summary[
        existing
    ]


def generate_report(df):

    lines = []

    lines.append(
        "Variable Candidate Validation Report"
    )

    lines.append(
        "=" * 45
    )

    lines.append("")

    lines.append(
        f"Total candidates analyzed: {len(df)}"
    )

    lines.append("")


    flag_counts = (
        df["validation_flag"]
        .value_counts()
    )


    lines.append(
        "Validation categories:"
    )


    for category, count in flag_counts.items():

        lines.append(
            f"  {category}: {count}"
        )


    lines.append("")

    lines.append(
        "Candidate Details:"
    )

    lines.append(
        "-" * 45
    )


    for _, row in df.iterrows():

        lines.append(
            f"TIC {row['tic_id']}"
        )

        lines.append(
            f"  Morphology: {row.get('morphology', 'unknown')}"
        )

        lines.append(
            f"  Confidence: {row.get('confidence', 'NA')}"
        )

        lines.append(
            f"  Score: {row.get('total_score', 'NA')}"
        )

        lines.append(
            f"  Period: {row.get('best_period_days', 'NA')} days"
        )

        lines.append(
            f"  Validation: {row['validation_flag']}"
        )

        lines.append("")


    return "\n".join(lines)


def main():

    df = build_summary()


    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    report = generate_report(
        df
    )


    with open(
        OUTPUT_TXT,
        "w"
    ) as f:

        f.write(
            report
        )


    print()

    print(
        "Finished."
    )

    print(
        f"Saved CSV: {OUTPUT_CSV}"
    )

    print(
        f"Saved report: {OUTPUT_TXT}"
    )


if __name__ == "__main__":

    main()