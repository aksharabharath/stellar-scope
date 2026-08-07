import os

import pandas as pd
import numpy as np


INPUT_FILE = (
    "results/final/"
    "ranked_final_candidates_v2.csv"
)

OUTPUT_DIR = (
    "results/final"
)

OUTPUT_MARKDOWN = (
    OUTPUT_DIR +
    "/scientific_summary.md"
)

OUTPUT_CSV = (
    OUTPUT_DIR +
    "/top_candidates_table.csv"
)

TOP_N = 5


def safe(row, column):
    value = row.get(column, np.nan)

    if pd.isna(value):
        return "N/A"

    return value


def interpret_candidate(row):

    classification = str(
        row.get(
            "refined_classification",
            ""
        )
    )

    period_behavior = str(
        row.get(
            "period_behavior",
            ""
        )
    )


    if "red_giant" in classification:

        return (
            "Consistent with an evolved giant variable candidate. "
            "Cool stellar temperature, large radius, and periodic "
            "multi-day variability suggest possible pulsational "
            "or rotational variability."
        )


    if "rotation" in period_behavior:

        return (
            "High-amplitude short-timescale variability may indicate "
            "rotational modulation, stellar activity, or pulsation."
        )


    if "short_period" in period_behavior:

        return (
            "Short-period variability candidate requiring additional "
            "analysis to distinguish pulsation, rotation, or binary effects."
        )


    if "transient" in classification:

        return (
            "Transient-like behavior detected. Additional validation "
            "is required to separate astrophysical events from artifacts."
        )


    return (
        "Variable source candidate requiring further classification."
    )


def write_markdown(df):

    with open(
        OUTPUT_MARKDOWN,
        "w"
    ) as f:

        f.write(
            "# Automated TESS Variable Star Candidate Discovery\n\n"
        )


        f.write(
            "## Project Overview\n\n"
        )

        f.write(
            "This project develops an automated pipeline for discovering "
            "and characterizing variable star candidates using TESS "
            "photometric observations. Candidates were identified through "
            "light curve variability analysis, periodicity detection, "
            "stellar parameter characterization, and physics-based ranking.\n\n"
        )


        f.write(
            "## Pipeline Methodology\n\n"
        )

        methods = [
            "TESS light curve acquisition using Lightkurve and MAST products",
            "Flux normalization and variability measurement",
            "Lomb-Scargle period analysis for periodic behavior",
            "TIC stellar parameter characterization",
            "Physics-based interpretation of variability",
            "Transparent candidate scoring and ranking"
        ]


        for i, method in enumerate(methods, 1):

            f.write(
                f"{i}. {method}\n"
            )


        f.write(
            "\n## Top Candidate Summary\n\n"
        )


        f.write(
            "| Rank | TIC ID | Classification | Period (days) | Amplitude | Score |\n"
        )

        f.write(
            "|---|---|---|---|---|---|\n"
        )


        for _, row in df.iterrows():

            f.write(
                "| "
                f"{row['rank']} | "
                f"{row['tic_id']} | "
                f"{row['refined_classification']} | "
                f"{safe(row,'period_days')} | "
                f"{safe(row,'amplitude')} | "
                f"{safe(row,'candidate_score')} |\n"
            )


        f.write(
            "\n## Candidate Interpretations\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## TIC {row['tic_id']} (Rank {row['rank']})\n\n"
            )


            f.write(
                f"- Classification: {row['refined_classification']}\n"
            )

            f.write(
                f"- Confidence: {safe(row,'confidence')}\n"
            )

            f.write(
                f"- Period: {safe(row,'period_days')} days\n"
            )

            f.write(
                f"- Variability amplitude: {safe(row,'amplitude')}\n"
            )

            f.write(
                f"- Effective temperature: {safe(row,'temperature_K')} K\n"
            )

            f.write(
                f"- Stellar radius: {safe(row,'radius_solar')} solar radii\n"
            )

            f.write(
                f"- Period behavior: {safe(row,'period_behavior')}\n"
            )

            f.write(
                f"- Periodogram strength: {safe(row,'period_power')}\n"
            )

            f.write(
                f"- Interpretation: {interpret_candidate(row)}\n\n"
            )


        f.write(
            "## Limitations\n\n"
        )

        limitations = [
            "Period detection was based on relative Lomb-Scargle peak strength and requires statistical validation.",
            "Some candidates lack complete stellar parameters from TIC characterization.",
            "Automated classifications represent candidate interpretations rather than confirmed variable star classes.",
            "Additional spectroscopy and archival observations would improve physical classification."
        ]


        for item in limitations:

            f.write(
                f"- {item}\n"
            )


        f.write(
            "\n## Future Follow-Up\n\n"
        )

        f.write(
            "Future work should include crossmatching with additional "
            "variable star catalogs, analyzing phase-folded light curves "
            "in detail, estimating false alarm probabilities, and "
            "performing targeted follow-up observations for the highest "
            "priority candidates.\n"
        )


def main():

    print(
        "Loading ranked candidates..."
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(df)} candidates"
    )


    top = df.head(
        TOP_N
    )


    top.to_csv(
        OUTPUT_CSV,
        index=False
    )


    write_markdown(
        top
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved summary: {OUTPUT_MARKDOWN}"
    )

    print(
        f"Saved table: {OUTPUT_CSV}"
    )


if __name__ == "__main__":

    main()