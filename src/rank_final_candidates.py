import os
import pandas as pd
import numpy as np


INPUT_FILE = (
    "results/final/"
    "refined_physical_classification.csv"
)

OUTPUT_FILE = (
    "results/final/"
    "ranked_final_candidates.csv"
)

REPORT_FILE = (
    "results/final/"
    "ranked_final_candidates_report.md"
)


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def calculate_score(row):

    score = 0

    reasons = []

    classification = row.get(
        "refined_classification",
        ""
    )

    amplitude = safe_float(
        row.get("amplitude")
    )

    radius = safe_float(
        row.get("radius_solar")
    )

    confidence = safe_float(
        row.get("confidence")
    )


    if pd.notna(confidence):

        score += confidence * 2


    if "red_giant" in classification:

        score += 5

        reasons.append(
            "red giant variable candidate"
        )


    if "variable" in classification:

        score += 1


    if pd.notna(radius):

        if radius > 10:

            score += 3

            reasons.append(
                "large evolved-star radius"
            )


        elif radius > 2:

            score += 1

            reasons.append(
                "moderately enlarged stellar radius"
            )


    if pd.notna(amplitude):

        if amplitude > 0.2:

            score += 3

            reasons.append(
                "very high variability amplitude"
            )


        elif amplitude > 0.1:

            score += 2

            reasons.append(
                "high variability amplitude"
            )


        elif amplitude > 0.03:

            score += 1


    if "transient" in classification:

        score -= 3

        reasons.append(
            "transient or artifact uncertainty"
        )


    return (
        round(score, 2),
        "; ".join(reasons)
    )


def rank_candidates(df):

    scores = []

    explanations = []


    for _, row in df.iterrows():

        score, reason = calculate_score(
            row
        )

        scores.append(
            score
        )

        explanations.append(
            reason
        )


    df["candidate_score"] = scores

    df["reason"] = explanations


    df = df.sort_values(
        by=[
            "candidate_score",
            "confidence",
            "amplitude"
        ],
        ascending=False
    )


    df.insert(
        0,
        "rank",
        range(
            1,
            len(df) + 1
        )
    )


    return df


def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "# Final Variable Star Candidate Ranking Report\n\n"
        )

        f.write(
            "Candidates were ranked using a transparent scoring system "
            "combining stellar parameters, variability amplitude, "
            "classification confidence, and artifact penalties.\n\n"
        )


        f.write(
            "## Ranking Summary\n\n"
        )

        f.write(
            "| Rank | TIC ID | Score | Classification | Amplitude | Radius | Reason |\n"
        )

        f.write(
            "|---|---|---|---|---|---|---|\n"
        )


        for _, row in df.iterrows():

            f.write(
                "| "
                f"{row['rank']} | "
                f"{row['tic_id']} | "
                f"{row['candidate_score']} | "
                f"{row['refined_classification']} | "
                f"{row['amplitude']} | "
                f"{row['radius_solar']} | "
                f"{row['reason']} |\n"
            )


        f.write(
            "\n## Candidate Details\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## Rank {row['rank']}: TIC {row['tic_id']}\n\n"
            )

            f.write(
                f"- Score: {row['candidate_score']}\n"
            )

            f.write(
                f"- Classification: {row['refined_classification']}\n"
            )

            f.write(
                f"- Confidence: {row['confidence']}\n"
            )

            f.write(
                f"- Period: {row['period_days']} days\n"
            )

            f.write(
                f"- Amplitude: {row['amplitude']}\n"
            )

            f.write(
                f"- Temperature: {row['temperature_K']} K\n"
            )

            f.write(
                f"- Radius: {row['radius_solar']} solar radii\n"
            )

            f.write(
                f"- Evidence: {row['evidence']}\n"
            )

            f.write(
                f"- Ranking reason: {row['reason']}\n\n"
            )


def main():

    print(
        "Loading refined classifications..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(df)} candidates"
    )


    ranked = rank_candidates(
        df
    )


    ranked.to_csv(
        OUTPUT_FILE,
        index=False
    )


    write_report(
        ranked
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