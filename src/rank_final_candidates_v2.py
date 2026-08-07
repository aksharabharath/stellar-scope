import os

import numpy as np
import pandas as pd


INPUT_FILE = (
    "results/final/"
    "final_candidates_with_periods.csv"
)

OUTPUT_DIR = (
    "results/final"
)

OUTPUT_CSV = (
    OUTPUT_DIR +
    "/ranked_final_candidates_v2.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR +
    "/ranked_final_candidates_v2_report.md"
)


def safe_value(row, key):
    value = row.get(key, np.nan)

    if pd.isna(value):
        return None

    return value


def calculate_candidate_score(row):

    score = 0

    reasons = []


    classification = str(
        row.get(
            "refined_classification",
            ""
        )
    )


    amplitude = safe_value(
        row,
        "amplitude"
    )


    radius = safe_value(
        row,
        "radius_solar"
    )


    temperature = safe_value(
        row,
        "temperature_K"
    )


    period_power = safe_value(
        row,
        "period_power"
    )


    period_confidence = safe_value(
        row,
        "period_confidence"
    )


    period_behavior = str(
        row.get(
            "period_behavior",
            ""
        )
    )


    confidence = safe_value(
        row,
        "confidence"
    )


    #
    # Physical classification
    #

    if "red_giant" in classification:

        score += 5

        reasons.append(
            "red giant stellar classification"
        )


    #
    # Stellar parameters
    #

    if radius is not None:

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


    if temperature is not None:

        if temperature < 5000:

            score += 1

            reasons.append(
                "cool stellar temperature"
            )


    #
    # Variability amplitude
    #

    if amplitude is not None:

        if amplitude > 0.2:

            score += 4

            reasons.append(
                "very high variability amplitude"
            )


        elif amplitude > 0.1:

            score += 3

            reasons.append(
                "high variability amplitude"
            )


        elif amplitude > 0.03:

            score += 1

            reasons.append(
                "measurable variability"
            )


    #
    # Period detection quality
    #

    if period_power is not None:

        if period_power > 0.5:

            score += 3

            reasons.append(
                "strong Lomb-Scargle period detection"
            )


        elif period_power > 0.25:

            score += 2

            reasons.append(
                "moderate period detection"
            )


        elif period_power < 0.05:

            score -= 2

            reasons.append(
                "weak period detection"
            )


    #
    # Period behavior
    #

    if period_behavior == "long_period_variable":

        score += 2

        reasons.append(
            "long-period coherent variability"
        )


    elif period_behavior == "possible_rotation_or_pulsation":

        score += 1

        reasons.append(
            "possible rotational or pulsational variability"
        )


    elif period_behavior == "short_period_variable":

        score += 1

        reasons.append(
            "short-period variability"
        )


    #
    # Confidence
    #

    if confidence is not None:

        score += (
            float(confidence)
            *
            2
        )


    #
    # Artifact penalty
    #

    if "transient" in classification:

        score -= 3

        reasons.append(
            "possible transient or artifact"
        )


    return (
        round(score, 2),
        "; ".join(reasons)
    )


def generate_report(df):

    with open(
        OUTPUT_REPORT,
        "w"
    ) as f:

        f.write(
            "# Final Variable Star Candidate Ranking Report\n\n"
        )

        f.write(
            "Candidates were ranked using a transparent scoring system "
            "combining stellar properties, variability amplitude, "
            "period detection strength, and physical classification.\n\n"
        )


        f.write(
            "## Ranking Summary\n\n"
        )


        f.write(
            "| Rank | TIC ID | Score | Classification | Period | Amplitude | Reason |\n"
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
                f"{row['period_days']} | "
                f"{row['amplitude']} | "
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
                f"- Period behavior: {row.get('period_behavior', 'unknown')}\n"
            )

            f.write(
                f"- Period power: {row.get('period_power', np.nan)}\n"
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
                f"- Evidence: {row.get('evidence', '')}\n"
            )

            f.write(
                f"- Ranking reason: {row['reason']}\n\n"
            )


def main():

    print(
        "Loading candidates with period analysis..."
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


    scores = []
    reasons = []


    for _, row in df.iterrows():

        score, reason = calculate_candidate_score(
            row
        )

        scores.append(
            score
        )

        reasons.append(
            reason
        )


    df["candidate_score"] = scores
    df["reason"] = reasons


    df = df.sort_values(
        by="candidate_score",
        ascending=False
    ).reset_index(
        drop=True
    )


    if "rank" in df.columns:
        df = df.drop(
            columns=["rank"]
        )


    df.insert(
        0,
        "rank",
        range(
            1,
            len(df) + 1
        )
    )


    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    generate_report(
        df
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved CSV: {OUTPUT_CSV}"
    )

    print(
        f"Saved report: {OUTPUT_REPORT}"
    )


if __name__ == "__main__":

    main()