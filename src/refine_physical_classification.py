import os

import numpy as np
import pandas as pd


INPUT_FILE = (
    "results/final/"
    "final_candidate_table.csv"
)

PHYSICS_FILE = (
    "results/final/"
    "physics_classification.csv"
)

OUTPUT_FILE = (
    "results/final/"
    "refined_physical_classification.csv"
)

REPORT_FILE = (
    "results/final/"
    "refined_physical_classification_report.md"
)


def safe_float(value):

    try:

        if pd.isna(value):

            return np.nan

        return float(value)

    except Exception:

        return np.nan


def variability_strength(amplitude):

    if pd.isna(amplitude):

        return 0.0

    if amplitude >= 0.2:

        return 1.0

    if amplitude >= 0.1:

        return 0.8

    if amplitude >= 0.03:

        return 0.5

    return 0.2


def classify_candidate(row):

    period = safe_float(
        row.get("period_days")
    )

    amplitude = safe_float(
        row.get("amplitude")
    )

    temperature = safe_float(
        row.get("temperature_K")
    )

    radius = safe_float(
        row.get("radius_solar")
    )

    morphology = str(
        row.get(
            "morphology",
            ""
        )
    ).lower()


    evidence = []

    score = 0.0


    classification = (
        "unclassified_variable_candidate"
    )


    variability = variability_strength(
        amplitude
    )


    score += (
        variability * 0.4
    )


    if (
        pd.notna(radius)
        and pd.notna(temperature)
        and radius > 5
        and temperature < 5000
    ):

        classification = (
            "red_giant_variable_candidate"
        )

        score += 0.3

        evidence.append(
            "cool giant stellar parameters"
        )


    elif (
        pd.notna(temperature)
        and pd.notna(radius)
        and temperature > 6000
        and radius < 3
        and pd.notna(period)
        and period < 20
    ):

        classification = (
            "pulsation_or_eclipsing_candidate"
        )

        score += 0.3

        evidence.append(
            "hot compact star with short periodic variability"
        )


    elif (
        pd.notna(temperature)
        and temperature < 5000
        and pd.notna(period)
        and period < 30
    ):

        classification = (
            "rotational_spot_candidate"
        )

        score += 0.25

        evidence.append(
            "cool star with rotation-like timescale"
        )


    if (
        pd.notna(period)
        and period < 1
    ):

        evidence.append(
            "very short period behavior"
        )

        score += 0.1


    if (
        pd.notna(period)
        and 1 <= period <= 30
    ):

        evidence.append(
            "periodic variability timescale"
        )

        score += 0.1


    if (
        pd.notna(amplitude)
        and amplitude > 0.1
    ):

        evidence.append(
            "high photometric amplitude"
        )

        score += 0.1


    if (
        "transient" in morphology
    ):

        classification = (
            "transient_or_artifact_candidate"
        )

        evidence.append(
            "transient morphology classification"
        )


    confidence = min(
        round(score, 2),
        0.95
    )


    if confidence < 0.4:

        confidence = 0.4


    if confidence >= 0.75:

        priority = "high"

    elif confidence >= 0.55:

        priority = "medium"

    else:

        priority = "low"


    return {

        "tic_id":
            row["tic_id"],

        "refined_classification":
            classification,

        "confidence":
            confidence,

        "priority":
            priority,

        "period_days":
            period,

        "amplitude":
            amplitude,

        "temperature_K":
            temperature,

        "radius_solar":
            radius,

        "evidence":
            "; ".join(
                evidence
            )

    }


def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "# Refined Physical Classification Report\n\n"
        )

        f.write(
            "Candidates were ranked using variability amplitude, "
            "period behavior, morphology, and TIC stellar parameters.\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## TIC {row['tic_id']}\n\n"
            )

            f.write(
                f"- Classification: {row['refined_classification']}\n"
            )

            f.write(
                f"- Confidence: {row['confidence']}\n"
            )

            f.write(
                f"- Priority: {row['priority']}\n"
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
                f"- Radius: {row['radius_solar']} Rsun\n"
            )

            f.write(
                f"- Evidence: {row['evidence']}\n\n"
            )


def main():

    print(
        "Loading candidate files..."
    )


    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            INPUT_FILE
        )


    candidates = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(candidates)} candidates"
    )


    if os.path.exists(PHYSICS_FILE):

        physics = pd.read_csv(
            PHYSICS_FILE
        )


        candidates = candidates.merge(
            physics[
                [
                    "tic_id",
                    "physics_classification"
                ]
            ],
            on="tic_id",
            how="left"
        )


    results = []


    for _, row in candidates.iterrows():

        print(
            f"TIC {row['tic_id']}"
        )


        results.append(
            classify_candidate(
                row
            )
        )


    output = pd.DataFrame(
        results
    )


    output = output.sort_values(
        by=[
            "priority",
            "confidence"
        ],
        ascending=[
            True,
            False
        ]
    )


    output.to_csv(
        OUTPUT_FILE,
        index=False
    )


    write_report(
        output
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