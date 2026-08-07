import os

import numpy as np
import pandas as pd


INPUT_FILE = (
    "results/final/"
    "final_candidate_table.csv"
)

OUTPUT_FILE = (
    "results/final/"
    "physics_classification.csv"
)

REPORT_FILE = (
    "results/final/"
    "physics_classification_report.md"
)


def safe_float(value):

    try:

        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:

        return np.nan


def classify_physics(row):

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


    classification = "unknown_variable"

    confidence = 0.5

    reasoning = []


    if (
        "transient" in morphology
        or "flare" in morphology
    ):

        classification = (
            "flare_or_transient_candidate"
        )

        confidence = 0.85

        reasoning.append(
            "Transient-like morphology detected"
        )


    elif (
        pd.notna(period)
        and pd.notna(amplitude)
        and period < 1
        and amplitude > 0.05
    ):

        classification = (
            "rapid_rotational_or_pulsating_variable"
        )

        confidence = 0.75

        reasoning.append(
            "Short period and high amplitude variability"
        )


    elif (
        pd.notna(period)
        and pd.notna(amplitude)
        and 1 <= period <= 20
        and amplitude > 0.02
    ):

        classification = (
            "rotational_variable_candidate"
        )

        confidence = 0.8

        reasoning.append(
            "Moderate period consistent with stellar rotation"
        )


    elif (
        pd.notna(period)
        and period > 20
    ):

        classification = (
            "long_period_variable_candidate"
        )

        confidence = 0.7

        reasoning.append(
            "Long variability timescale"
        )


    if (
        pd.notna(radius)
        and radius > 5
    ):

        classification = (
            "giant_star_variable_candidate"
        )

        confidence = max(
            confidence,
            0.8
        )

        reasoning.append(
            "Large stellar radius suggests evolved star"
        )


    if (
        pd.notna(temperature)
        and temperature < 4500
    ):

        reasoning.append(
            "Cool stellar temperature"
        )


    if (
        pd.notna(amplitude)
        and amplitude > 0.1
    ):

        reasoning.append(
            "Large photometric variability amplitude"
        )


    return {

        "tic_id":
            row["tic_id"],

        "physics_classification":
            classification,

        "physics_confidence":
            confidence,

        "reasoning":
            "; ".join(
                reasoning
            ),

        "period_days":
            period,

        "amplitude":
            amplitude,

        "temperature_K":
            temperature,

        "radius_solar":
            radius

    }


def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:


        f.write(
            "# Physics-Based Candidate Classification Report\n\n"
        )


        f.write(
            "Candidates were interpreted using stellar "
            "properties and variability behavior.\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## TIC {row['tic_id']}\n\n"
            )

            f.write(
                f"- Classification: {row['physics_classification']}\n"
            )

            f.write(
                f"- Confidence: {row['physics_confidence']}\n"
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
                f"- Reasoning: {row['reasoning']}\n\n"
            )


def main():

    print(
        "Loading candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(df)} candidates"
    )


    results = []


    for _, row in df.iterrows():

        tic = row["tic_id"]

        print(
            f"TIC {tic}"
        )


        results.append(
            classify_physics(
                row
            )
        )


    output = pd.DataFrame(
        results
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