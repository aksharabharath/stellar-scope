import os

import numpy as np
import pandas as pd


INPUT_FILE = (
    "results/validation/"
    "candidate_analysis_refined.csv"
)

OUTPUT_FILE = (
    "results/validation/"
    "candidate_morphology_classification.csv"
)


def classify_morphology(row):
    """
    Classify variable star morphology using
    light curve statistics.
    """

    amplitude = row.get(
        "amplitude",
        np.nan
    )

    rms = row.get(
        "rms",
        np.nan
    )

    robust_sigma = row.get(
        "robust_sigma",
        np.nan
    )

    max_sigma = row.get(
        "max_sigma",
        np.nan
    )

    outliers = row.get(
        "num_5sigma_outliers",
        0
    )

    period = row.get(
        "best_period_days",
        np.nan
    )


    if not np.isfinite(
        amplitude
    ):

        return "unknown"


    # Large isolated events
    if (
        max_sigma > 100
        and outliers < 500
    ):

        return "transient_event"


    # Repeated strong excursions
    if (
        outliers > 100
        and amplitude > 2
    ):

        return "flare_like"


    # Strong periodic modulation
    if (
        np.isfinite(period)
        and period < 20
        and amplitude > 0.05
    ):

        return "periodic_variable"


    # Long period / slow variability
    if (
        np.isfinite(period)
        and period >= 20
    ):

        return "long_period_variable"


    # Low amplitude changes
    if (
        amplitude < 0.1
        and rms < 0.02
    ):

        return "low_amplitude_variable"


    return "irregular_variable"



def assign_confidence(row):
    """
    Estimate confidence in morphology label.
    """

    morphology = row["morphology"]

    amplitude = row.get(
        "amplitude",
        np.nan
    )

    max_sigma = row.get(
        "max_sigma",
        np.nan
    )


    if morphology == "transient_event":

        if max_sigma > 1000:
            return 0.95

        return 0.80


    if morphology == "flare_like":

        if amplitude > 3:
            return 0.95

        return 0.75


    if morphology == "periodic_variable":

        return 0.90


    if morphology == "long_period_variable":

        return 0.80


    if morphology == "low_amplitude_variable":

        return 0.70


    return 0.50



def main():

    print(
        "Loading refined analysis..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Classifying {len(df)} candidates..."
    )


    classifications = []


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        morphology = classify_morphology(
            row
        )


        confidence = assign_confidence(
            {
                **row,
                "morphology": morphology
            }
        )


        classifications.append(
            {
                "tic_id": tic,

                "prediction":
                    row.get(
                        "prediction",
                        ""
                    ),

                "model_confidence":
                    row.get(
                        "confidence",
                        np.nan
                    ),

                "total_score":
                    row.get(
                        "total_score",
                        np.nan
                    ),

                "morphology":
                    morphology,

                "morphology_confidence":
                    confidence,

                "amplitude":
                    row.get(
                        "amplitude",
                        np.nan
                    ),

                "rms":
                    row.get(
                        "rms",
                        np.nan
                    ),

                "robust_sigma":
                    row.get(
                        "robust_sigma",
                        np.nan
                    ),

                "max_sigma":
                    row.get(
                        "max_sigma",
                        np.nan
                    ),

                "num_5sigma_outliers":
                    row.get(
                        "num_5sigma_outliers",
                        np.nan
                    ),

                "best_period_days":
                    row.get(
                        "best_period_days",
                        np.nan
                    )
            }
        )


        print(
            f"TIC {tic}: {morphology}"
        )


    output = pd.DataFrame(
        classifications
    )


    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )


    output.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()