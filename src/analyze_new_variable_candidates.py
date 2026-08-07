import os
import numpy as np
import pandas as pd


INPUT_FILE = (
    "results/validation/"
    "potential_new_variables.csv"
)

REFINED_FILE = (
    "results/validation/"
    "candidate_analysis_refined.csv"
)

MORPHOLOGY_FILE = (
    "results/validation/"
    "candidate_morphology_classification.csv"
)

OUTPUT_FILE = (
    "results/validation/"
    "new_variable_candidate_analysis.csv"
)

REPORT_FILE = (
    "results/validation/"
    "new_variable_candidate_report.txt"
)


def load_csv(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return pd.read_csv(path)



def classify_priority(row):

    morphology = row.get(
        "morphology",
        ""
    )

    confidence = row.get(
        "morphology_confidence",
        0
    )

    amplitude = row.get(
        "amplitude",
        0
    )

    mad = row.get(
        "mad",
        0
    )


    if (
        morphology == "periodic_variable"
        and confidence >= 0.9
    ):

        if amplitude > 0.1:

            return "high_priority_periodic"


        return "periodic_candidate"


    if mad > 0.02:

        return "strong_variability"


    return "moderate_candidate"



def merge_files():

    print(
        "Loading candidate files..."
    )


    candidates = load_csv(
        INPUT_FILE
    )

    refined = load_csv(
        REFINED_FILE
    )

    morphology = load_csv(
        MORPHOLOGY_FILE
    )


    refined_cols = [
        "tic_id",
        "num_points",
        "num_sectors",
        "amplitude",
        "rms",
        "mad",
        "best_period_days",
        "time_span_days"
    ]

    refined = refined[
        [
            c for c in refined_cols
            if c in refined.columns
        ]
    ]


    morphology_cols = [
        "tic_id",
        "morphology",
        "morphology_confidence",
        "best_period_days"
    ]


    morphology = morphology[
        [
            c for c in morphology_cols
            if c in morphology.columns
        ]
    ]


    df = candidates.merge(
        refined,
        on="tic_id",
        how="left"
    )


    df = df.merge(
        morphology,
        on="tic_id",
        how="left",
        suffixes=(
            "",
            "_morph"
        )
    )


    return df



def analyze(df):

    results = []


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )

        print(
            f"TIC {tic}"
        )


        output = {

            "tic_id":
                tic,

            "classification":
                row.get(
                    "classification",
                    ""
                ),

            "morphology":
                row.get(
                    "morphology",
                    "unknown"
                ),

            "confidence":
                row.get(
                    "morphology_confidence",
                    np.nan
                ),

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

            "mad":
                row.get(
                    "mad",
                    np.nan
                ),

            "period_days":
                row.get(
                    "best_period_days",
                    np.nan
                ),

            "time_span_days":
                row.get(
                    "time_span_days",
                    np.nan
                ),

            "num_points":
                row.get(
                    "num_points",
                    np.nan
                ),

            "vsx_match":
                False
        }


        output["priority"] = (
            classify_priority(
                output
            )
        )


        results.append(
            output
        )


    return pd.DataFrame(
        results
    )



def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "NEW VARIABLE CANDIDATE REPORT\n"
        )

        f.write(
            "="*40+"\n\n"
        )


        f.write(
            f"Total candidates: {len(df)}\n\n"
        )


        f.write(
            "Priority:\n"
        )


        for k,v in (
            df["priority"]
            .value_counts()
            .items()
        ):

            f.write(
                f"{k}: {v}\n"
            )


        f.write(
            "\nCandidates:\n\n"
        )


        for _,row in df.iterrows():

            f.write(
                f"TIC {row['tic_id']}\n"
            )

            f.write(
                f"Type: {row['morphology']}\n"
            )

            f.write(
                f"Period: {row['period_days']} days\n"
            )

            f.write(
                f"Amplitude: {row['amplitude']}\n"
            )

            f.write(
                f"Priority: {row['priority']}\n\n"
            )



def main():

    df = merge_files()

    print(
        f"Loaded {len(df)} candidates"
    )


    results = analyze(
        df
    )


    results = results.sort_values(
        "priority"
    )


    results.to_csv(
        OUTPUT_FILE,
        index=False
    )


    write_report(
        results
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