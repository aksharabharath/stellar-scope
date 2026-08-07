import os

import numpy as np
import pandas as pd


BASE_DIR = "results/validation"
OUTPUT_DIR = "results/final"


MORPHOLOGY_FILE = (
    f"{BASE_DIR}/candidate_morphology_classification.csv"
)

TIC_FILE = (
    f"{BASE_DIR}/tic_characterization.csv"
)

VSX_FILE = (
    f"{BASE_DIR}/vsx_coordinate_crossmatch.csv"
)

NEW_VARIABLE_FILE = (
    f"{BASE_DIR}/new_variable_candidate_analysis.csv"
)


OUTPUT_CSV = (
    f"{OUTPUT_DIR}/final_candidate_table.csv"
)

OUTPUT_REPORT = (
    f"{OUTPUT_DIR}/final_candidate_report.md"
)


def load_file(path):

    if os.path.exists(path):

        return pd.read_csv(path)

    print(
        f"Missing file: {path}"
    )

    return pd.DataFrame()


def merge_columns(df, other, columns):

    if other.empty:

        return df


    available = [
        c
        for c in columns
        if c in other.columns
    ]


    if "tic_id" not in available:

        return df


    return df.merge(
        other[available],
        on="tic_id",
        how="left",
        suffixes=("", "_extra")
    )


def get_value(row, key):

    if key in row.index:

        return row[key]

    return np.nan


def classify_priority(row):

    amplitude = get_value(
        row,
        "amplitude"
    )

    period = get_value(
        row,
        "period_days"
    )

    morphology = str(
        get_value(
            row,
            "morphology"
        )
    )

    vsx = get_value(
        row,
        "vsx_match"
    )


    if bool(vsx):

        return "known_variable"


    if (
        pd.notna(amplitude)
        and amplitude >= 0.15
    ):

        return "strong_candidate"


    if (
        pd.notna(amplitude)
        and amplitude >= 0.05
    ):

        return "moderate_candidate"


    if morphology == "periodic_variable":

        return "candidate"


    return "needs_review"


def scientific_interpretation(row):

    morphology = row.get(
        "morphology",
        ""
    )

    amplitude = row.get(
        "amplitude",
        np.nan
    )


    if morphology == "periodic_variable":

        if pd.notna(amplitude) and amplitude > 0.1:

            return (
                "Strong periodic variability. "
                "Possible rotational variable, "
                "pulsating star, or eclipsing system."
            )

        return (
            "Periodic variability detected. "
            "Requires additional classification."
        )


    if morphology == "transient_event":

        return (
            "Transient-like behavior detected. "
            "Requires inspection for instrumental "
            "effects or astrophysical events."
        )


    return (
        "Low-confidence variability candidate."
    )


def build_summary(df):

    rows = []


    for _, row in df.iterrows():

        rows.append(
            {
                "tic_id":
                    int(row["tic_id"]),

                "classification":
                    row.get(
                        "classification",
                        "possible_new_candidate"
                    ),

                "morphology":
                    row.get(
                        "morphology",
                        "unknown"
                    ),

                "confidence":
                    row.get(
                        "morphology_confidence",
                        row.get(
                            "confidence",
                            np.nan
                        )
                    ),

                "period_days":
                    row.get(
                        "period_days",
                        np.nan
                    ),

                "amplitude":
                    row.get(
                        "amplitude",
                        np.nan
                    ),

                "temperature_K":
                    row.get(
                        "Teff_K",
                        row.get(
                            "temperature_K",
                            np.nan
                        )
                    ),

                "radius_solar":
                    row.get(
                        "radius_solar",
                        np.nan
                    ),

                "mass_solar":
                    row.get(
                        "mass_solar",
                        np.nan
                    ),

                "luminosity_solar":
                    row.get(
                        "luminosity_solar",
                        np.nan
                    ),

                "vsx_match":
                    row.get(
                        "vsx_match",
                        False
                    ),

                "priority":
                    classify_priority(
                        row
                    ),

                "interpretation":
                    scientific_interpretation(
                        row
                    )
            }
        )


    return pd.DataFrame(rows)


def write_report(df):

    priority_order = {
        "strong_candidate": 0,
        "moderate_candidate": 1,
        "candidate": 2,
        "needs_review": 3,
        "known_variable": 4
    }


    df["sort"] = df["priority"].map(
        priority_order
    )


    df = df.sort_values(
        "sort"
    )


    with open(
        OUTPUT_REPORT,
        "w"
    ) as f:


        f.write(
            "# Stellar-Scope Final Variable Star Candidate Report\n\n"
        )


        f.write(
            "## Overview\n\n"
        )


        f.write(
            "This report summarizes candidate variable stars "
            "identified from TESS photometric observations. "
            "Candidates were evaluated using automated "
            "variability analysis, period detection, morphology "
            "classification, VSX catalog comparison, and TIC "
            "stellar characterization.\n\n"
        )


        f.write(
            "## Candidate Summary\n\n"
        )


        f.write(
            "| TIC | Morphology | Period (days) | Amplitude | Teff (K) | VSX | Priority |\n"
        )

        f.write(
            "|---|---|---|---|---|---|---|\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"| {row['tic_id']} | "
                f"{row['morphology']} | "
                f"{row['period_days']} | "
                f"{row['amplitude']} | "
                f"{row['temperature_K']} | "
                f"{row['vsx_match']} | "
                f"{row['priority']} |\n"
            )


        f.write(
            "\n## Candidate Details\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"## TIC {row['tic_id']}\n\n"
            )

            f.write(
                f"- Classification: {row['classification']}\n"
            )

            f.write(
                f"- Morphology: {row['morphology']}\n"
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
                f"- Radius: {row['radius_solar']} R☉\n"
            )

            f.write(
                f"- VSX match: {row['vsx_match']}\n"
            )

            f.write(
                f"- Priority: {row['priority']}\n"
            )

            f.write(
                f"- Interpretation: {row['interpretation']}\n\n"
            )


        f.write(
            "## Methodology\n\n"
        )

        f.write(
            "- TESS light curve acquisition through MAST/Lightkurve\n"
        )

        f.write(
            "- Flux normalization and variability measurement\n"
        )

        f.write(
            "- Lomb-Scargle period analysis\n"
        )

        f.write(
            "- Automated morphology classification\n"
        )

        f.write(
            "- VSX variable star catalog crossmatching\n"
        )

        f.write(
            "- TIC stellar parameter characterization\n"
        )


        f.write(
            "\n## Future Work\n\n"
        )

        f.write(
            "Future validation should include deeper inspection "
            "of phase-folded light curves, archival comparison, "
            "and follow-up observations to determine physical "
            "variability mechanisms.\n"
        )


def main():

    print(
        "Loading candidate files..."
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    morphology = load_file(
        MORPHOLOGY_FILE
    )

    tic = load_file(
        TIC_FILE
    )

    vsx = load_file(
        VSX_FILE
    )

    new_variables = load_file(
        NEW_VARIABLE_FILE
    )


    if morphology.empty:

        raise ValueError(
            "Missing morphology classification file"
        )


    df = morphology.copy()


    df = merge_columns(
        df,
        tic,
        [
            "tic_id",
            "Teff_K",
            "radius_solar",
            "mass_solar",
            "luminosity_solar"
        ]
    )


    df = merge_columns(
        df,
        vsx,
        [
            "tic_id",
            "vsx_match"
        ]
    )


    df = merge_columns(
        df,
        new_variables,
        [
            "tic_id",
            "priority",
            "classification"
        ]
    )


    summary = build_summary(
        df
    )


    summary.to_csv(
        OUTPUT_CSV,
        index=False
    )


    write_report(
        summary
    )


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