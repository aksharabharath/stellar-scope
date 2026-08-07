import os
import time

import pandas as pd
import requests


INPUT_FILE = (
    "results/validation/"
    "candidate_validation_summary.csv"
)

OUTPUT_DIR = (
    "results/validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR +
    "/vsx_crossmatch_candidates.csv"
)

STRONG_ONLY = True

SLEEP_SECONDS = 1


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def load_candidates():

    print(
        "Loading validation candidates..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    if STRONG_ONLY:

        df = df[
            df["validation_flag"]
            ==
            "strong_candidate"
        ]

    return df.reset_index(
        drop=True
    )


def query_vsx(
    tic_id
):
    """
    Query VSX through SIMBAD/VizieR cone search.

    Returns:
        match status,
        object name,
        details
    """

    url = (
        "https://vizier.cds.unistra.fr/"
        "viz-bin/asu-tsv"
    )

    params = {

        "source":
            "B/vsx",

        "TIC":
            tic_id,

        "-out.all":
            "",

        "-out.max":
            "5"

    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )


        if response.status_code != 200:

            return {
                "vsx_match": False,
                "vsx_name": "",
                "vsx_type": "",
                "vsx_period": "",
                "error":
                    "HTTP error"
            }


        text = response.text


        if (
            "No records found"
            in text
            or
            len(text.strip()) < 20
        ):

            return {

                "vsx_match": False,

                "vsx_name":
                    "",

                "vsx_type":
                    "",

                "vsx_period":
                    "",

                "error":
                    ""

            }


        lines = (
            text
            .splitlines()
        )


        data_lines = [
            line
            for line in lines
            if not line.startswith(
                "#"
            )
            and line.strip()
        ]


        if len(data_lines) < 2:

            return {

                "vsx_match": False,
                "vsx_name": "",
                "vsx_type": "",
                "vsx_period": "",
                "error": ""

            }


        header = (
            data_lines[0]
            .split("\t")
        )


        values = (
            data_lines[1]
            .split("\t")
        )


        record = dict(
            zip(
                header,
                values
            )
        )


        return {

            "vsx_match":
                True,

            "vsx_name":
                record.get(
                    "Name",
                    ""
                ),

            "vsx_type":
                record.get(
                    "Type",
                    ""
                ),

            "vsx_period":
                record.get(
                    "Period",
                    ""
                ),

            "error":
                ""

        }


    except Exception as e:

        return {

            "vsx_match":
                False,

            "vsx_name":
                "",

            "vsx_type":
                "",

            "vsx_period":
                "",

            "error":
                str(e)

        }


def classify_status(
    match
):

    if match:

        return (
            "known_variable"
        )

    return (
        "possible_new_candidate"
    )


def main():

    candidates = load_candidates()


    print(
        f"Loaded "
        f"{len(candidates)} candidates"
    )


    results = []


    for _, row in candidates.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"TIC {tic}: "
            "checking VSX..."
        )


        vsx = query_vsx(
            tic
        )


        result = {

            "tic_id":
                tic,

            "morphology":
                row.get(
                    "morphology",
                    ""
                ),

            "confidence":
                row.get(
                    "confidence",
                    ""
                ),

            "validation_flag":
                row.get(
                    "validation_flag",
                    ""
                ),

            **vsx,

            "classification":
                classify_status(
                    vsx[
                        "vsx_match"
                    ]
                )

        }


        results.append(
            result
        )


        print(
            f"  "
            f"{result['classification']}"
        )


        time.sleep(
            SLEEP_SECONDS
        )


    output = pd.DataFrame(
        results
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