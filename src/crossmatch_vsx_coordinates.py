import os
import time

import numpy as np
import pandas as pd

from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u


INPUT_FILE = (
    "results/validation/"
    "candidate_validation_summary.csv"
)

OUTPUT_DIR = (
    "results/validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR +
    "/vsx_coordinate_crossmatch.csv"
)

KNOWN_FILE = (
    OUTPUT_DIR +
    "/known_variables.csv"
)

NEW_FILE = (
    OUTPUT_DIR +
    "/potential_new_variables.csv"
)


SEARCH_RADIUS_ARCSEC = 30

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


def get_tic_coordinates(
    tic_id
):

    try:

        catalog = Catalogs.query_criteria(
            ID=tic_id,
            catalog="TIC"
        )


        if len(catalog) == 0:

            raise ValueError(
                "TIC not found"
            )


        row = catalog[0]


        ra = float(
            row["ra"]
        )

        dec = float(
            row["dec"]
        )


        return (
            ra,
            dec
        )


    except Exception as e:

        raise ValueError(
            f"TIC lookup failed: {e}"
        )


def query_vsx(
    ra,
    dec
):

    try:

        coord = SkyCoord(
            ra=ra,
            dec=dec,
            unit="deg"
        )


        vizier = Vizier(
            columns=[
                "*"
            ],
            row_limit=5
        )


        result = vizier.query_region(
            coord,
            radius=(
                SEARCH_RADIUS_ARCSEC *
                u.arcsec
            ),
            catalog="B/vsx"
        )


        if len(result) == 0:

            return {

                "vsx_match":
                    False,

                "vsx_name":
                    "",

                "vsx_type":
                    "",

                "vsx_period":
                    ""

            }


        table = result[0]


        first = table[0]


        return {

            "vsx_match":
                True,

            "vsx_name":
                str(
                    first["Name"]
                ),

            "vsx_type":
                str(
                    first["Type"]
                ),

            "vsx_period":
                str(
                    first["Period"]
                )

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
            f"TIC {tic}"
        )


        result = {

            "tic_id":
                tic,

            "morphology":
                row["morphology"],

            "confidence":
                row["confidence"],

            "validation_flag":
                row["validation_flag"]

        }


        try:

            print(
                "  Resolving coordinates..."
            )


            ra, dec = (
                get_tic_coordinates(
                    tic
                )
            )


            result["ra"] = ra
            result["dec"] = dec


            print(
                f"  RA={ra:.6f}, "
                f"DEC={dec:.6f}"
            )


            print(
                "  Searching VSX..."
            )


            vsx = query_vsx(
                ra,
                dec
            )


            result.update(
                vsx
            )


            if vsx["vsx_match"]:

                result[
                    "classification"
                ] = (
                    "known_variable"
                )

                print(
                    "  Known variable"
                )

            else:

                result[
                    "classification"
                ] = (
                    "possible_new_candidate"
                )

                print(
                    "  No VSX match"
                )


        except Exception as e:

            result.update(
                {
                    "classification":
                        "crossmatch_failed",

                    "error":
                        str(e)
                }
            )


            print(
                f"  Failed: {e}"
            )


        results.append(
            result
        )


        time.sleep(
            SLEEP_SECONDS
        )


    df = pd.DataFrame(
        results
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    df[
        df["classification"]
        ==
        "known_variable"
    ].to_csv(
        KNOWN_FILE,
        index=False
    )


    df[
        df["classification"]
        ==
        "possible_new_candidate"
    ].to_csv(
        NEW_FILE,
        index=False
    )


    print()
    print(
        "Finished."
    )


    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Known variables: {KNOWN_FILE}"
    )

    print(
        f"Potential new candidates: {NEW_FILE}"
    )


if __name__ == "__main__":

    main()