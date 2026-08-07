import time

import numpy as np
import pandas as pd

from astroquery.mast import Catalogs


INPUT_FILE = (
    "results/validation/"
    "variable_characterization.csv"
)

OUTPUT_FILE = (
    "results/validation/"
    "tic_characterization.csv"
)

REPORT_FILE = (
    "results/validation/"
    "tic_characterization_report.txt"
)


RETRIES = 3
SLEEP_SECONDS = 5


def get_tic_data(tic_id):

    for attempt in range(RETRIES):

        try:

            catalog = Catalogs.query_object(
                f"TIC {tic_id}",
                catalog="Tic"
            )


            if len(catalog) == 0:

                raise ValueError(
                    "No TIC entry found"
                )


            return catalog[0]


        except Exception as e:

            print(
                f"  TIC query attempt {attempt + 1} failed: {e}"
            )

            time.sleep(
                SLEEP_SECONDS
            )


    raise ValueError(
        "TIC query failed"
    )


def safe_value(row, key):

    try:

        if key not in row.colnames:

            return np.nan


        value = row[key]


        if value is None:

            return np.nan


        if hasattr(value, "mask"):

            if value.mask:

                return np.nan


        return value.item()


    except Exception:

        return np.nan


def characterize_candidate(row):

    tic_id = int(
        row["tic_id"]
    )


    print(
        f"TIC {tic_id}"
    )


    tic = get_tic_data(
        tic_id
    )


    result = {

        "tic_id":
            tic_id,

        "ra":
            safe_value(
                tic,
                "ra"
            ),

        "dec":
            safe_value(
                tic,
                "dec"
            ),

        "Tmag":
            safe_value(
                tic,
                "Tmag"
            ),

        "Teff_K":
            safe_value(
                tic,
                "Teff"
            ),

        "radius_solar":
            safe_value(
                tic,
                "rad"
            ),

        "mass_solar":
            safe_value(
                tic,
                "mass"
            ),

        "luminosity_solar":
            safe_value(
                tic,
                "lum"
            ),

        "J_mag":
            safe_value(
                tic,
                "Jmag"
            ),

        "H_mag":
            safe_value(
                tic,
                "Hmag"
            ),

        "K_mag":
            safe_value(
                tic,
                "Kmag"
            ),

        "stellar_density":
            safe_value(
                tic,
                "rho"
            )
    }


    columns = [

        "classification",
        "morphology",
        "confidence",
        "period_days",
        "amplitude",
        "mad",
        "priority"

    ]


    for column in columns:

        if column in row.index:

            result[column] = row[column]


    return result


def write_report(df):

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        f.write(
            "TIC CHARACTERIZATION REPORT\n"
        )

        f.write(
            "===========================\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"TIC {row['tic_id']}\n"
            )

            f.write(
                f"Classification: {row.get('classification', '')}\n"
            )

            f.write(
                f"Morphology: {row.get('morphology', '')}\n"
            )

            f.write(
                f"Temperature: {row.get('Teff_K', np.nan)} K\n"
            )

            f.write(
                f"Radius: {row.get('radius_solar', np.nan)} Rsun\n"
            )

            f.write(
                f"Mass: {row.get('mass_solar', np.nan)} Msun\n"
            )

            f.write(
                f"TESS magnitude: {row.get('Tmag', np.nan)}\n"
            )

            f.write(
                f"Period: {row.get('period_days', np.nan)} days\n"
            )

            f.write(
                "\n"
            )


def main():

    print(
        "Loading candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    if "tic_id" not in df.columns:

        raise ValueError(
            "Missing tic_id column"
        )


    print(
        f"Loaded {len(df)} candidates"
    )


    results = []


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        try:

            result = characterize_candidate(
                row
            )


            results.append(
                result
            )


            print(
                "  complete"
            )


        except Exception as e:

            print(
                f"  failed: {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "error": str(e)
                }
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