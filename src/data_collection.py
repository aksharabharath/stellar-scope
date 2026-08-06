"""
StellarScope Data Collection Pipeline

Downloads TESS light curves for TIC targets.

Input:
    data/metadata/targets.csv

Output:
    data/processed/TIC_<id>.csv
    data/metadata/download_log.csv
"""

from pathlib import Path

from jmespath import search
import pandas as pd
import lightkurve as lk


TARGET_FILE = Path(
    "data/metadata/targets.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

LOG_FILE = Path(
    "data/metadata/download_log.csv"
)


def download_light_curve(tic_id):
    """
    Download and preprocess one TIC light curve.
    """

    try:

        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS"
        )


        if len(search) == 0:
            return None, "No light curve found"


        # Download all available observations
        collection = search.download_all()


        if collection is None:
            return None, "Download failed"


        # Select longest observation
        lc = collection.stitch()


        if lc is None:
            return None, "Download failed"


        # Remove missing values
        lc = lc.remove_nans()


        # Normalize flux
        lc = lc.normalize()


        df = pd.DataFrame(
            {
                "time": lc.time.value,
                "flux": lc.flux.value
            }
        )


        return df, None


    except Exception as e:

        return None, str(e)



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    targets = pd.read_csv(
        TARGET_FILE
    ).head(5)


    logs = []


    print(
        f"Processing {len(targets)} targets..."
    )


    for index, row in targets.iterrows():

        tic_id = row["tic_id"]


        output_file = (
            OUTPUT_DIR /
            f"TIC_{tic_id}.csv"
        )


        print(
            f"\n[{index+1}/{len(targets)}] TIC {tic_id}"
        )


        df, error = download_light_curve(
            tic_id
        )


        if df is not None:

            df.to_csv(
                output_file,
                index=False
            )


            logs.append(
                {
                    "tic_id": tic_id,
                    "status": "success",
                    "rows": len(df),
                    "error": ""
                }
            )


            print(
                f"Saved {len(df)} rows"
            )


        else:

            logs.append(
                {
                    "tic_id": tic_id,
                    "status": "failed",
                    "rows": 0,
                    "error": error
                }
            )


            print(
                f"Failed: {error}"
            )


    log_df = pd.DataFrame(
        logs
    )


    log_df.to_csv(
        LOG_FILE,
        index=False
    )


    print(
        "\nDownload complete."
    )

    print(
        log_df["status"].value_counts()
    )



if __name__ == "__main__":
    main()