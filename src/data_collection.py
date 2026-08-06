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
import shutil


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

    for attempt in range(2):

        try:

            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS"
            )


            if len(search) == 0:
                return None, "No light curve found"


            collection = search.download_all()


            if collection is None:
                return None, "Download failed"


            lc = collection.stitch()


            lc = lc.remove_nans()

            lc = lc.normalize()


            df = pd.DataFrame(
                {
                    "time": lc.time.value,
                    "flux": lc.flux.value
                }
            )


            return df, None


        except Exception as e:

            error = str(e)

            if "corrupt" in error.lower():

                print(
                    "Corrupt download detected. Clearing cache..."
                )

                cache = Path.home() / ".cache/lightkurve"


                if cache.exists():
                    shutil.rmtree(cache)


                continue


            return None, error


    return None, "Failed after retry"


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
    )


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

        if output_file.exists():
            print("Already exists, skipping")
            continue


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