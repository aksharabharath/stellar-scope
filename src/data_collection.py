"""
StellarScope Data Collection Pipeline

Downloads and preprocesses TESS light curves
from TIC target lists.
"""

from pathlib import Path
import pandas as pd
import lightkurve as lk
from tqdm import tqdm


RAW_METADATA = Path("data/metadata/targets.csv")
OUTPUT_DIR = Path("data/processed")


def clean_lightcurve(lc):
    """
    Standard TESS cleaning.
    """

    lc = lc.remove_nans()
    lc = lc.normalize()
    lc = lc.remove_outliers(sigma=5)

    return lc


def download_star(tic_id):
    """
    Download one TESS light curve.
    """

    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author="SPOC"
    )

    if len(search) == 0:
        return False

    lc = search[0].download()

    if lc is None:
        return False

    lc = clean_lightcurve(lc)

    df = pd.DataFrame(
        {
            "time": lc.time.value,
            "flux": lc.flux.value
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DIR / f"TIC_{tic_id}.csv",
        index=False
    )

    return True


def main():

    targets = pd.read_csv(
        RAW_METADATA
    )

    results = []

    for tic in tqdm(targets["tic_id"]):

        success = download_star(tic)

        results.append(
            {
                "tic_id": tic,
                "success": success
            }
        )


    pd.DataFrame(results).to_csv(
        "data/metadata/download_log.csv",
        index=False
    )


if __name__ == "__main__":
    main()