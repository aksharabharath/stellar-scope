"""
Save cleaned TESS light curves.

Output:
data/processed/
"""

import lightkurve as lk
import pandas as pd
from pathlib import Path


def clean_lightcurve(lc):

    lc = lc.remove_nans()
    lc = lc.normalize()
    lc = lc.remove_outliers(
        sigma=5
    )

    return lc


def save_lightcurve(lc, target):

    output_dir = Path("data/processed")
    output_dir.mkdir(
        exist_ok=True
    )

    df = pd.DataFrame(
        {
            "time": lc.time.value,
            "flux": lc.flux.value
        }
    )

    filename = output_dir / f"{target}.csv"

    df.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":

    target = "TIC_25155310"

    search_result = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    lc = search_result[0].download()

    cleaned = clean_lightcurve(lc)

    save_lightcurve(
        cleaned,
        target
    )