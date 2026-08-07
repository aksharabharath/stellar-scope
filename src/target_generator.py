"""
StellarScope Target Generator

Creates a manageable TIC target sample
using a sky cone search.
"""

from pathlib import Path

import pandas as pd

from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u


OUTPUT = Path(
    "data/metadata/targets.csv"
)


def generate_targets(
    n_targets=500
):

    print("Querying TIC catalog...")


    # Center of a random sky field
    # (representative TESS region)
    coord = SkyCoord(
        ra=180,
        dec=0,
        unit="deg"
    )


    catalog = Catalogs.query_region(
        coord,
        radius=8 * u.deg,
        catalog="Tic"
    )


    df = catalog.to_pandas()


    print(
        f"Retrieved {len(df)} objects"
    )


    # Keep stars only
    df = df[
        df["objType"] == "STAR"
    ]


    # Remove missing identifiers
    df = df.dropna(
        subset=["ID"]
    )


    # Prefer brighter stars
    if "Tmag" in df.columns:

        df = df[
            df["Tmag"] < 12
        ]


    print(
        f"After filtering: {len(df)} stars"
    )


    targets = df.sample(
        n=min(n_targets, len(df)),
        random_state=42
    )


    output = pd.DataFrame(
        {
            "tic_id": targets["ID"].astype(int),
            "classification": "unknown",
            "source": "TIC"
        }
    )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    output.to_csv(
        OUTPUT,
        index=False
    )


    print(
        f"Saved {len(output)} targets"
    )


if __name__ == "__main__":
    generate_targets()