import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk


CANDIDATE_FILE = (
    "results/validation/"
    "ranked_variable_candidates.csv"
)

OUTPUT_DIR = (
    "results/validation/"
    "candidate_lightcurves"
)

STATUS_FILE = (
    "results/validation/"
    "candidate_lightcurves_status.csv"
)

TOP_N = 10
MAX_SECTORS = 3
MIN_POINTS = 50
MAX_PLOT_POINTS = 5000


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def load_candidates():

    print(
        "Loading ranked candidates..."
    )

    df = pd.read_csv(
        CANDIDATE_FILE
    )

    required = [
        "tic_id",
        "prediction",
        "confidence",
        "total_score"
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    return df.head(
        TOP_N
    )


def convert_to_array(data):

    try:

        data = data.value

    except AttributeError:

        pass

    return np.asarray(
        data,
        dtype=float
    )


def normalize_flux(flux):

    flux = convert_to_array(
        flux
    )

    valid = np.isfinite(
        flux
    )

    flux = flux[valid]


    if len(flux) < MIN_POINTS:

        raise ValueError(
            "Not enough flux points"
        )


    median = np.median(
        flux
    )


    if not np.isfinite(median):

        raise ValueError(
            "Invalid flux median"
        )


    if abs(median) < 1e-10:

        raise ValueError(
            "Flux median too small"
        )


    return flux / median



def extract_lightcurve(lc):

    flux = None


    if "pdcsap_flux" in lc.columns:

        flux = lc["pdcsap_flux"]


    elif "sap_flux" in lc.columns:

        flux = lc["sap_flux"]


    elif hasattr(lc, "flux"):

        flux = lc.flux


    if flux is None:

        raise ValueError(
            "No usable flux column"
        )


    try:

        flux = flux.value

    except AttributeError:

        pass


    try:

        time_values = lc.time.value

    except AttributeError:

        time_values = lc.time


    flux = np.asarray(
        flux,
        dtype=float
    )


    time_values = np.asarray(
        time_values,
        dtype=float
    )


    min_length = min(
        len(flux),
        len(time_values)
    )


    flux = flux[
        :min_length
    ]

    time_values = time_values[
        :min_length
    ]


    valid = (
        np.isfinite(flux)
        &
        np.isfinite(time_values)
    )


    flux = flux[
        valid
    ]

    time_values = time_values[
        valid
    ]


    if len(flux) < MIN_POINTS:

        raise ValueError(
            "Not enough valid points"
        )


    median = np.median(
        flux
    )


    if not np.isfinite(median):

        raise ValueError(
            "Invalid median"
        )


    if abs(median) < 1e-10:

        raise ValueError(
            "Median flux too small"
        )


    normalized_flux = (
        flux /
        median
    )


    return (
        time_values,
        normalized_flux
    )


def download_sector_data(tic):

    print(
        f"TIC {tic}: searching MAST..."
    )


    search = lk.search_lightcurve(
        f"TIC {tic}",
        mission="TESS"
    )


    if len(search) == 0:

        raise ValueError(
            "No TESS observations"
        )


    search = search[
        :MAX_SECTORS
    ]


    print(
        f"TIC {tic}: downloading "
        f"{len(search)} sectors"
    )


    curves = []


    for i, item in enumerate(search):

        print(
            f"TIC {tic}: sector "
            f"{i+1}/{len(search)}"
        )


        try:

            lc = item.download()


            if lc is None:

                continue


            time_values, flux = (
                extract_lightcurve(
                    lc
                )
            )


            sector = getattr(
                lc,
                "sector",
                i + 1
            )


            curves.append(
                {
                    "sector": sector,
                    "time": time_values,
                    "flux": flux
                }
            )


        except Exception as e:

            print(
                f"Sector failed: {e}"
            )


    if len(curves) == 0:

        raise ValueError(
            "No usable sectors"
        )


    return curves



def downsample(
    x,
    y
):

    if len(x) <= MAX_PLOT_POINTS:

        return x, y


    indices = np.linspace(
        0,
        len(x)-1,
        MAX_PLOT_POINTS
    ).astype(int)


    return (
        x[indices],
        y[indices]
    )



def plot_candidate(
    tic,
    curves,
    row
):

    path = os.path.join(
        OUTPUT_DIR,
        f"TIC_{tic}_lightcurve.png"
    )


    plt.figure(
        figsize=(12,5)
    )


    for curve in curves:

        x, y = downsample(
            curve["time"],
            curve["flux"]
        )


        plt.plot(
            x,
            y,
            linewidth=0.5,
            label=(
                f"Sector "
                f"{curve['sector']}"
            )
        )


    plt.axhline(
        1,
        linestyle="--",
        linewidth=0.8
    )


    plt.title(
        f"TIC {tic} | "
        f"{row['prediction']} | "
        f"Confidence "
        f"{row['confidence']:.3f}"
    )


    plt.xlabel(
        "Time (BTJD)"
    )


    plt.ylabel(
        "Normalized Flux"
    )


    plt.legend()

    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {path}"
    )


    return path



def main():

    candidates = load_candidates()


    print(
        f"Processing "
        f"{len(candidates)} candidates..."
    )


    results = []


    for index, row in candidates.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"\n[{index+1}/{len(candidates)}] "
            f"TIC {tic}"
        )


        start = time.time()


        try:

            curves = download_sector_data(
                tic
            )


            output = plot_candidate(
                tic,
                curves,
                row
            )


            results.append(
                {
                    "tic_id": tic,
                    "status": "success",
                    "output": output,
                    "seconds": round(
                        time.time()-start,
                        2
                    )
                }
            )


        except Exception as e:

            print(
                f"TIC {tic}: failed {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "status": "failed",
                    "reason": str(e)
                }
            )


    pd.DataFrame(
        results
    ).to_csv(
        STATUS_FILE,
        index=False
    )


    print(
        "\nFinished."
    )


    print(
        f"Saved status: {STATUS_FILE}"
    )



if __name__ == "__main__":

    main()