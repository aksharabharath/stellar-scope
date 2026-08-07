import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.timeseries import LombScargle


INPUT_FILE = (
    "results/final/"
    "ranked_final_candidates_v2.csv"
)

OUTPUT_DIR = (
    "results/final/"
    "validation"
)

TOP_N = 5


def download_lightcurve(tic_id):

    print(
        f"TIC {tic_id}: downloading light curve..."
    )

    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author="QLP"
    )

    if len(search) == 0:

        raise ValueError(
            "No QLP products found"
        )


    print("Using QLP")


    times = []
    fluxes = []


    for product in search[:3]:

        try:

            lc = product.download()

            if lc is None:
                continue


            time = np.asarray(
                lc.time.value,
                dtype=float
            )


            if "pdcsap_flux" in lc.columns:

                flux = np.asarray(
                    lc.pdcsap_flux.value,
                    dtype=float
                )

            elif "sap_flux" in lc.columns:

                flux = np.asarray(
                    lc.sap_flux.value,
                    dtype=float
                )

            else:

                flux = np.asarray(
                    lc.flux.value,
                    dtype=float
                )


            # FIX SHAPE MISMATCH
            n = min(
                len(time),
                len(flux)
            )

            time = time[:n]
            flux = flux[:n]


            mask = (
                np.isfinite(time)
                &
                np.isfinite(flux)
            )


            time = time[mask]
            flux = flux[mask]


            if len(time) < 100:
                continue


            flux = (
                flux /
                np.nanmedian(flux)
            )


            times.extend(
                time
            )

            fluxes.extend(
                flux
            )


        except Exception as e:

            print(
                f"Skipping product: {e}"
            )


    if len(times) == 0:

        raise ValueError(
            "No usable light curve"
        )


    time = np.array(
        times
    )

    flux = np.array(
        fluxes
    )


    order = np.argsort(
        time
    )


    time = time[order]
    flux = flux[order]


    return (
        time,
        flux
    )



def calculate_period(
    time,
    flux
):

    frequency, power = LombScargle(
        time,
        flux
    ).autopower(
        minimum_frequency=0.01,
        maximum_frequency=5
    )


    idx = np.argmax(
        power
    )


    return (
        1 / frequency[idx],
        float(power[idx])
    )



def save_plots(
    time,
    flux,
    period,
    tic,
    folder
):

    os.makedirs(
        folder,
        exist_ok=True
    )


    plt.figure(
        figsize=(10,4)
    )

    plt.scatter(
        time,
        flux,
        s=2
    )

    plt.xlabel(
        "BTJD"
    )

    plt.ylabel(
        "Normalized Flux"
    )

    plt.title(
        f"TIC {tic} Light Curve"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "light_curve.png"
        ),
        dpi=300
    )

    plt.close()



    phase = (
        (time % period)
        /
        period
    )


    order = np.argsort(
        phase
    )


    plt.figure(
        figsize=(8,4)
    )

    plt.scatter(
        phase[order],
        flux[order],
        s=3
    )

    plt.xlabel(
        "Phase"
    )

    plt.ylabel(
        "Flux"
    )

    plt.title(
        f"TIC {tic} Phase Folded"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            "phase_folded.png"
        ),
        dpi=300
    )

    plt.close()



def main():

    print(
        "Loading ranked candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    df = df.head(
        TOP_N
    )


    results = []


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"Validating TIC {tic}"
        )


        try:

            time, flux = download_lightcurve(
                tic
            )


            period, power = calculate_period(
                time,
                flux
            )


            folder = os.path.join(
                OUTPUT_DIR,
                f"TIC_{tic}"
            )


            save_plots(
                time,
                flux,
                period,
                tic,
                folder
            )


            results.append(
                {
                    "tic_id": tic,
                    "validation_status": "success",
                    "period_days": period,
                    "period_power": power,
                    "points": len(time)
                }
            )


            print(
                f"Complete: {period:.3f} days"
            )


        except Exception as e:

            print(
                f"Failed: {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "validation_status": "failed",
                    "period_days": np.nan,
                    "period_power": np.nan,
                    "points": 0
                }
            )


    pd.DataFrame(
        results
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_summary.csv"
        ),
        index=False
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved validation: {OUTPUT_DIR}"
    )



if __name__ == "__main__":
    main()