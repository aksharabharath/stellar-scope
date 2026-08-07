import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.timeseries import LombScargle


INPUT_FILE = (
    "results/final/"
    "final_candidate_table.csv"
)

OUTPUT_DIR = (
    "results/final/"
    "plots"
)

PERIOD_OUTPUT = (
    "results/final/"
    "period_analysis.csv"
)

MAX_SECTORS = 10


def download_lightcurve(tic_id):

    print(
        f"TIC {tic_id}: downloading light curve..."
    )

    searches = [
        ("SPOC", "SPOC"),
        ("QLP", "QLP"),
        ("MAST", None)
    ]

    search = None
    source_name = None


    for name, author in searches:

        try:

            if author:

                result = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    author=author
                )

            else:

                result = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS"
                )


            if len(result) > 0:

                search = result
                source_name = name

                print(
                    f"  Found {len(search)} products using {name}"
                )

                break


        except Exception as e:

            print(
                f"  {name} search failed: {e}"
            )


    if search is None:

        raise ValueError(
            "No TESS light curve products found"
        )


    times = []
    fluxes = []

    sectors_used = 0


    for product in search:

        if sectors_used >= MAX_SECTORS:
            break


        try:

            lc = product.download()

            if lc is None:
                continue


            if hasattr(
                lc,
                "pdcsap_flux"
            ):

                flux_column = lc.pdcsap_flux


            elif hasattr(
                lc,
                "sap_flux"
            ):

                flux_column = lc.sap_flux


            else:

                flux_column = lc.flux



            time = np.asarray(
                lc.time.value,
                dtype=float
            )


            flux = np.asarray(
                flux_column.value,
                dtype=float
            )


            mask = (
                np.isfinite(time)
                &
                np.isfinite(flux)
            )


            time = time[mask]
            flux = flux[mask]


            if len(time) < 100:
                continue


            median = np.nanmedian(
                flux
            )


            if median <= 0:
                continue


            flux = (
                flux /
                median
            )


            times.extend(
                time.tolist()
            )

            fluxes.extend(
                flux.tolist()
            )


            sectors_used += 1


        except Exception as e:

            print(
                f"  Sector skipped: {e}"
            )


    if len(times) == 0:

        raise ValueError(
            "No usable light curve data"
        )


    time = np.asarray(
        times
    )

    flux = np.asarray(
        fluxes
    )


    order = np.argsort(
        time
    )

    time = time[order]
    flux = flux[order]


    unique = np.ones(
        len(time),
        dtype=bool
    )

    unique[1:] = (
        np.diff(time) > 0
    )


    return (
        time[unique],
        flux[unique]
    )



def clean_flux(
    flux
):

    median = np.nanmedian(
        flux
    )

    deviation = np.abs(
        flux - median
    )


    sigma = np.nanstd(
        deviation
    )


    mask = (
        deviation <
        5 * sigma
    )


    return flux[mask]



def calculate_period(
    time,
    flux
):

    flux = clean_flux(
        flux
    )


    time = time[:len(flux)]


    frequency, power = LombScargle(
        time,
        flux
    ).autopower(
        minimum_frequency=1/100,
        maximum_frequency=5
    )


    best = np.argmax(
        power
    )


    period = (
        1 /
        frequency[best]
    )


    if (
        period < 0.1
        or period > 100
    ):

        period = np.nan


    return (
        period,
        frequency,
        power
    )



def save_lightcurve_plot(
    time,
    flux,
    tic,
    folder
):

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



def save_periodogram(
    frequency,
    power,
    tic,
    folder
):

    periods = (
        1 /
        frequency
    )


    plt.figure(
        figsize=(8,4)
    )


    plt.plot(
        periods,
        power
    )


    plt.xscale(
        "log"
    )


    plt.xlabel(
        "Period (days)"
    )


    plt.ylabel(
        "Lomb-Scargle Power"
    )


    plt.title(
        f"TIC {tic} Periodogram"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            folder,
            "periodogram.png"
        ),
        dpi=300
    )


    plt.close()



def save_phase_fold(
    time,
    flux,
    period,
    tic,
    folder
):

    if np.isnan(period):
        return


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
        "Normalized Flux"
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



def process_candidate(
    tic
):

    folder = os.path.join(
        OUTPUT_DIR,
        f"TIC_{tic}"
    )


    os.makedirs(
        folder,
        exist_ok=True
    )


    time, flux = download_lightcurve(
        tic
    )


    period, frequency, power = calculate_period(
        time,
        flux
    )


    save_lightcurve_plot(
        time,
        flux,
        tic,
        folder
    )


    save_periodogram(
        frequency,
        power,
        tic,
        folder
    )


    save_phase_fold(
        time,
        flux,
        period,
        tic,
        folder
    )


    return {
        "tic_id": tic,
        "period_days": period,
        "period_power": float(
            np.max(power)
        )
    }



def main():

    print(
        "Loading candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(df)} candidates"
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    results = []


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        try:

            result = process_candidate(
                tic
            )


            results.append(
                result
            )


            print(
                f"TIC {tic}: complete ({result['period_days']:.3f} days)"
            )


        except Exception as e:

            print(
                f"TIC {tic}: failed: {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "period_days": np.nan,
                    "period_power": np.nan
                }
            )


    output = pd.DataFrame(
        results
    )


    output.to_csv(
        PERIOD_OUTPUT,
        index=False
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved plots: {OUTPUT_DIR}"
    )

    print(
        f"Saved periods: {PERIOD_OUTPUT}"
    )



if __name__ == "__main__":
    main()