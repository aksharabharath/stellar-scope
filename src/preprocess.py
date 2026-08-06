"""
StellarScope preprocessing pipeline.

Cleans and normalizes TESS light curves
before feature extraction.
"""

import lightkurve as lk
import matplotlib.pyplot as plt


def clean_lightcurve(lc):
    """
    Apply basic cleaning steps.
    """

    # Remove missing observations
    lc = lc.remove_nans()

    # Normalize flux around median = 1
    lc = lc.normalize()

    # Remove extreme outliers
    lc = lc.remove_outliers(
        sigma=5
    )

    return lc


if __name__ == "__main__":

    target = "TIC 25155310"

    search_result = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    lc = search_result[0].download()

    cleaned_lc = clean_lightcurve(lc)

    print(cleaned_lc)

    cleaned_lc.plot()

    plt.show()