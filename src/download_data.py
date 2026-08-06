"""
Download a test TESS light curve.

Purpose:
Verify that StellarScope can access astronomical data.
"""

import lightkurve as lk


def download_lightcurve(target):
    """
    Download the first available TESS light curve.
    """

    search_result = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    print(search_result)

    light_curve = search_result[0].download()

    return light_curve


if __name__ == "__main__":

    target = "TIC 25155310"

    lc = download_lightcurve(target)

    print("\nDownloaded light curve:")
    print(lc)