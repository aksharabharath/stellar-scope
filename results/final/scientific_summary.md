# Automated TESS Variable Star Candidate Discovery

## Project Overview

This project develops an automated pipeline for discovering and characterizing variable star candidates using TESS photometric observations. Candidates were identified through light curve variability analysis, periodicity detection, stellar parameter characterization, and physics-based ranking.

## Pipeline Methodology

1. TESS light curve acquisition using Lightkurve and MAST products
2. Flux normalization and variability measurement
3. Lomb-Scargle period analysis for periodic behavior
4. TIC stellar parameter characterization
5. Physics-based interpretation of variability
6. Transparent candidate scoring and ranking

## Top Candidate Summary

| Rank | TIC ID | Classification | Period (days) | Amplitude | Score |
|---|---|---|---|---|---|
| 1 | 397013579 | red_giant_variable_candidate | 14.46508530381148 | 0.1534403470896848 | 17.44 |
| 2 | 149951723 | red_giant_variable_candidate | 17.447234157658052 | 0.0517838867652918 | 16.0 |
| 3 | 380668425 | unclassified_variable_candidate | 12.084159854352722 | 0.1838622242967773 | 8.84 |
| 4 | 49503072 | unclassified_variable_candidate | 6.787358121154216 | 0.2956368320712953 | 8.0 |
| 5 | 376787680 | unclassified_variable_candidate | 0.2002715141065975 | 0.1183522859950925 | 7.84 |

## Candidate Interpretations

## TIC 397013579 (Rank 1)

- Classification: red_giant_variable_candidate
- Confidence: 0.72
- Period: 14.46508530381148 days
- Variability amplitude: 0.1534403470896848
- Effective temperature: 4795.0 K
- Stellar radius: 13.2206 solar radii
- Period behavior: long_period_variable
- Periodogram strength: 0.3100493816492404
- Interpretation: Consistent with an evolved giant variable candidate. Cool stellar temperature, large radius, and periodic multi-day variability suggest possible pulsational or rotational variability.

## TIC 149951723 (Rank 2)

- Classification: red_giant_variable_candidate
- Confidence: 0.5
- Period: 17.447234157658052 days
- Variability amplitude: 0.0517838867652918
- Effective temperature: 4269.0 K
- Stellar radius: 42.9515 solar radii
- Period behavior: long_period_variable
- Periodogram strength: 0.6014384463677453
- Interpretation: Consistent with an evolved giant variable candidate. Cool stellar temperature, large radius, and periodic multi-day variability suggest possible pulsational or rotational variability.

## TIC 380668425 (Rank 3)

- Classification: unclassified_variable_candidate
- Confidence: 0.42
- Period: 12.084159854352722 days
- Variability amplitude: 0.1838622242967773
- Effective temperature: N/A K
- Stellar radius: N/A solar radii
- Period behavior: long_period_variable
- Periodogram strength: 0.5940952831572415
- Interpretation: Variable source candidate requiring further classification.

## TIC 49503072 (Rank 4)

- Classification: unclassified_variable_candidate
- Confidence: 0.5
- Period: 6.787358121154216 days
- Variability amplitude: 0.2956368320712953
- Effective temperature: 6629.0 K
- Stellar radius: 1.66675 solar radii
- Period behavior: possible_rotation_or_pulsation
- Periodogram strength: 0.4412334377550881
- Interpretation: High-amplitude short-timescale variability may indicate rotational modulation, stellar activity, or pulsation.

## TIC 376787680 (Rank 5)

- Classification: unclassified_variable_candidate
- Confidence: 0.42
- Period: 0.2002715141065975 days
- Variability amplitude: 0.1183522859950925
- Effective temperature: N/A K
- Stellar radius: N/A solar radii
- Period behavior: short_period_variable
- Periodogram strength: 0.5751343796932321
- Interpretation: Short-period variability candidate requiring additional analysis to distinguish pulsation, rotation, or binary effects.

## Limitations

- Period detection was based on relative Lomb-Scargle peak strength and requires statistical validation.
- Some candidates lack complete stellar parameters from TIC characterization.
- Automated classifications represent candidate interpretations rather than confirmed variable star classes.
- Additional spectroscopy and archival observations would improve physical classification.

## Future Follow-Up

Future work should include crossmatching with additional variable star catalogs, analyzing phase-folded light curves in detail, estimating false alarm probabilities, and performing targeted follow-up observations for the highest priority candidates.
