"""
tools.py — Exoplanet Swarm
──────────────────────────
All NASA MAST data retrieval, photometric signal processing, and
Box-fitting Least Squares (BLS) math live here as @tool-decorated
functions that CrewAI agents can invoke.

No agent or LLM references in this file — pure astronomy + data science.
"""

import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from crewai.tools import tool
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════
#  TOOL 1: NASA MAST Data Retrieval
# ══════════════════════════════════════════════════════════════════

@tool("Fetch Lightcurve")
def fetch_lightcurve_tool(star_id: str) -> str:
    """
    Searches the NASA MAST archive for photometric light curve data for a
    given star identifier (e.g. 'Kepler-186', 'TOI 700', 'TIC 261136679').

    Downloads all available Kepler/TESS PDCSAP_FLUX observations, stitches
    multi-quarter data into a single time series, and normalizes the flux
    to unit median.

    Returns a JSON string containing:
        - 'time'   : list of BJD timestamps (sorted ascending)
        - 'flux'   : list of normalized flux values
        - 'records': number of cadences
        - 'mission': mission name string (e.g. 'Kepler')
    On error returns a JSON object with an 'error' key describing the failure.
    """
    try:
        print(f"[Space Scraper] Querying MAST archive for: {star_id}")

        # Try Kepler first, fall back to TESS SPOC
        search_result = lk.search_lightcurve(star_id, author="Kepler")
        mission = "Kepler"

        if len(search_result) == 0:
            print(f"[Space Scraper] No Kepler data — trying TESS for {star_id}")
            search_result = lk.search_lightcurve(star_id, author="SPOC")
            mission = "TESS"

        if len(search_result) == 0:
            return json.dumps({
                "error": f"No light curve data found for '{star_id}' in Kepler or TESS/SPOC archives."
            })

        # Download up to 8 quarters/sectors to keep runtime reasonable
        max_downloads = min(len(search_result), 8)
        print(f"[Space Scraper] Downloading {max_downloads} observation segments...")
        lc_collection = search_result[:max_downloads].download_all()

        if lc_collection is None or len(lc_collection) == 0:
            return json.dumps({"error": "Download returned empty collection from MAST."})

        # Stitch quarters into one continuous light curve
        lc_stitched = lc_collection.stitch()

        # Use PDCSAP flux (pre-conditioned, systematic-corrected) if available
        if "pdcsap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("pdcsap_flux")
        elif "sap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("sap_flux")
        else:
            lc_final = lc_stitched

        # Normalize: divide every flux value by the median so unit ≈ 1.0
        lc_final = lc_final.normalize()

        # Remove NaN cadences from inter-quarter gaps and cosmic rays
        lc_clean = lc_final.remove_nans()

        # Sort by time — lightkurve .stitch() can produce a handful of
        # backwards-stepping timestamps at Kepler quarter seam boundaries
        sort_idx    = np.argsort(lc_clean.time.value)
        time_values = lc_clean.time.value[sort_idx].tolist()
        flux_values = lc_clean.flux.value[sort_idx].tolist()

        print(f"[Space Scraper] Retrieved {len(time_values):,} cadences from {mission}.")

        return json.dumps({
            "star_id": star_id,
            "mission": mission,
            "records": len(time_values),
            "time":    time_values,
            "flux":    flux_values,
        })

    except Exception as exc:
        return json.dumps({
            "error": f"MAST fetch failed for '{star_id}': {type(exc).__name__}: {str(exc)}"
        })


# ══════════════════════════════════════════════════════════════════
#  TOOL 2: Photometric Signal Cleaning
# ══════════════════════════════════════════════════════════════════

@tool("Clean Signal")
def clean_signal_tool(lightcurve_json: str) -> str:
    """
    Accepts the JSON string produced by fetch_lightcurve_tool and applies
    two-stage photometric noise reduction:

    Stage 1 — Savitzky-Golay (SG) high-pass filter:
        Fits a polynomial through a rolling window to model slow stellar
        variability (star spots, pulsations). Divides this trend out so
        only short-timescale, periodic dips survive. Window is set to
        ~1% of total cadences (odd integer) to avoid smoothing out
        genuine transit dips which are typically <0.1% duration fraction.

    Stage 2 — Sigma-clipping:
        Removes flux outliers beyond ±3σ from the median. This discards
        cosmic ray hits, momentum dumps, and data glitches without
        touching the shallow, periodic dips caused by planetary transits.

    Returns a JSON string with the same structure as the input, updated
    with cleaned flux values, plus 'removed_outliers' count and
    'filter_window_size' metadata.
    """
    try:
        data = json.loads(lightcurve_json)

        if "error" in data:
            return lightcurve_json  # Propagate upstream errors unchanged

        print("[Signal Processor] Beginning photometric detrending...")

        time_arr = np.array(data["time"], dtype=np.float64)
        flux_arr = np.array(data["flux"], dtype=np.float64)
        n = len(flux_arr)

        # Stage 1: Savitzky-Golay trend removal
        # Window ~1% of series length; bitwise OR ensures odd; floor at 11
        window_len = max(int(n * 0.01) | 1, 11)
        polyorder  = 3  # cubic polynomial preserves dip shape well

        print(f"[Signal Processor] SG filter: window={window_len}, polyorder={polyorder}")
        trend          = savgol_filter(flux_arr, window_length=window_len, polyorder=polyorder)
        flux_detrended = flux_arr / trend  # result ≈ 1.0 ± transits

        # Stage 2: 3-sigma outlier clipping
        median_flux = np.median(flux_detrended)
        std_flux    = np.std(flux_detrended)
        mask_keep   = np.abs(flux_detrended - median_flux) <= 3.0 * std_flux

        removed    = int(np.sum(~mask_keep))
        time_clean = time_arr[mask_keep]
        flux_clean = flux_detrended[mask_keep]

        print(f"[Signal Processor] Removed {removed} outlier cadences ({removed/n*100:.2f}%).")
        print(f"[Signal Processor] Clean series: {len(time_clean):,} cadences remain.")

        return json.dumps({
            "star_id":           data.get("star_id", "unknown"),
            "mission":           data.get("mission", "unknown"),
            "records":           len(time_clean),
            "removed_outliers":  removed,
            "filter_window_size": window_len,
            "time":              time_clean.tolist(),
            "flux":              flux_clean.tolist(),
        })

    except Exception as exc:
        return json.dumps({
            "error": f"Signal cleaning failed: {type(exc).__name__}: {str(exc)}"
        })


# ══════════════════════════════════════════════════════════════════
#  TOOL 3: Box-fitting Least Squares Periodogram
# ══════════════════════════════════════════════════════════════════

@tool("BLS Periodogram")
def bls_periodogram_tool(clean_data_json: str) -> str:
    """
    Executes the Box-fitting Least Squares (BLS) algorithm on cleaned
    photometric data to detect periodic transit signals.

    The BLS algorithm (Kovács et al. 2002) folds the light curve at
    thousands of trial periods and fits a trapezoidal box model to find
    the period that maximizes the signal-to-noise of a transit dip.

    Algorithm steps:
        1. Define a logarithmically-spaced period grid from 0.5 days up to
           one-third of the total observation baseline (prevents aliasing).
        2. For each trial period, BLS computes the best-fitting transit
           duration and depth.
        3. The period with the peak BLS power is selected as the best
           candidate orbital period.
        4. Transit S/N proxy: BLS_power / median(BLS_power)
           Planet probability heuristic: P = 1 - exp(-SNR / 10)

    Returns a JSON string with:
        - 'period_days'      : best-fit orbital period (days)
        - 'transit_depth_ppm': fractional flux decrease during transit (ppm)
        - 'duration_days'    : best-fit transit duration (days)
        - 'bls_power'        : peak BLS power (dimensionless)
        - 'snr'              : signal-to-noise ratio of the detection
        - 'planet_probability': heuristic probability [0, 1]
        - 'detection_quality': 'Strong' | 'Moderate' | 'Weak' | 'Noise'
    """
    try:
        data = json.loads(clean_data_json)

        if "error" in data:
            return clean_data_json  # Propagate errors

        print("[Astrophysicist] Initializing Box-fitting Least Squares periodogram...")

        time_arr = np.array(data["time"], dtype=np.float64)
        flux_arr = np.array(data["flux"], dtype=np.float64)

        # Baseline sets the maximum testable period
        baseline_days = float(time_arr.max() - time_arr.min())
        max_period    = baseline_days / 3.0  # aliasing prevention
        min_period    = 0.5                  # sub-day orbits exist but are rare

        if max_period <= min_period:
            return json.dumps({
                "error": (
                    f"Observation baseline ({baseline_days:.1f} days) is too short "
                    "to robustly sample BLS periods. Need at least 1.5 days of data."
                )
            })

        # Attach astropy units required by BoxLeastSquares interface
        bls_model = BoxLeastSquares(time_arr * u.day, flux_arr * u.dimensionless_unscaled)

        # Log-spaced period grid: denser sampling at short periods where
        # hot Jupiters and super-Earths are most commonly found
        n_periods = 5000
        periods   = np.exp(
            np.linspace(np.log(min_period), np.log(max_period), n_periods)
        ) * u.day

        print(f"[Astrophysicist] Testing {n_periods} trial periods: "
              f"{min_period:.2f}–{max_period:.2f} days...")

        periodogram = bls_model.power(periods, duration=[0.05, 0.1, 0.15, 0.2] * u.day)

        # Extract best period
        best_idx    = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx].to(u.day).value)
        best_power  = float(periodogram.power[best_idx])

        # Compute transit depth and duration at the best period
        stats = bls_model.compute_stats(
            periodogram.period[best_idx],
            periodogram.duration[best_idx],
            periodogram.transit_time[best_idx],
        )
        transit_depth_frac = float(stats["depth"][0])
        transit_depth_ppm  = transit_depth_frac * 1_000_000
        duration_days      = float(periodogram.duration[best_idx].to(u.day).value)

        # SNR heuristic and planet probability
        median_power = float(np.median(periodogram.power))
        snr          = best_power / median_power if median_power > 0 else 0.0
        planet_prob  = float(1.0 - np.exp(-snr / 10.0))

        # Qualitative classification for the Science Communicator
        if snr >= 15:
            quality = "Strong"
        elif snr >= 7:
            quality = "Moderate"
        elif snr >= 3:
            quality = "Weak"
        else:
            quality = "Noise"

        print(f"[Astrophysicist] Best period: {best_period:.4f} days | "
              f"Depth: {transit_depth_ppm:.1f} ppm | SNR: {snr:.2f} | Quality: {quality}")

        return json.dumps({
            "star_id":            data.get("star_id", "unknown"),
            "mission":            data.get("mission", "unknown"),
            "period_days":        round(best_period, 6),
            "transit_depth_ppm":  round(transit_depth_ppm, 2),
            "duration_days":      round(duration_days, 6),
            "bls_power":          round(best_power, 6),
            "snr":                round(snr, 4),
            "planet_probability": round(planet_prob, 4),
            "detection_quality":  quality,
        })

    except Exception as exc:
        return json.dumps({
            "error": f"BLS periodogram failed: {type(exc).__name__}: {str(exc)}"
        })
