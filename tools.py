"""
tools.py — Exoplanet Swarm
──────────────────────────
All @tool-decorated functions used by the CrewAI agents.

  fetch_lightcurve_tool   — NASA MAST data retrieval (demo-cache aware)
  clean_signal_tool       — Savitzky-Golay + sigma-clip + signal_plot.png
  bls_periodogram_tool    — BLS periodogram + bls_plot.png + PlanetMetrics

No agent or LLM references in this file — pure astronomy + data science.
"""

import json
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Headless matplotlib (safe on Zerve.ai / CI) ───────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Pydantic schema for strict structured output ──────────────────
from pydantic import BaseModel

# ── CrewAI / astronomy / signal ───────────────────────────────────
from crewai.tools import tool
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════
#  PYDANTIC OUTPUT SCHEMA
#  bls_periodogram_tool always returns data conforming to this model.
#  The Science Communicator receives the serialized JSON and is
#  strictly prompted never to alter the raw float values.
# ══════════════════════════════════════════════════════════════════

class PlanetMetrics(BaseModel):
    """Strict structured output for BLS transit detection results."""
    star_id:              str
    mission:              str
    orbital_period_days:  float   # exact best-fit orbital period
    transit_depth_ppm:    float   # exact transit depth in parts-per-million
    transit_duration_days: float  # exact transit duration
    planet_probability:   float   # heuristic [0, 1]
    snr:                  float   # signal-to-noise ratio
    detection_quality:    str     # 'Strong' | 'Moderate' | 'Weak' | 'Noise'
    planet_detected:      bool    # True if SNR >= 7 (Moderate or better)


# ══════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════

_DARK = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "border": "#30363d",
    "text":   "#e6edf3",
    "sub":    "#8b949e",
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "purple": "#d2a8ff",
    "red":    "#f78166",
    "orange": "#ffa657",
}

def _apply_dark(ax):
    ax.set_facecolor(_DARK["panel"])
    ax.tick_params(colors=_DARK["sub"])
    ax.xaxis.label.set_color(_DARK["text"])
    ax.yaxis.label.set_color(_DARK["text"])
    ax.title.set_color(_DARK["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_DARK["border"])
    ax.grid(True, alpha=0.3, color=_DARK["border"])


# ══════════════════════════════════════════════════════════════════
#  TOOL 1: NASA MAST Data Retrieval  (demo-cache aware)
# ══════════════════════════════════════════════════════════════════

DEMO_CACHE_PATH = "demo_kepler186_data.csv"
DEMO_STAR       = "Kepler-186"

@tool("Fetch Lightcurve")
def fetch_lightcurve_tool(star_id: str) -> str:
    """
    Retrieves photometric light curve data for a given star.

    DEMO FAST-PATH: If star_id is exactly 'Kepler-186', loads the
    pre-cached lightweight CSV (demo_kepler186_data.csv) instead of
    hitting the NASA MAST API — zero latency, zero download risk.

    LIVE PATH (all other stars): Downloads a single quarter of
    Kepler or TESS data (prevents memory overflow on large targets).

    Returns JSON: { star_id, mission, records, time[], flux[] }
    Returns JSON with 'error' key on failure.
    """
    try:
        # ── Demo fast-path ────────────────────────────────────────
        if star_id.strip() == DEMO_STAR and os.path.exists(DEMO_CACHE_PATH):
            print(f"[Space Scraper] ⚡ Demo cache hit: loading {DEMO_CACHE_PATH}")
            import pandas as pd
            df = pd.read_csv(DEMO_CACHE_PATH)
            time_vals = df["time"].tolist()
            flux_vals = df["flux"].tolist()
            print(f"[Space Scraper] Loaded {len(time_vals):,} cadences from CSV cache.")
            return json.dumps({
                "star_id": star_id,
                "mission": "Kepler",
                "records": len(time_vals),
                "time":    time_vals,
                "flux":    flux_vals,
            })

        # ── Live MAST path ────────────────────────────────────────
        print(f"[Space Scraper] Querying MAST archive for: {star_id}")
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

        # Limit to ONE quarter to prevent memory overflow on Zerve containers
        print(f"[Space Scraper] Downloading 1 observation segment (memory-safe mode)...")
        lc_collection = search_result[:1].download_all()

        if lc_collection is None or len(lc_collection) == 0:
            return json.dumps({"error": "Download returned empty collection from MAST."})

        lc_stitched = lc_collection.stitch()

        if "pdcsap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("pdcsap_flux")
        elif "sap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("sap_flux")
        else:
            lc_final = lc_stitched

        lc_final = lc_final.normalize()
        lc_clean = lc_final.remove_nans()

        # Sort to eliminate inter-quarter timestamp seam artifacts
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
#  TOOL 2: Photometric Signal Cleaning  (+visual paper trail)
# ══════════════════════════════════════════════════════════════════

@tool("Clean Signal")
def clean_signal_tool(lightcurve_json: str) -> str:
    """
    Applies two-stage photometric noise reduction and saves signal_plot.png.

    Stage 1 — Savitzky-Golay high-pass filter:
        Window ~1% of cadences (always odd) removes slow stellar variability
        while preserving short transit dips.

    Stage 2 — 3σ sigma-clipping:
        Removes cosmic ray hits, momentum dumps, and data glitches.

    Saves signal_plot.png to working directory showing:
        - Raw flux (blue)
        - SG trend model (orange overlay)
        - Detrended result (green)

    Returns JSON: { star_id, mission, records, removed_outliers,
                    filter_window_size, time[], flux[] }
    """
    try:
        data = json.loads(lightcurve_json)

        if "error" in data:
            return lightcurve_json

        print("[Signal Processor] Beginning photometric detrending...")

        time_arr = np.array(data["time"], dtype=np.float64)
        flux_arr = np.array(data["flux"], dtype=np.float64)
        n = len(flux_arr)

        # Stage 1: SG trend removal
        window_len = max(int(n * 0.01) | 1, 11)
        polyorder  = 3
        print(f"[Signal Processor] SG filter: window={window_len}, polyorder={polyorder}")
        trend          = savgol_filter(flux_arr, window_length=window_len, polyorder=polyorder)
        flux_detrended = flux_arr / trend

        # Stage 2: sigma-clipping
        median_flux = np.median(flux_detrended)
        std_flux    = np.std(flux_detrended)
        mask_keep   = np.abs(flux_detrended - median_flux) <= 3.0 * std_flux

        removed    = int(np.sum(~mask_keep))
        time_clean = time_arr[mask_keep]
        flux_clean = flux_detrended[mask_keep]

        print(f"[Signal Processor] Removed {removed} outlier cadences ({removed/n*100:.2f}%).")
        print(f"[Signal Processor] Clean series: {len(time_clean):,} cadences remain.")

        # ── Visual paper trail: signal_plot.png ───────────────────
        _save_signal_plot(time_arr, flux_arr, trend, time_clean, flux_clean,
                          star_id=data.get("star_id", "unknown"))

        return json.dumps({
            "star_id":            data.get("star_id", "unknown"),
            "mission":            data.get("mission", "unknown"),
            "records":            len(time_clean),
            "removed_outliers":   removed,
            "filter_window_size": window_len,
            "time":               time_clean.tolist(),
            "flux":               flux_clean.tolist(),
        })

    except Exception as exc:
        return json.dumps({
            "error": f"Signal cleaning failed: {type(exc).__name__}: {str(exc)}"
        })


def _save_signal_plot(time_raw, flux_raw, trend, time_clean, flux_clean, star_id):
    """Save a 3-panel signal cleaning diagnostic to signal_plot.png."""
    try:
        # Subsample for speed
        step = max(1, len(time_raw) // 15_000)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 facecolor=_DARK["bg"], sharex=False)
        fig.suptitle(f"Signal Cleaning  ·  {star_id}",
                     color=_DARK["text"], fontsize=14, fontweight="bold", y=0.98)

        # Panel 1: Raw flux
        axes[0].scatter(time_raw[::step], flux_raw[::step],
                        s=0.4, color=_DARK["blue"], alpha=0.5, rasterized=True)
        axes[0].plot(time_raw[::step], trend[::step],
                     color=_DARK["orange"], lw=1.2, label="SG trend model", alpha=0.85)
        axes[0].set_title("Raw Flux + Savitzky-Golay Trend")
        axes[0].set_ylabel("Normalized Flux")
        axes[0].legend(fontsize=8, facecolor=_DARK["panel"], labelcolor=_DARK["text"])

        # Panel 2: Detrended flux (all points)
        flux_det = flux_raw / trend
        axes[1].scatter(time_raw[::step], flux_det[::step],
                        s=0.4, color=_DARK["purple"], alpha=0.5, rasterized=True)
        axes[1].axhline(1.0, color=_DARK["sub"], lw=0.8, linestyle="--")
        axes[1].set_title("Detrended Flux (before sigma-clip)")
        axes[1].set_ylabel("Normalized Flux")

        # Panel 3: Cleaned flux
        step2 = max(1, len(time_clean) // 15_000)
        axes[2].scatter(time_clean[::step2], flux_clean[::step2],
                        s=0.4, color=_DARK["green"], alpha=0.5, rasterized=True)
        axes[2].axhline(1.0, color=_DARK["sub"], lw=0.8, linestyle="--")
        axes[2].set_title(f"Cleaned Flux  ·  {len(time_clean):,} cadences retained")
        axes[2].set_ylabel("Normalized Flux")
        axes[2].set_xlabel("Time (BJD)")

        for ax in axes:
            _apply_dark(ax)

        plt.tight_layout()
        fig.savefig("signal_plot.png", dpi=130, bbox_inches="tight",
                    facecolor=_DARK["bg"])
        plt.close(fig)
        print("[Signal Processor] 📊 Saved → signal_plot.png")

    except Exception as e:
        print(f"[Signal Processor] Warning: could not save signal_plot.png: {e}")


# ══════════════════════════════════════════════════════════════════
#  TOOL 3: BLS Periodogram  (+visual paper trail +PlanetMetrics)
# ══════════════════════════════════════════════════════════════════

@tool("BLS Periodogram")
def bls_periodogram_tool(clean_data_json: str) -> str:
    """
    Executes BLS transit search and saves bls_plot.png.

    Returns a PlanetMetrics JSON with exact float values:
        orbital_period_days, transit_depth_ppm, transit_duration_days,
        planet_probability, snr, detection_quality, planet_detected.

    All floats are exact scientific measurements — never rounded
    or paraphrased downstream.

    Saves bls_plot.png to working directory showing:
        - BLS power spectrum (log x-axis)
        - Red vertical line at detected orbital period
        - Phase-folded light curve at best period
    """
    try:
        data = json.loads(clean_data_json)

        if "error" in data:
            return clean_data_json

        print("[Astrophysicist] Initializing Box-fitting Least Squares periodogram...")

        time_arr = np.array(data["time"], dtype=np.float64)
        flux_arr = np.array(data["flux"], dtype=np.float64)

        baseline_days = float(time_arr.max() - time_arr.min())
        max_period    = baseline_days / 3.0
        min_period    = 0.5

        if max_period <= min_period:
            return json.dumps({
                "error": (
                    f"Observation baseline ({baseline_days:.1f} days) too short "
                    "for BLS. Need at least 1.5 days of data."
                )
            })

        bls_model = BoxLeastSquares(time_arr * u.day, flux_arr * u.dimensionless_unscaled)

        n_periods = 5000
        periods   = np.exp(
            np.linspace(np.log(min_period), np.log(max_period), n_periods)
        ) * u.day

        print(f"[Astrophysicist] Testing {n_periods} trial periods: "
              f"{min_period:.2f}–{max_period:.2f} days...")

        periodogram = bls_model.power(periods, duration=[0.05, 0.1, 0.15, 0.2] * u.day)

        best_idx    = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx].to(u.day).value)
        best_power  = float(periodogram.power[best_idx])

        stats = bls_model.compute_stats(
            periodogram.period[best_idx],
            periodogram.duration[best_idx],
            periodogram.transit_time[best_idx],
        )
        transit_depth_ppm  = float(stats["depth"][0]) * 1_000_000
        duration_days      = float(periodogram.duration[best_idx].to(u.day).value)

        median_power = float(np.median(periodogram.power))
        snr          = best_power / median_power if median_power > 0 else 0.0
        planet_prob  = float(1.0 - np.exp(-snr / 10.0))

        if snr >= 15:   quality = "Strong"
        elif snr >= 7:  quality = "Moderate"
        elif snr >= 3:  quality = "Weak"
        else:           quality = "Noise"

        planet_detected = snr >= 7  # Moderate or better

        print(f"[Astrophysicist] Best period: {best_period:.4f} days | "
              f"Depth: {transit_depth_ppm:.1f} ppm | SNR: {snr:.2f} | "
              f"Quality: {quality} | Detected: {planet_detected}")

        # ── Build PlanetMetrics (strict schema) ───────────────────
        metrics = PlanetMetrics(
            star_id=              data.get("star_id", "unknown"),
            mission=              data.get("mission", "unknown"),
            orbital_period_days=  round(best_period, 6),
            transit_depth_ppm=    round(transit_depth_ppm, 2),
            transit_duration_days= round(duration_days, 6),
            planet_probability=   round(planet_prob, 4),
            snr=                  round(snr, 4),
            detection_quality=    quality,
            planet_detected=      planet_detected,
        )

        # ── Visual paper trail: bls_plot.png ──────────────────────
        _save_bls_plot(
            period_vals=periodogram.period.value,
            power_vals=periodogram.power.value,
            best_period=best_period,
            time_arr=time_arr,
            flux_arr=flux_arr,
            star_id=data.get("star_id", "unknown"),
        )

        return metrics.model_dump_json()

    except Exception as exc:
        return json.dumps({
            "error": f"BLS periodogram failed: {type(exc).__name__}: {str(exc)}"
        })


def _save_bls_plot(period_vals, power_vals, best_period, time_arr, flux_arr, star_id):
    """Save a 2-panel BLS diagnostic to bls_plot.png."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                 facecolor=_DARK["bg"])
        fig.suptitle(f"BLS Transit Detection  ·  {star_id}",
                     color=_DARK["text"], fontsize=14, fontweight="bold", y=1.01)

        # Panel 1: Power spectrum
        ax1 = axes[0]
        ax1.plot(period_vals, power_vals, color=_DARK["purple"], lw=0.7, alpha=0.9)
        ax1.axvline(best_period, color=_DARK["red"], lw=2.0, linestyle="--",
                    label=f"Best period: {best_period:.4f} d")
        ax1.set_xscale("log")
        ax1.set_title("BLS Power Spectrum")
        ax1.set_xlabel("Trial Period (days, log scale)")
        ax1.set_ylabel("BLS Power")
        ax1.legend(fontsize=9, facecolor=_DARK["panel"], labelcolor=_DARK["text"])
        _apply_dark(ax1)

        # Panel 2: Phase-folded light curve
        ax2 = axes[1]
        phase = (time_arr % best_period) / best_period
        idx   = np.argsort(phase)
        phase_s = phase[idx]
        flux_s  = flux_arr[idx]

        # Bin for clean overlay
        n_bins  = 200
        bins    = np.linspace(0, 1, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        binned  = np.array([
            np.median(flux_s[(phase_s >= bins[i]) & (phase_s < bins[i+1])])
            if np.any((phase_s >= bins[i]) & (phase_s < bins[i+1])) else np.nan
            for i in range(n_bins)
        ])

        step = max(1, len(phase_s) // 20_000)
        ax2.scatter(phase_s[::step], flux_s[::step],
                    s=0.5, color=_DARK["orange"], alpha=0.2, rasterized=True)
        ax2.plot(centers, binned, color="white", lw=1.8, zorder=5, label="Binned median")
        ax2.axvline(0.0, color=_DARK["red"], lw=1.0, linestyle=":", alpha=0.6)
        ax2.set_title(f"Phase-Folded at {best_period:.4f} days")
        ax2.set_xlabel("Orbital Phase")
        ax2.set_ylabel("Normalized Flux")
        ax2.legend(fontsize=9, facecolor=_DARK["panel"], labelcolor=_DARK["text"],
                   markerscale=6)
        _apply_dark(ax2)

        plt.tight_layout()
        fig.savefig("bls_plot.png", dpi=130, bbox_inches="tight",
                    facecolor=_DARK["bg"])
        plt.close(fig)
        print("[Astrophysicist] 📊 Saved → bls_plot.png")

    except Exception as e:
        print(f"[Astrophysicist] Warning: could not save bls_plot.png: {e}")
