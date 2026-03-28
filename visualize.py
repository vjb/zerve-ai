"""
visualize.py
────────────
Standalone visualization script for the Exoplanet Swarm pipeline.

Fetches real Kepler-186 data from NASA MAST via lightkurve, runs the
cleaning and BLS tools, then renders a 4-panel diagnostic figure:

  Panel 1 — Raw normalized light curve
  Panel 2 — Detrended / cleaned light curve
  Panel 3 — BLS power spectrum (period vs power, peak highlighted)
  Panel 4 — Phase-folded light curve at best-fit period

Usage:
    python visualize.py                   # uses Kepler-186 (default)
    python visualize.py "TOI 700"         # any star name lightkurve can find
    python visualize.py --cached          # skip MAST, use tests/fixtures cache

Output:
    kepler186_transit_analysis.png  (saved to current directory)
"""

import sys
import os
import json
import warnings
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe backend for Zerve.ai
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# ── Import tools ───────────────────────────────────────────────────
from tools import fetch_lightcurve_tool, clean_signal_tool, bls_periodogram_tool

fetch_tool = fetch_lightcurve_tool.func if hasattr(fetch_lightcurve_tool, "func") else fetch_lightcurve_tool
clean_tool = clean_signal_tool.func     if hasattr(clean_signal_tool, "func")     else clean_signal_tool
bls_tool   = bls_periodogram_tool.func  if hasattr(bls_periodogram_tool, "func")  else bls_periodogram_tool

# ── Also import astropy BLS directly for the power spectrum plot ──
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

# ── Design tokens ─────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "raw":       "#58a6ff",
    "clean":     "#3fb950",
    "power":     "#d2a8ff",
    "peak":      "#f78166",
    "fold":      "#ffa657",
    "fold_bin":  "#ffffff",
    "grid":      "#21262d",
}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "legend.framealpha": 0.15,
        "legend.edgecolor":  PALETTE["border"],
    })


def _phase_fold(time, flux, period):
    """Fold time array to phase [0, 1) at given period."""
    phase = (time % period) / period
    idx = np.argsort(phase)
    return phase[idx], flux[idx]


def _bin_folded(phase, flux, n_bins=150):
    """Bin a phase-folded light curve for a clean overlay."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.sum() > 0:
            binned[i] = np.median(flux[mask])
    return centers, binned


def run(star_id: str, use_cache: bool = False, output_path: str = None):
    """
    Full pipeline + 4-panel plot for the given star.

    Args:
        star_id    : Any name recognized by lightkurve (e.g. 'Kepler-186')
        use_cache  : If True, load from tests/fixtures/ instead of MAST
        output_path: Where to save the PNG. Defaults to <star_id>_analysis.png
    """
    _setup_style()

    FIXTURE_DIR = os.path.join("tests", "fixtures")
    safe_name   = star_id.replace(" ", "").replace("-", "").lower()

    # ── 1. Fetch ───────────────────────────────────────────────────
    raw_cache = os.path.join(FIXTURE_DIR, "kepler186_raw.json")
    if use_cache and os.path.exists(raw_cache):
        print(f"[viz] Loading raw data from cache: {raw_cache}")
        with open(raw_cache) as f:
            raw_data = json.load(f)
    else:
        print(f"[viz] Fetching {star_id} from NASA MAST via lightkurve...")
        raw_json = fetch_tool(star_id)
        raw_data = json.loads(raw_json)
        if "error" in raw_data:
            print(f"[viz] ERROR: {raw_data['error']}")
            sys.exit(1)

    print(f"[viz] {raw_data['records']:,} cadences from {raw_data['mission']}")

    # ── 2. Clean ───────────────────────────────────────────────────
    clean_cache = os.path.join(FIXTURE_DIR, "kepler186_clean.json")
    if use_cache and os.path.exists(clean_cache):
        print(f"[viz] Loading clean data from cache: {clean_cache}")
        with open(clean_cache) as f:
            clean_data = json.load(f)
    else:
        print("[viz] Cleaning signal...")
        clean_json = clean_tool(json.dumps(raw_data))
        clean_data = json.loads(clean_json)
        if "error" in clean_data:
            print(f"[viz] ERROR: {clean_data['error']}")
            sys.exit(1)

    # ── 3. BLS ────────────────────────────────────────────────────
    bls_cache = os.path.join(FIXTURE_DIR, "kepler186_bls.json")
    if use_cache and os.path.exists(bls_cache):
        print(f"[viz] Loading BLS result from cache: {bls_cache}")
        with open(bls_cache) as f:
            bls_data = json.load(f)
    else:
        print("[viz] Running BLS periodogram...")
        bls_json = bls_tool(json.dumps(clean_data))
        bls_data = json.loads(bls_json)
        if "error" in bls_data:
            print(f"[viz] ERROR: {bls_data['error']}")
            sys.exit(1)

    best_period = bls_data["period_days"]
    depth_ppm   = bls_data["transit_depth_ppm"]
    snr         = bls_data["snr"]
    prob        = bls_data["planet_probability"]
    quality     = bls_data["detection_quality"]
    mission     = raw_data["mission"]

    print(f"[viz] Best period: {best_period:.4f} days | Depth: {depth_ppm:.1f} ppm | "
          f"SNR: {snr:.2f} | Quality: {quality}")

    # ── 4. Re-run BLS to get full power spectrum for plotting ─────
    print("[viz] Computing full BLS power spectrum for plot...")
    t = np.array(clean_data["time"])
    f = np.array(clean_data["flux"])
    baseline = float(t.max() - t.min())
    periods  = np.exp(np.linspace(np.log(0.5), np.log(baseline / 3), 3000)) * u.day
    bls_model = BoxLeastSquares(t * u.day, f * u.dimensionless_unscaled)
    periodogram = bls_model.power(periods, duration=[0.05, 0.1, 0.15, 0.2] * u.day)
    period_vals = periodogram.period.value
    power_vals  = periodogram.power.value

    # ── 5. Layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Exoplanet Swarm  ·  {star_id}  ·  {mission} Mission",
        fontsize=18, fontweight="bold", color=PALETTE["text"],
        y=0.97,
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.40,
        wspace=0.28,
        left=0.07, right=0.97,
        top=0.92, bottom=0.08,
    )
    ax_raw   = fig.add_subplot(gs[0, 0])
    ax_clean = fig.add_subplot(gs[0, 1])
    ax_bls   = fig.add_subplot(gs[1, 0])
    ax_fold  = fig.add_subplot(gs[1, 1])

    def _style_ax(ax):
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

    # ── Panel 1: Raw light curve ───────────────────────────────────
    t_raw = np.array(raw_data["time"])
    f_raw = np.array(raw_data["flux"])

    # Subsample for speed if very large
    step = max(1, len(t_raw) // 20_000)
    ax_raw.scatter(t_raw[::step], f_raw[::step],
                   s=0.3, color=PALETTE["raw"], alpha=0.5, rasterized=True)
    ax_raw.set_title("Raw Normalized Light Curve")
    ax_raw.set_xlabel(f"Time (BJD)  ·  {raw_data['records']:,} cadences")
    ax_raw.set_ylabel("Normalized Flux")
    _style_ax(ax_raw)

    # ── Panel 2: Cleaned light curve ──────────────────────────────
    step2 = max(1, len(t) // 20_000)
    ax_clean.scatter(t[::step2], f[::step2],
                     s=0.3, color=PALETTE["clean"], alpha=0.5, rasterized=True)
    ax_clean.set_title("Detrended & Cleaned Light Curve")
    ax_clean.set_xlabel(
        f"Time (BJD)  ·  {clean_data['removed_outliers']:,} outliers removed"
    )
    ax_clean.set_ylabel("Normalized Flux")
    _style_ax(ax_clean)

    # ── Panel 3: BLS periodogram ───────────────────────────────────
    ax_bls.plot(period_vals, power_vals,
                color=PALETTE["power"], lw=0.7, alpha=0.85)
    ax_bls.axvline(best_period, color=PALETTE["peak"], lw=1.8,
                   linestyle="--", label=f"Best period: {best_period:.4f} d")

    # Annotate known Kepler-186 periods if target matches
    known = {"b": 3.887, "c": 7.267, "d": 13.342, "e": 22.408, "f": 129.945}
    if "186" in star_id:
        ymax = power_vals.max()
        for letter, p in known.items():
            if p < period_vals.max():
                ax_bls.axvline(p, color=PALETTE["fold_bin"], lw=0.6,
                               linestyle=":", alpha=0.5)
                ax_bls.text(p, ymax * 0.92, f" {letter}", fontsize=7,
                            color=PALETTE["subtext"], va="top")

    ax_bls.set_xscale("log")
    ax_bls.set_title("BLS Periodogram")
    ax_bls.set_xlabel("Trial Period (days, log scale)")
    ax_bls.set_ylabel("BLS Power")
    ax_bls.legend(fontsize=9, loc="upper left",
                  facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    _style_ax(ax_bls)

    # ── Panel 4: Phase-folded light curve ─────────────────────────
    phase, f_sorted = _phase_fold(t, f, best_period)
    # Scatter (raw points)
    ax_fold.scatter(phase, f_sorted, s=0.5, color=PALETTE["fold"],
                    alpha=0.25, rasterized=True, label="Individual cadences")
    # Binned overlay
    bin_phase, bin_flux = _bin_folded(phase, f_sorted, n_bins=200)
    ax_fold.plot(bin_phase, bin_flux, color=PALETTE["fold_bin"],
                 lw=1.8, label="Binned median", zorder=5)

    ax_fold.set_title(f"Phase-Folded at {best_period:.4f} days")
    ax_fold.set_xlabel("Orbital Phase")
    ax_fold.set_ylabel("Normalized Flux")
    ax_fold.legend(fontsize=9, loc="upper right",
                   facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                   markerscale=6)
    _style_ax(ax_fold)

    # ── Stats annotation block ─────────────────────────────────────
    stats_text = (
        f"Period:  {best_period:.4f} d\n"
        f"Depth:   {depth_ppm:.0f} ppm\n"
        f"SNR:     {snr:.2f}\n"
        f"P(planet): {prob:.1%}\n"
        f"Quality: {quality}"
    )
    fig.text(
        0.985, 0.50, stats_text,
        fontsize=9.5, va="center", ha="right",
        color=PALETTE["text"],
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["panel"],
                  edgecolor=PALETTE["border"], alpha=0.9),
    )

    # ── Save ──────────────────────────────────────────────────────
    if output_path is None:
        output_path = f"{safe_name}_transit_analysis.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"[viz] Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize exoplanet transit analysis for any MAST target."
    )
    parser.add_argument("star_id", nargs="?", default="Kepler-186",
                        help="Star identifier (default: Kepler-186)")
    parser.add_argument("--cached", action="store_true",
                        help="Use cached fixtures from tests/fixtures/ instead of MAST")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <star>_transit_analysis.png)")
    args = parser.parse_args()

    run(args.star_id, use_cache=args.cached, output_path=args.output)
