"""
visualize.py — Exoplanet Swarm (Plotly Edition)
────────────────────────────────────────────────
Interactive Plotly figures for the 4-panel diagnostic view.
Each function returns a go.Figure that can be embedded in Streamlit
via st.plotly_chart() or exported as HTML.

Usage (command line):
    python visualize.py                  # Kepler-186, pulls from MAST or CSV cache
    python visualize.py --cached         # Use tests/fixtures/ cache
    python visualize.py "TOI 700"
"""

import sys
import os
import json
import warnings
import argparse
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools import fetch_lightcurve_tool, clean_signal_tool, bls_periodogram_tool
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

# ── Resolve underlying callables (bypass @tool wrapper) ───────────
fetch_tool = fetch_lightcurve_tool.func if hasattr(fetch_lightcurve_tool, "func") else fetch_lightcurve_tool
clean_tool = clean_signal_tool.func     if hasattr(clean_signal_tool, "func")     else clean_signal_tool
bls_tool   = bls_periodogram_tool.func  if hasattr(bls_periodogram_tool, "func")  else bls_periodogram_tool

# ── Dark palette ──────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "purple":  "#d2a8ff",
    "red":     "#f78166",
    "orange":  "#ffa657",
    "white":   "#ffffff",
}

_LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor= PALETTE["panel"],
    font=dict(color=PALETTE["text"], family="Inter, DejaVu Sans, sans-serif"),
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis=dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False,
               tickfont=dict(color=PALETTE["subtext"])),
    yaxis=dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False,
               tickfont=dict(color=PALETTE["subtext"])),
)


# ══════════════════════════════════════════════════════════════════
#  Figure builders — each returns a standalone go.Figure
# ══════════════════════════════════════════════════════════════════

def make_raw_lc_figure(raw_data: dict) -> go.Figure:
    """Panel 1: Raw normalized light curve."""
    t = np.array(raw_data["time"])
    f = np.array(raw_data["flux"])
    step = max(1, len(t) // 20_000)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t[::step], y=f[::step],
        mode="markers",
        marker=dict(color=PALETTE["blue"], size=1.5, opacity=0.5),
        name="Normalized flux",
        hovertemplate="BJD: %{x:.2f}<br>Flux: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=f"Raw Normalized Light Curve  ·  {raw_data['records']:,} cadences",
                   font=dict(size=14)),
        xaxis_title="Time (BJD)",
        yaxis_title="Normalized Flux",
        height=380,
    )
    return fig


def make_clean_lc_figure(clean_data: dict) -> go.Figure:
    """Panel 2: Detrended and cleaned light curve."""
    t = np.array(clean_data["time"])
    f = np.array(clean_data["flux"])
    step = max(1, len(t) // 20_000)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t[::step], y=f[::step],
        mode="markers",
        marker=dict(color=PALETTE["green"], size=1.5, opacity=0.5),
        name="Cleaned flux",
        hovertemplate="BJD: %{x:.2f}<br>Flux: %{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot",
                  line_color=PALETTE["subtext"], opacity=0.6)
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(f"Detrended & Cleaned  ·  "
                  f"{clean_data.get('removed_outliers', 0):,} outliers removed"),
            font=dict(size=14),
        ),
        xaxis_title="Time (BJD)",
        yaxis_title="Normalized Flux",
        height=380,
    )
    return fig


def make_bls_figure(clean_data: dict, bls_data: dict,
                    n_periods: int = 2000) -> go.Figure:
    """
    Panel 3: BLS power spectrum with red vertical line at best period.
    Recomputes a quick periodogram (n_periods=2000 for speed) from clean data.
    """
    t = np.array(clean_data["time"], dtype=np.float64)
    f = np.array(clean_data["flux"], dtype=np.float64)
    best_period = float(bls_data["orbital_period_days"])
    quality     = bls_data.get("detection_quality", "?")
    snr         = bls_data.get("snr", 0)

    baseline  = float(t.max() - t.min())
    max_period = min(baseline / 3.0, 200.0)

    periods_arr = np.exp(
        np.linspace(np.log(0.5), np.log(max_period), n_periods)
    )
    bls_model   = BoxLeastSquares(t * u.day, f * u.dimensionless_unscaled)
    periodogram = bls_model.power(
        periods_arr * u.day, duration=[0.05, 0.1, 0.2] * u.day
    )
    pv = periodogram.period.value
    pw = periodogram.power.value

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pv, y=pw,
        mode="lines",
        line=dict(color=PALETTE["purple"], width=0.8),
        name="BLS power",
        hovertemplate="Period: %{x:.4f} d<br>Power: %{y:.6f}<extra></extra>",
    ))
    fig.add_vline(
        x=best_period,
        line_color=PALETTE["red"], line_dash="dash", line_width=2,
        annotation_text=f"  {best_period:.4f} d  ({quality}, SNR {snr:.1f})",
        annotation_font_color=PALETTE["red"],
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="BLS Power Spectrum", font=dict(size=14)),
        xaxis_title="Trial Period (days, log scale)",
        yaxis_title="BLS Power",
        xaxis_type="log",
        height=380,
    )
    return fig


def make_phase_fold_figure(clean_data: dict, bls_data: dict) -> go.Figure:
    """Panel 4: Phase-folded light curve at best-fit period."""
    t = np.array(clean_data["time"], dtype=np.float64)
    f = np.array(clean_data["flux"], dtype=np.float64)
    best_period = float(bls_data["orbital_period_days"])
    depth_ppm   = bls_data.get("transit_depth_ppm", 0)

    phase   = (t % best_period) / best_period
    idx     = np.argsort(phase)
    phase_s = phase[idx]
    flux_s  = f[idx]

    # Binned median overlay
    n_bins  = 200
    bins    = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    binned  = np.array([
        np.median(flux_s[(phase_s >= bins[i]) & (phase_s < bins[i + 1])])
        if np.any((phase_s >= bins[i]) & (phase_s < bins[i + 1]))
        else float("nan")
        for i in range(n_bins)
    ])

    step = max(1, len(phase_s) // 20_000)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=phase_s[::step], y=flux_s[::step],
        mode="markers",
        marker=dict(color=PALETTE["orange"], size=1.5, opacity=0.2),
        name="Individual cadences",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=binned,
        mode="lines",
        line=dict(color=PALETTE["white"], width=2),
        name="Binned median",
        hovertemplate="Phase: %{x:.3f}<br>Flux: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"Phase-Folded at {best_period:.4f} days  ·  Depth {depth_ppm:.0f} ppm",
            font=dict(size=14),
        ),
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


def make_combined_figure(raw_data: dict, clean_data: dict, bls_data: dict) -> go.Figure:
    """
    4-panel combined figure (2×2 grid) — for command-line / HTML export.
    """
    from plotly.subplots import make_subplots as _ms
    fig = _ms(rows=2, cols=2, subplot_titles=[
        "Raw Normalized Light Curve",
        "Detrended & Cleaned Light Curve",
        "BLS Power Spectrum",
        f"Phase-Folded at {bls_data['orbital_period_days']:.4f} d",
    ])

    for src, dest_row, dest_col in [
        (make_raw_lc_figure(raw_data),            1, 1),
        (make_clean_lc_figure(clean_data),         1, 2),
    ]:
        for trace in src.data:
            fig.add_trace(trace, row=dest_row, col=dest_col)

    bls_fig = make_bls_figure(clean_data, bls_data)
    for trace in bls_fig.data:
        fig.add_trace(trace, row=2, col=1)
    fig.update_xaxes(type="log", row=2, col=1)

    pf_fig = make_phase_fold_figure(clean_data, bls_data)
    for trace in pf_fig.data:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor= PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        height=800,
        showlegend=False,
        title=dict(
            text=f"Exoplanet Swarm  ·  {raw_data.get('star_id','?')}  ·  {raw_data.get('mission','?')} Mission",
            font=dict(size=16),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

def run(star_id: str, use_cache: bool = False, output_html: str = None):
    FIXTURE_DIR = os.path.join("tests", "fixtures")

    def _load_or_run(cache_path, fn, *args):
        if use_cache and os.path.exists(cache_path):
            print(f"[viz] Loading cache: {cache_path}")
            with open(cache_path) as fp:
                return json.load(fp)
        result = json.loads(fn(*args))
        with open(cache_path, "w") as fp:
            json.dump(result, fp)
        return result

    raw_data   = _load_or_run(os.path.join(FIXTURE_DIR, "kepler186_raw.json"),
                               fetch_tool, star_id)
    clean_data = _load_or_run(os.path.join(FIXTURE_DIR, "kepler186_clean.json"),
                               clean_tool, json.dumps(raw_data))
    bls_data   = _load_or_run(os.path.join(FIXTURE_DIR, "kepler186_bls.json"),
                               bls_tool,   json.dumps(clean_data))

    if "error" in raw_data or "error" in clean_data or "error" in bls_data:
        print("[viz] ERROR in pipeline:", raw_data.get("error") or clean_data.get("error") or bls_data.get("error"))
        return

    print(f"[viz] Best period: {bls_data['orbital_period_days']:.4f} d | "
          f"Depth: {bls_data['transit_depth_ppm']:.1f} ppm | SNR: {bls_data['snr']:.2f}")

    fig = make_combined_figure(raw_data, clean_data, bls_data)

    if output_html is None:
        safe = star_id.replace(" ", "").replace("-", "").lower()
        output_html = f"{safe}_transit_analysis.html"

    fig.write_html(output_html)
    print(f"[viz] Saved interactive HTML → {output_html}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Plotly visualization for Exoplanet Swarm.")
    parser.add_argument("star_id", nargs="?", default="Kepler-186")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.star_id, use_cache=args.cached, output_html=args.output)
