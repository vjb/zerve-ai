"""
streamlit_app.py — Exoplanet Swarm
────────────────────────────────────
Interactive Streamlit frontend for the Exoplanet Swarm pipeline.

DEPLOYMENT MODES
────────────────
  Zerve Hosted App (full Zerve stack):
    1. Run zerve_canvas_block.py on the Zerve canvas using Fleets (spread/gather)
       across 3 stars in parallel.
    2. Deploy THIS file as the Hosted App.
    3. The app reads pre-computed results instantly via `from zerve import variable`
       — no re-running any agents.

  Local / standalone:
    streamlit run streamlit_app.py
    (Falls back to running the pipeline directly when zerve package is absent.)

Environment (Zerve Secrets or local .env):
    OPENAI_API_KEY   — required for Science Communicator
    LANGCHAIN_API_KEY — optional, enables LangSmith tracing
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing ─────────────────────────────────────────────
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT",    "exoplanet-swarm")

# ── Zerve vs local mode detection ─────────────────────────────────
# On Zerve Hosted Apps the `zerve` package is injected at runtime.
# Locally it doesn't exist, so we catch the ImportError and run
# the pipeline self-contained.
try:
    from zerve import variable as zerve_variable   # type: ignore
    ZERVE_MODE = True
except ImportError:
    ZERVE_MODE = False


# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Exoplanet Swarm",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
  h1, h2, h3 { color: #e6edf3 !important; }
  div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #8b949e !important; }
  div[data-testid="metric-container"] div   { color: #e6edf3 !important; }
  div[data-testid="stStatus"] { background: #161b22; border-color: #30363d; }
  .summary-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #388bfd; border-radius: 12px;
    padding: 24px 28px; margin: 16px 0;
    box-shadow: 0 0 20px rgba(56,139,253,0.15);
  }
  .badge-detected   { background:#196c2e; color:#3fb950; padding:4px 10px; border-radius:20px; font-size:0.85em; font-weight:600; }
  .badge-undetected { background:#3d1f1f; color:#f78166; padding:4px 10px; border-radius:20px; font-size:0.85em; font-weight:600; }
  button[kind="primary"] { background: linear-gradient(90deg,#1f6feb,#388bfd); border:none; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

STARS = ["Kepler-186", "Kepler-442", "TOI 700", "Kepler-62", "Kepler-452"]

STAR_INFO = {
    "Kepler-186": "5 confirmed planets · Habitable zone Earth-size (186f) · ~490 ly",
    "Kepler-442": "Super-Earth in habitable zone · High habitability score · ~1,200 ly",
    "TOI 700":    "TESS discovery · Earth-size habitable zone planet · ~100 ly",
    "Kepler-62":  "Two habitable zone planets (62e, 62f) · ~1,200 ly",
    "Kepler-452": "Earth's 'cousin' · 385-day year · ~1,400 ly",
}


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🚀 Exoplanet Swarm")
    st.markdown("*Multi-agent AI pipeline for autonomous transit detection*")
    st.divider()

    if ZERVE_MODE:
        st.success(
            "⚡ **Zerve Fleet Mode**\n\n"
            "Results pre-computed in parallel via Zerve canvas Fleets. Instant load.",
            icon="⚡"
        )
    else:
        st.info("🖥️ **Local Mode** — runs pipeline on demand.", icon="🖥️")

    st.divider()

    star_id = st.selectbox(
        "🌟 Target Star", options=STARS, index=0,
        help="Kepler-186 loads from demo CSV cache (no MAST download).",
    )
    st.caption(STAR_INFO.get(star_id, ""))
    st.divider()

    run_button = st.button(
        "🔭 Run Swarm", type="primary", use_container_width=True,
        help="Zerve: reads Fleet results. Local: runs pipeline step-by-step.",
    )

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("""
    1. 🛸 Space Scraper → NASA MAST
    2. 📡 Signal Processor → SG filter
    3. 🔭 Astrophysicist → BLS math
    4. 📝 Science Communicator → GPT-4o
    """)
    st.divider()
    st.caption("CrewAI · lightkurve · astropy · Zerve.ai")


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("# 🚀 Exoplanet Swarm")
st.markdown(
    "Autonomous multi-agent AI pipeline: raw NASA telescope data "
    "→ BLS transit detection → public science summary."
)
st.divider()


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_tools():
    from tools import fetch_lightcurve_tool, clean_signal_tool, bls_periodogram_tool
    ft = fetch_lightcurve_tool.func if hasattr(fetch_lightcurve_tool, "func") else fetch_lightcurve_tool
    ct = clean_signal_tool.func     if hasattr(clean_signal_tool, "func")     else clean_signal_tool
    bt = bls_periodogram_tool.func  if hasattr(bls_periodogram_tool, "func")  else bls_periodogram_tool
    return ft, ct, bt


def _get_science_summary(bls_data: dict, star_id: str) -> str:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2,
                     api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f"""You are the Science Communicator agent. Star: {star_id}.
PlanetMetrics: {json.dumps(bls_data, indent=2)}
RULE: Quote every float VERBATIM. Never round or approximate any number.
Write exactly two paragraphs:
Paragraph 1 (The Observation): telescope/mission, exact orbital_period_days, exact transit_depth_ppm, vivid analogy, detection_quality.
Paragraph 2 (The Meaning): exact planet_probability as %, confirmation steps needed, one sentence on why it matters."""
    return llm.invoke(prompt).content


def _display_results(raw_data, clean_data, bls_data, summary, star_id):
    """Render metric row, science summary card, and 4 Plotly charts."""
    from visualize import (make_raw_lc_figure, make_clean_lc_figure,
                           make_bls_figure, make_phase_fold_figure)

    detected = bls_data.get("planet_detected", False)
    badge = ('<span class="badge-detected">🟢 SIGNAL DETECTED</span>'
             if detected else
             '<span class="badge-undetected">🔴 BELOW THRESHOLD</span>')
    st.markdown(f"## 📡 Results for **{star_id}**  {badge}", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orbital Period",  f"{bls_data['orbital_period_days']:.4f} d")
    c2.metric("Transit Depth",   f"{bls_data['transit_depth_ppm']:.0f} ppm")
    c3.metric("Duration",        f"{bls_data['transit_duration_days']:.3f} d")
    c4.metric("SNR",             f"{bls_data['snr']:.2f}")
    c5.metric("P(planet)",       f"{bls_data['planet_probability']:.1%}")

    if summary:
        st.markdown(f'<div class="summary-card">{summary}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("## 📊 Interactive Diagnostic Charts")
    st.caption("Hover to inspect individual data points. Zoom, pan, download.")

    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering raw light curve..."):
            st.plotly_chart(make_raw_lc_figure(raw_data),
                            use_container_width=True, key=f"raw_{star_id}")
        with st.spinner("Rendering BLS periodogram..."):
            st.plotly_chart(make_bls_figure(clean_data, bls_data),
                            use_container_width=True, key=f"bls_{star_id}")
    with col_r:
        with st.spinner("Rendering cleaned light curve..."):
            st.plotly_chart(make_clean_lc_figure(clean_data),
                            use_container_width=True, key=f"clean_{star_id}")
        with st.spinner("Rendering phase-folded curve..."):
            st.plotly_chart(make_phase_fold_figure(clean_data, bls_data),
                            use_container_width=True, key=f"fold_{star_id}")

    st.caption(f"Data: {raw_data['mission']}  |  Cadences: {raw_data['records']:,}  |  "
               f"Powered by CrewAI · lightkurve · astropy · GPT-4o")


# ══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════

if run_button:

    # ── ZERVE PATH: read pre-computed Fleet results ────────────────
    if ZERVE_MODE:
        with st.spinner("⚡ Loading pre-computed Zerve Fleet results..."):
            try:
                # `from zerve import variable` equivalent at runtime.
                # The 'swarm_execution' canvas block ran spread(stars) in parallel
                # and gathered results into final_results_dict.
                final_results_dict = zerve_variable("swarm_execution", "final_results_dict")
                star_result = final_results_dict.get(star_id)
                if star_result is None:
                    st.error(f"No result for '{star_id}' in Zerve Fleet output. "
                             "Run the swarm_execution canvas block first.")
                    st.stop()

                st.success("⚡ Loaded from Zerve Fleet — no re-computation needed!")
                st.divider()

                bls = star_result["bls"]
                detected = bls.get("planet_detected", False)
                badge = ('<span class="badge-detected">🟢 SIGNAL DETECTED</span>'
                         if detected else
                         '<span class="badge-undetected">🔴 BELOW THRESHOLD</span>')
                st.markdown(f"## 📡 Results for **{star_id}**  {badge}",
                            unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Orbital Period", f"{bls['orbital_period_days']:.4f} d")
                c2.metric("Transit Depth",  f"{bls['transit_depth_ppm']:.0f} ppm")
                c3.metric("Duration",       f"{bls['transit_duration_days']:.3f} d")
                c4.metric("SNR",            f"{bls['snr']:.2f}")
                c5.metric("P(planet)",      f"{bls['planet_probability']:.1%}")

                st.markdown(
                    f'<div class="summary-card">{star_result["communicator_text"]}</div>',
                    unsafe_allow_html=True,
                )
                st.divider()
                st.markdown("## 📊 Interactive Diagnostic Charts")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(star_result["fig_raw"],   use_container_width=True, key="zv_raw")
                    st.plotly_chart(star_result["fig_bls"],   use_container_width=True, key="zv_bls")
                with col_r:
                    st.plotly_chart(star_result["fig_clean"], use_container_width=True, key="zv_clean")
                    st.plotly_chart(star_result["fig_fold"],  use_container_width=True, key="zv_fold")

            except Exception as e:
                st.error(f"Zerve variable import failed: {e}")
                st.info("Falling back to self-contained local run...")
                ZERVE_MODE = False

    # ── LOCAL PATH: run pipeline step-by-step ─────────────────────
    if not ZERVE_MODE:
        fetch_tool, clean_tool, bls_tool = _get_tools()
        raw_data = clean_data = bls_data = summary = None

        with st.status(f"🚀 Running Exoplanet Swarm on **{star_id}**...",
                       expanded=True) as status:

            st.write("🛸 **Space Scraper** — querying NASA MAST archive...")
            t0 = time.time()
            raw_json = fetch_tool(star_id)
            raw_data = json.loads(raw_json)
            if "error" in raw_data:
                status.update(label="❌ Fetch failed", state="error")
                st.error(raw_data["error"]); st.stop()
            st.write(f"   ✅ {raw_data['records']:,} cadences from "
                     f"{raw_data['mission']} in {time.time()-t0:.1f}s")

            st.write("📡 **Signal Processor** — detrending photometry...")
            t0 = time.time()
            clean_json = clean_tool(raw_json)
            clean_data = json.loads(clean_json)
            if "error" in clean_data:
                status.update(label="❌ Clean failed", state="error")
                st.error(clean_data["error"]); st.stop()
            st.write(f"   ✅ Removed {clean_data['removed_outliers']:,} outliers "
                     f"in {time.time()-t0:.1f}s")

            st.write("🔭 **Astrophysicist** — running BLS periodogram...")
            t0 = time.time()
            bls_json = bls_tool(clean_json)
            bls_data = json.loads(bls_json)
            if "error" in bls_data:
                status.update(label="❌ BLS failed", state="error")
                st.error(bls_data["error"]); st.stop()
            st.write(f"   ✅ Period: {bls_data['orbital_period_days']:.4f} d | "
                     f"SNR: {bls_data['snr']:.2f} | "
                     f"{bls_data['detection_quality']} in {time.time()-t0:.1f}s")

            st.write("📝 **Science Communicator** — writing public summary...")
            t0 = time.time()
            try:
                summary = _get_science_summary(bls_data, star_id)
                st.write(f"   ✅ Done in {time.time()-t0:.1f}s")
            except Exception as e:
                summary = f"*Science Communicator unavailable: {e}*"

            status.update(label="✅ Exoplanet Swarm Complete!", state="complete")

        _display_results(raw_data, clean_data, bls_data, summary, star_id)

# ──  EMPTY STATE ──────────────────────────────────────────────────
else:
    mode_msg = (
        "⚡ Zerve Fleet Mode: results pre-computed in parallel across 3 stars. Click to load instantly."
        if ZERVE_MODE else
        "🖥️ Local Mode: click to run the full pipeline step-by-step."
    )
    st.markdown(f"""
    <div style="text-align:center; padding:60px 20px; color:#8b949e;">
        <div style="font-size:72px; margin-bottom:16px;">🌌</div>
        <h3 style="color:#8b949e;">Select a star and click
            <strong style="color:#58a6ff;">Run Swarm</strong></h3>
        <p style="margin-bottom:8px;">{mode_msg}</p>
        <br>
        <p style="font-size:0.85em;">
            🛸 Space Scraper &nbsp;→&nbsp; 📡 Signal Processor &nbsp;→&nbsp;
            🔭 Astrophysicist &nbsp;→&nbsp; 📝 Science Communicator
        </p>
    </div>
    """, unsafe_allow_html=True)
