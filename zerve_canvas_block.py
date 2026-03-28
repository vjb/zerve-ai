"""
zerve_canvas_block.py
──────────────────────
ZERVE CANVAS BLOCK — paste this code into a Python block on the Zerve canvas.
Name the block: swarm_execution

How to use on Zerve:
  1. Create a new Python block, paste this entire file.
  2. At the top of the Zerve canvas, create a list variable:
         stars = ["Kepler-186", "Kepler-442", "TOI 700"]
  3. Use Zerve's spread(stars) UI button to fan this block into 3 parallel
     Fleet executions — each execution receives a different star_id.
  4. Connect the outputs to a Gather block; the gathered dict becomes
     final_results_dict which streamlit_app.py reads via from zerve import variable.

This block receives `star_id` as an injected variable from the spread().
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# ── star_id is injected by Zerve spread() — override for local testing ──
if "star_id" not in dir():
    star_id = "Kepler-186"   # default for local testing

print(f"[Zerve Fleet] Running Exoplanet Swarm for: {star_id}")

# ── Import tools ──────────────────────────────────────────────────
from tools import fetch_lightcurve_tool, clean_signal_tool, bls_periodogram_tool
from visualize import (make_raw_lc_figure, make_clean_lc_figure,
                       make_bls_figure, make_phase_fold_figure)
from langchain_openai import ChatOpenAI

ft = fetch_lightcurve_tool.func if hasattr(fetch_lightcurve_tool, "func") else fetch_lightcurve_tool
ct = clean_signal_tool.func     if hasattr(clean_signal_tool, "func")     else clean_signal_tool
bt = bls_periodogram_tool.func  if hasattr(bls_periodogram_tool, "func")  else bls_periodogram_tool

# ── Step 1: Fetch ─────────────────────────────────────────────────
print(f"[{star_id}] Step 1/4 — Space Scraper fetching MAST data...")
raw_json  = ft(star_id)
raw_data  = json.loads(raw_json)
assert "error" not in raw_data, f"Fetch failed: {raw_data['error']}"
print(f"[{star_id}] ✅ {raw_data['records']:,} cadences from {raw_data['mission']}")

# ── Step 2: Clean ─────────────────────────────────────────────────
print(f"[{star_id}] Step 2/4 — Signal Processor detrending...")
clean_json = ct(raw_json)
clean_data = json.loads(clean_json)
assert "error" not in clean_data, f"Clean failed: {clean_data['error']}"
print(f"[{star_id}] ✅ Removed {clean_data['removed_outliers']:,} outliers")

# ── Step 3: BLS ───────────────────────────────────────────────────
print(f"[{star_id}] Step 3/4 — Astrophysicist running BLS periodogram...")
bls_json = bt(clean_json)
bls_data = json.loads(bls_json)
assert "error" not in bls_data, f"BLS failed: {bls_data['error']}"
print(f"[{star_id}] ✅ Period: {bls_data['orbital_period_days']:.4f} d | "
      f"SNR: {bls_data['snr']:.2f} | {bls_data['detection_quality']}")

# ── Step 4: Science Communicator (LLM) ────────────────────────────
print(f"[{star_id}] Step 4/4 — Science Communicator writing summary...")
llm = ChatOpenAI(
    model="gpt-4o", temperature=0.2,
    api_key=os.environ.get("OPENAI_API_KEY"),
)
prompt = f"""You are the Science Communicator agent. Star: {star_id}.

PlanetMetrics JSON:
{json.dumps(bls_data, indent=2)}

RULE: Quote every float VERBATIM — never round or approximate.

Write exactly two paragraphs:
Paragraph 1 (The Observation): telescope/mission, orbital_period_days verbatim, transit_depth_ppm verbatim, analogy for the transit depth, detection_quality.
Paragraph 2 (The Meaning): planet_probability verbatim as %, confirmation needed, one sentence on why it matters.
"""
communicator_text = llm.invoke(prompt).content
print(f"[{star_id}] ✅ Summary written ({len(communicator_text)} chars)")

# ── Step 5: Build Plotly figures ──────────────────────────────────
print(f"[{star_id}] Building interactive Plotly figures...")
fig_raw   = make_raw_lc_figure(raw_data)
fig_clean = make_clean_lc_figure(clean_data)
fig_bls   = make_bls_figure(clean_data, bls_data)
fig_fold  = make_phase_fold_figure(clean_data, bls_data)

# ── Final output dict ─────────────────────────────────────────────
# Zerve's gather() collects one of these per star into final_results_dict.
# streamlit_app.py reads it via: variable("swarm_execution", "final_results_dict")
star_result = {
    "star_id":           star_id,
    "mission":           raw_data["mission"],
    "records":           raw_data["records"],
    "bls":               bls_data,
    "communicator_text": communicator_text,
    "fig_raw":           fig_raw,
    "fig_clean":         fig_clean,
    "fig_bls":           fig_bls,
    "fig_fold":          fig_fold,
}

print(f"[{star_id}] ✅ Complete. Returning result to Zerve gather().")
