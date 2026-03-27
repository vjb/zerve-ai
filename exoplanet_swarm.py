"""
╔══════════════════════════════════════════════════════════════════╗
║           EXOPLANET SWARM — CrewAI Multi-Agent Pipeline           ║
║                                                                    ║
║  Autonomously ingests NASA MAST telescope data, cleans signals,   ║
║  runs Box-fitting Least Squares (BLS) transit detection, and      ║
║  produces a public-friendly science summary.                      ║
║                                                                    ║
║  Agents: Space Scraper → Signal Processor → Astrophysicist        ║
║           → Science Communicator                                  ║
╚══════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install crewai langchain-openai lightkurve astropy scipy numpy

Environment:
    OPENAI_API_KEY must be set (or swap LLM config below for another provider).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

# ── Suppress lightkurve and astropy verbose warnings in pipelines ──
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── CrewAI core ────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ── LLM provider ──────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

# ── Signal processing & astronomy ─────────────────────────────────
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════
#  SECTION 1: CUSTOM TOOLS
#  These @tool-decorated functions are the "hands" of each agent.
#  CrewAI passes them as callable tools; agents decide when to invoke
#  them and how to parse their outputs.
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
        - 'time'  : list of BJD timestamps
        - 'flux'  : list of normalized flux values
        - 'records': number of cadences
        - 'mission': mission name string (e.g. 'Kepler')
    On error returns a JSON object with an 'error' key describing the failure.
    """
    try:
        print(f"[Space Scraper] Querying MAST archive for: {star_id}")

        # Search MAST — try Kepler first, fall back to TESS
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

        # Stitch quarters together into one continuous light curve
        lc_stitched = lc_collection.stitch()

        # Use PDCSAP flux when available (pre-conditioned, systematic-corrected)
        # Fall back to SAP or normalized flux if column is absent
        if "pdcsap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("pdcsap_flux")
        elif "sap_flux" in lc_stitched.columns:
            lc_final = lc_stitched.select_flux("sap_flux")
        else:
            lc_final = lc_stitched

        # Normalize: divide every flux value by the median so unit ≈ 1.0
        lc_final = lc_final.normalize()

        # Drop NaN cadences that arise from inter-quarter gaps or cosmic rays
        lc_clean = lc_final.remove_nans()

        time_values = lc_clean.time.value.tolist()
        flux_values = lc_clean.flux.value.tolist()

        print(f"[Space Scraper] Retrieved {len(time_values):,} cadences from {mission}.")

        return json.dumps({
            "star_id": star_id,
            "mission": mission,
            "records": len(time_values),
            "time": time_values,
            "flux": flux_values,
        })

    except Exception as exc:
        return json.dumps({
            "error": f"MAST fetch failed for '{star_id}': {type(exc).__name__}: {str(exc)}"
        })


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

        # ── Stage 1: Savitzky-Golay trend removal ─────────────────
        # Window length must be odd; ~1% of series length keeps short
        # transits (hours wide) intact while removing day-scale variability.
        window_len = max(int(n * 0.01) | 1, 11)  # bitwise OR ensures odd; floor at 11
        polyorder = 3  # cubic polynomial preserves dip shape well

        print(f"[Signal Processor] SG filter: window={window_len}, polyorder={polyorder}")
        trend = savgol_filter(flux_arr, window_length=window_len, polyorder=polyorder)

        # Divide by trend to flatten stellar variations; result ≈ 1.0 ± transits
        flux_detrended = flux_arr / trend

        # ── Stage 2: 3-sigma outlier clipping ─────────────────────
        median_flux = np.median(flux_detrended)
        std_flux = np.std(flux_detrended)
        mask_keep = np.abs(flux_detrended - median_flux) <= 3.0 * std_flux

        removed = int(np.sum(~mask_keep))
        time_clean = time_arr[mask_keep]
        flux_clean = flux_detrended[mask_keep]

        print(f"[Signal Processor] Removed {removed} outlier cadences ({removed/n*100:.2f}%).")
        print(f"[Signal Processor] Clean series: {len(time_clean):,} cadences remain.")

        return json.dumps({
            "star_id": data.get("star_id", "unknown"),
            "mission": data.get("mission", "unknown"),
            "records": len(time_clean),
            "removed_outliers": removed,
            "filter_window_size": window_len,
            "time": time_clean.tolist(),
            "flux": flux_clean.tolist(),
        })

    except Exception as exc:
        return json.dumps({
            "error": f"Signal cleaning failed: {type(exc).__name__}: {str(exc)}"
        })


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
        4. A transit S/N proxy is computed as BLS_power / median(BLS_power),
           and converted to a heuristic planet probability:
              P(transit) = 1 - exp(-SNR / 10)
           This is a simplified proxy, not a Bayesian posterior.

    Returns a JSON string with:
        - 'period_days'  : best-fit orbital period (days)
        - 'transit_depth': fractional flux decrease during transit (ppm)
        - 'duration_days': best-fit transit duration (days)
        - 'bls_power'    : peak BLS power (dimensionless)
        - 'snr'          : signal-to-noise ratio of the detection
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

        # Baseline of observations sets the maximum testable period
        baseline_days = float(time_arr.max() - time_arr.min())
        max_period = baseline_days / 3.0  # aliasing prevention threshold
        min_period = 0.5                  # sub-day orbits exist but are rare

        if max_period <= min_period:
            return json.dumps({
                "error": (
                    f"Observation baseline ({baseline_days:.1f} days) is too short "
                    "to robustly sample BLS periods. Need at least 1.5 days of data."
                )
            })

        # ── Run BLS ───────────────────────────────────────────────
        # Attach astropy units for the BLS interface
        time_with_units = time_arr * u.day
        flux_with_units = flux_arr * u.dimensionless_unscaled

        bls_model = BoxLeastSquares(time_with_units, flux_with_units)

        # Logarithmically spaced periods give denser sampling at short periods
        # where hot Jupiter / super-Earth detections are most common
        n_periods = 5000
        periods = np.exp(
            np.linspace(np.log(min_period), np.log(max_period), n_periods)
        ) * u.day

        print(f"[Astrophysicist] Testing {n_periods} trial periods: "
              f"{min_period:.2f}–{max_period:.2f} days...")

        periodogram = bls_model.power(periods, duration=[0.05, 0.1, 0.15, 0.2] * u.day)

        # ── Extract best period ────────────────────────────────────
        best_idx = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx].to(u.day).value)
        best_power  = float(periodogram.power[best_idx])

        # Use BLS stats at the best period to get depth and duration
        stats = bls_model.compute_stats(
            periodogram.period[best_idx],
            periodogram.duration[best_idx],
            periodogram.transit_time[best_idx],
        )
        transit_depth_frac = float(stats["depth"][0])  # fractional flux drop
        transit_depth_ppm  = transit_depth_frac * 1_000_000
        duration_days      = float(periodogram.duration[best_idx].to(u.day).value)

        # ── Signal-to-Noise heuristic ─────────────────────────────
        median_power = float(np.median(periodogram.power))
        snr = best_power / median_power if median_power > 0 else 0.0

        # Heuristic planet probability: saturates at 1.0 for strong signals
        planet_prob = float(1.0 - np.exp(-snr / 10.0))

        # Qualitative classification for the Science Communicator agent
        if snr >= 15:
            quality = "Strong"
        elif snr >= 7:
            quality = "Moderate"
        elif snr >= 3:
            quality = "Weak"
        else:
            quality = "Noise"

        print(f"[Astrophysicist] Best period: {best_period:.4f} days | "
              f"Depth: {transit_depth_ppm:.1f} ppm | SNR: {snr:.2f} | "
              f"Quality: {quality}")

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


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: LLM CONFIGURATION
#  Swap this block to use Anthropic, Gemini, Ollama, etc.
#  CrewAI accepts any LangChain-compatible chat model.
# ══════════════════════════════════════════════════════════════════

llm = ChatOpenAI(
    model="gpt-4o",            # Best reasoning for scientific interpretation
    temperature=0.2,           # Low temp = more deterministic analysis
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: AGENT DEFINITIONS
#  Each agent has a Role, Goal, and Backstory that shapes how the
#  LLM reasons about its sub-problem. Tools are injected at this
#  level so the agent knows what capabilities it has.
# ══════════════════════════════════════════════════════════════════

space_scraper = Agent(
    role="Space Scraper",
    goal=(
        "Retrieve complete, high-quality raw FITS photometric data from the "
        "NASA MAST archive for the target star. Ensure all available quarters "
        "or sectors are downloaded and the light curve is stitched and normalized."
    ),
    backstory=(
        "You are a veteran data pipeline engineer who cut your teeth at the "
        "Space Telescope Science Institute. You know every quirk of the MAST "
        "archive API and can coax data out of stubborn catalog IDs. You pride "
        "yourself on never returning incomplete datasets — if Kepler doesn't "
        "have it, you'll find it in TESS. You always sanity-check record counts "
        "before declaring success."
    ),
    tools=[fetch_lightcurve_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,  # This agent owns its task end-to-end
)

signal_processor = Agent(
    role="Signal Processor",
    goal=(
        "Apply rigorous photometric detrending to remove stellar variability "
        "and instrumental systematics from the light curve, while carefully "
        "preserving the shallow, brief flux dips that indicate planetary transits. "
        "Do NOT over-smooth — a missed transit dip is worse than residual noise."
    ),
    backstory=(
        "You hold a PhD in time-series analysis and spent a decade at NASA Ames "
        "developing the PDC-MAP detrending pipeline for Kepler. You have an almost "
        "paranoid respect for the signal preservation problem: over-smoothing "
        "erases the very dips you are hunting. You treat every cadence as precious "
        "and document exactly what your filters removed and why."
    ),
    tools=[clean_signal_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

astrophysicist = Agent(
    role="Astrophysicist",
    goal=(
        "Execute the Box-fitting Least Squares periodogram on the cleaned photometric "
        "data, identify the most statistically significant periodic signal, and "
        "quantify the probability that this signal represents a genuine exoplanet transit. "
        "Provide the orbital period, transit depth (ppm), duration, SNR, and a "
        "clear detection quality assessment."
    ),
    backstory=(
        "You are an exoplanet detection specialist who has personally confirmed "
        "47 exoplanets using transit photometry. You championed BLS periodograms "
        "before the Kepler mission launched and have tuned the method to squeeze "
        "signals out of noisy data. You are rigorous: you label weak detections "
        "as weak and never hype marginal signals. Your transit probability estimates "
        "are trusted throughout the community."
    ),
    tools=[bls_periodogram_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

science_communicator = Agent(
    role="Science Communicator",
    goal=(
        "Transform the raw numerical BLS outputs (period, depth, SNR, probability) "
        "into exactly two paragraphs of clear, accurate, and exciting prose. "
        "The first paragraph must explain what was observed; the second must explain "
        "what it means for the possibility of a planet. Write for a scientifically "
        "curious public, not specialists. Use analogies. Be honest about uncertainty."
    ),
    backstory=(
        "You are a former astrophysics professor turned science journalist who has "
        "written cover stories for Scientific American and Sky & Telescope. You have "
        "a gift for translating arcane mathematics into vivid, accurate narratives "
        "that move people. You never sensationalize — you earn wonder through clarity. "
        "You know that 'probability' and 'certainty' are very different words and "
        "you use them precisely."
    ),
    tools=[],   # This agent reasons over text — no tool calls needed
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 4: TASK DEFINITIONS
#  Tasks wire agents to specific instructions and chain outputs.
#  context=[prev_task] makes the prior task's output available as
#  additional context in the next agent's prompt.
# ══════════════════════════════════════════════════════════════════

task_scrape = Task(
    description=(
        "Use the fetch_lightcurve_tool to retrieve all available photometric data "
        "for the star '{star_id}' from the NASA MAST archive. "
        "Verify the download succeeded (no error key in the output). "
        "Report: mission name, number of cadences retrieved, and the time baseline in days. "
        "Pass the full JSON result forward for signal processing."
    ),
    expected_output=(
        "A valid JSON string containing keys: star_id, mission, records, time, flux. "
        "The 'records' value must be greater than 100. No 'error' key should be present."
    ),
    agent=space_scraper,
)

task_clean = Task(
    description=(
        "Using the raw light curve JSON from the Space Scraper, apply the clean_signal_tool "
        "to remove stellar variability and outliers. "
        "Report how many outlier cadences were removed and confirm the Savitzky-Golay "
        "window size used. Emphasize: do NOT use a window so wide that it could flatten "
        "transit dips shorter than 0.1% of the total baseline. "
        "Pass the cleaned JSON forward for BLS analysis."
    ),
    expected_output=(
        "A valid JSON string with keys: star_id, mission, records, removed_outliers, "
        "filter_window_size, time, flux. Flux values should be centered near 1.0."
    ),
    agent=signal_processor,
    context=[task_scrape],
)

task_bls = Task(
    description=(
        "Using the cleaned light curve JSON from the Signal Processor, execute the "
        "bls_periodogram_tool to run a full BLS transit search. "
        "Identify the best candidate orbital period and report: period (days), "
        "transit depth (ppm), transit duration (days), BLS power, SNR, detection_quality, "
        "and planet_probability. "
        "If detection_quality is 'Noise', still report all numbers and flag the low confidence."
    ),
    expected_output=(
        "A valid JSON string with keys: star_id, mission, period_days, transit_depth_ppm, "
        "duration_days, bls_power, snr, planet_probability, detection_quality."
    ),
    agent=astrophysicist,
    context=[task_clean],
)

task_communicate = Task(
    description=(
        "You have received the BLS analysis results for star '{star_id}'. "
        "Write EXACTLY two paragraphs for a public science communication response:\n\n"
        "**Paragraph 1 (The Observation):** Describe what telescope data was used, "
        "how long it was observed, and what the BLS algorithm found — specifically the "
        "orbital period ({period_days} days) and transit depth ({transit_depth_ppm} ppm). "
        "Explain transit depth using an analogy (e.g., compare the star's disk to a beach ball "
        "and the planet to a marble).\n\n"
        "**Paragraph 2 (The Meaning):** Explain the planet probability score and what it "
        "implies. Mention the detection quality rating. Be honest about what further "
        "confirmation (e.g., radial velocity follow-up) would be needed to call this a "
        "confirmed exoplanet. End with one sentence about why this finding matters — "
        "what it implies for the star's habitability or our understanding of planetary systems."
    ),
    expected_output=(
        "Exactly two clearly labeled paragraphs. No bullet points or JSON. "
        "Written in English for a general audience. Scientifically accurate but accessible. "
        "Should reference the actual numbers from the BLS output."
    ),
    agent=science_communicator,
    context=[task_bls],
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 5: CREW ASSEMBLY
#  Process.sequential means tasks execute one at a time in order.
#  The output of each Task feeds into the context of the next.
# ══════════════════════════════════════════════════════════════════

crew = Crew(
    agents=[space_scraper, signal_processor, astrophysicist, science_communicator],
    tasks=[task_scrape, task_clean, task_bls, task_communicate],
    process=Process.sequential,
    verbose=True,          # Print agent reasoning steps to console
    memory=False,          # Stateless per-run — no cross-run memory needed
    max_rpm=10,            # Polite rate limit for the LLM API
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 6: ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Allow override via CLI: python exoplanet_swarm.py "Kepler-442"
    star_target = sys.argv[1] if len(sys.argv) > 1 else "Kepler-186"

    print("\n" + "═" * 65)
    print(f"  🚀  EXOPLANET SWARM INITIALIZING")
    print(f"  🌟  Target Star : {star_target}")
    print(f"  🤖  Agents      : 4 (sequential)")
    print("═" * 65 + "\n")

    # kickoff() starts the sequential pipeline; inputs are interpolated
    # into task description strings wherever {star_id} etc. appear.
    result = crew.kickoff(inputs={"star_id": star_target})

    print("\n" + "═" * 65)
    print("  ✅  EXOPLANET SWARM COMPLETE")
    print("═" * 65)
    print("\n📡 FINAL SCIENCE SUMMARY:\n")
    print(result)
    print()
