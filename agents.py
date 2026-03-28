"""
agents.py — Exoplanet Swarm
────────────────────────────
CrewAI Agent and Task definitions for the Exoplanet Swarm pipeline.

Imports the three @tool functions from tools.py and wires them to four
specialized agents in a sequential task chain.

LLM is lazy-initialized so this module can be imported without
OPENAI_API_KEY (tests only exercise the tools, not the agents).
"""

import os

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from tools import fetch_lightcurve_tool, clean_signal_tool, bls_periodogram_tool


# ══════════════════════════════════════════════════════════════════
#  LLM — Lazy singleton
#  Swap ChatOpenAI for Anthropic, Gemini, Ollama, etc. as needed.
# ══════════════════════════════════════════════════════════════════

_llm_instance = None

def _get_llm():
    """Return a cached ChatOpenAI instance; instantiated on first call."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model="gpt-4o",    # Best reasoning for scientific interpretation
            temperature=0.2,   # Low temp = deterministic analysis
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _llm_instance


# ══════════════════════════════════════════════════════════════════
#  AGENTS
# ══════════════════════════════════════════════════════════════════

def make_agents():
    """
    Instantiate and return the four Exoplanet Swarm agents.
    Triggers LLM initialization — requires OPENAI_API_KEY.

    Returns:
        (space_scraper, signal_processor, astrophysicist, science_communicator)
    """
    llm = _get_llm()

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
        allow_delegation=False,
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

    return space_scraper, signal_processor, astrophysicist, science_communicator


# ══════════════════════════════════════════════════════════════════
#  TASKS + CREW
# ══════════════════════════════════════════════════════════════════

def make_crew(star_id: str) -> Crew:
    """
    Build and return a fully wired sequential Crew for the given star_id.

    Task chain:
        task_scrape → task_clean → task_bls → task_communicate

    Each task's output is passed as context to the next via context=[...].

    Args:
        star_id: Any star identifier recognized by lightkurve
                 (e.g. 'Kepler-186', 'TOI 700', 'Kepler-442')

    Returns:
        A configured Crew ready for .kickoff()
    """
    space_scraper, signal_processor, astrophysicist, science_communicator = make_agents()

    task_scrape = Task(
        description=(
            "Use the fetch_lightcurve_tool to retrieve all available photometric data "
            f"for the star '{star_id}' from the NASA MAST archive. "
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
            f"You have received a PlanetMetrics JSON object for star '{star_id}'. "
            "This JSON contains exact scientific measurements. "

            "CRITICAL RULE: You MUST quote every float VERBATIM as it appears in the JSON. "
            "Do NOT round, approximate, or paraphrase any numerical value. "
            "If orbital_period_days is 30.012777, you write '30.012777 days' — "
            "never 'about 30 days' or 'roughly a month'. "
            "If transit_depth_ppm is 2115.15, you write '2,115.15 ppm' exactly. "

            "Write EXACTLY two paragraphs:\n\n"

            "**Paragraph 1 (The Observation):** State the telescope and mission. "
            "Describe what BLS found — quote orbital_period_days and transit_depth_ppm "
            "verbatim. Explain transit depth with an analogy (e.g. if the star's disk "
            "were a soccer field, the planet's shadow would cover roughly X blades of grass). "
            "Reference planet_detected and detection_quality.\n\n"

            "**Paragraph 2 (The Meaning):** Quote planet_probability verbatim as a percentage. "
            "Explain what further confirmation (radial velocity or additional transits) would "
            "be needed to officially confirm the planet. "
            "End with one sentence on why this matters — habitability, planetary system "
            "architecture, or the search for life."
        ),
        expected_output=(
            "Exactly two labeled paragraphs. No bullet points or JSON. "
            "Every float from the PlanetMetrics JSON is quoted verbatim. "
            "Scientifically accurate and written for a general audience."
        ),
        agent=science_communicator,
        context=[task_bls],
    )

    return Crew(
        agents=[space_scraper, signal_processor, astrophysicist, science_communicator],
        tasks=[task_scrape, task_clean, task_bls, task_communicate],
        process=Process.sequential,
        verbose=True,
        memory=False,
        max_rpm=10,
    )
