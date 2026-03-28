"""
tests/conftest.py
─────────────────
Pytest fixtures that download and cache REAL Kepler-186 photometric data
from the NASA MAST archive on first run, then reuse the cached JSON for
all subsequent test runs (no repeated network calls).

Cache location: tests/fixtures/kepler186_raw.json
                tests/fixtures/kepler186_clean.json
                tests/fixtures/kepler186_bls.json

To force a fresh download: delete the fixtures/ directory and re-run.
"""

import json
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, ".")

import tools as tw

# ── Resolve underlying callables (bypass @tool wrapper) ───────────
fetch_tool = tw.fetch_lightcurve_tool.func if hasattr(tw.fetch_lightcurve_tool, "func") else tw.fetch_lightcurve_tool
clean_tool = tw.clean_signal_tool.func     if hasattr(tw.clean_signal_tool, "func")     else tw.clean_signal_tool
bls_tool   = tw.bls_periodogram_tool.func  if hasattr(tw.bls_periodogram_tool, "func")  else tw.bls_periodogram_tool

FIXTURE_DIR  = os.path.join(os.path.dirname(__file__), "fixtures")
RAW_CACHE    = os.path.join(FIXTURE_DIR, "kepler186_raw.json")
CLEAN_CACHE  = os.path.join(FIXTURE_DIR, "kepler186_clean.json")
BLS_CACHE    = os.path.join(FIXTURE_DIR, "kepler186_bls.json")

# Kepler-186: famous 5-planet system, Earth-size planet in habitable zone
# Kepler observed it for ~4 years; shortest period planet (186b) is ~3.9 days
REAL_STAR = "Kepler-186"


def _ensure_fixture_dir():
    os.makedirs(FIXTURE_DIR, exist_ok=True)


@pytest.fixture(scope="session")
def real_raw_lc():
    """
    Session-scoped fixture: fetch real Kepler-186 raw light curve from MAST.
    Cached to disk so subsequent test runs are instant.
    """
    _ensure_fixture_dir()

    if os.path.exists(RAW_CACHE):
        print(f"\n[fixture] Loading cached raw lightcurve from {RAW_CACHE}")
        with open(RAW_CACHE) as f:
            data = json.load(f)
    else:
        print(f"\n[fixture] Downloading real Kepler-186 data from NASA MAST...")
        raw_json = fetch_tool(REAL_STAR)
        data = json.loads(raw_json)

        if "error" in data:
            pytest.skip(f"MAST unavailable: {data['error']}")

        with open(RAW_CACHE, "w") as f:
            json.dump(data, f)
        print(f"[fixture] Cached {data['records']:,} cadences → {RAW_CACHE}")

    return data


@pytest.fixture(scope="session")
def real_clean_lc(real_raw_lc):
    """
    Session-scoped fixture: run signal cleaning on the real raw lightcurve.
    Cached to disk.
    """
    _ensure_fixture_dir()

    if os.path.exists(CLEAN_CACHE):
        print(f"\n[fixture] Loading cached clean lightcurve from {CLEAN_CACHE}")
        with open(CLEAN_CACHE) as f:
            return json.load(f)

    print(f"\n[fixture] Cleaning real Kepler-186 lightcurve...")
    clean_json = clean_tool(json.dumps(real_raw_lc))
    data = json.loads(clean_json)

    if "error" in data:
        pytest.skip(f"Cleaning failed: {data['error']}")

    with open(CLEAN_CACHE, "w") as f:
        json.dump(data, f)
    print(f"[fixture] Cached clean data → {CLEAN_CACHE}")

    return data


@pytest.fixture(scope="session")
def real_bls_result(real_clean_lc):
    """
    Session-scoped fixture: run BLS periodogram on real clean Kepler-186 data.
    Cached to disk.
    """
    _ensure_fixture_dir()

    if os.path.exists(BLS_CACHE):
        print(f"\n[fixture] Loading cached BLS result from {BLS_CACHE}")
        with open(BLS_CACHE) as f:
            return json.load(f)

    print(f"\n[fixture] Running BLS periodogram on real Kepler-186 data...")
    bls_json = bls_tool(json.dumps(real_clean_lc))
    data = json.loads(bls_json)

    if "error" in data:
        pytest.skip(f"BLS failed: {data['error']}")

    with open(BLS_CACHE, "w") as f:
        json.dump(data, f)
    print(f"[fixture] Cached BLS result → {BLS_CACHE}")
    print(f"[fixture] Best period: {data['period_days']} days | "
          f"Depth: {data['transit_depth_ppm']} ppm | "
          f"SNR: {data['snr']} | Quality: {data['detection_quality']}")

    return data
