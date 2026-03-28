"""
main.py — Exoplanet Swarm
──────────────────────────
Entry point. Imports the crew from agents.py and kicks off the
sequential pipeline for a target star.

Usage:
    python main.py                  # Default: Kepler-186
    python main.py "Kepler-442"
    python main.py "TOI 700"
    python main.py "TRAPPIST-1"

Environment:
    OPENAI_API_KEY must be set. See .env.example for reference.
"""

import sys
from agents import make_crew


def run(star_id: str) -> str:
    """
    Execute the full Exoplanet Swarm pipeline for a given star.

    Args:
        star_id: Any star identifier recognized by lightkurve
                 (e.g. 'Kepler-186', 'TOI 700', 'Kepler-442')

    Returns:
        The Science Communicator's final two-paragraph summary string.
    """
    print("\n" + "═" * 65)
    print(f"  🚀  EXOPLANET SWARM INITIALIZING")
    print(f"  🌟  Target Star : {star_id}")
    print(f"  🤖  Agents      : Space Scraper → Signal Processor")
    print(f"                    → Astrophysicist → Science Communicator")
    print("═" * 65 + "\n")

    crew   = make_crew(star_id)
    result = crew.kickoff(inputs={"star_id": star_id})

    print("\n" + "═" * 65)
    print("  ✅  EXOPLANET SWARM COMPLETE")
    print("═" * 65)
    print("\n📡 FINAL SCIENCE SUMMARY:\n")
    print(result)
    print()

    return str(result)


if __name__ == "__main__":
    # Allow override via CLI argument: python main.py "Kepler-442"
    star_target = sys.argv[1] if len(sys.argv) > 1 else "Kepler-186"
    run(star_target)
