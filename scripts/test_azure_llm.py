from __future__ import annotations

from pprint import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_scout.src.services.azure_openai import generate_medical_coordination_notes


if __name__ == "__main__":
    match_context = {
        "team_a": "England",
        "team_b": "Australia",
        "venue": "Lords",
        "date": "2023-01-29",
    }

    roster_snapshot = {
        "high_risk_report": "Jofra Archer (England, driver acute_ratio, risk 0.78); Pat Cummins (Australia, driver load, risk 0.75)",
        "moderate_risk_report": "Ben Stokes (England, driver soreness, risk 0.55); Mitchell Starc (Australia, driver travel, risk 0.52)",
        "readiness_mean": 0.61,
        "workload_flags": "AC ratio spike >1.3 for Jofra Archer; Acute load >125% of chronic for Pat Cummins",
        "per_team_high": {
            "England": "Jofra Archer (risk 0.78)",
            "Australia": "Pat Cummins (risk 0.75)",
        },
        "per_team_moderate": {
            "England": "Ben Stokes (risk 0.55)",
            "Australia": "Mitchell Starc (risk 0.52)",
        },
        "per_team_workload": {
            "England": "AC>1.3: Jofra Archer",
            "Australia": "AC>1.3: Pat Cummins; Travel: Mitchell Starc",
        },
    }

    fallback_notes = [
        "Fallback: rotate Archer early spell, keep Cummins in short bursts",
    ]

    notes = generate_medical_coordination_notes(match_context, roster_snapshot, fallback_notes)
    pprint(notes)
