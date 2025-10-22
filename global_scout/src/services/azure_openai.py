from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, cast

from openai import AzureOpenAI

AZURE_DEFAULT_DEPLOYMENT = "gpt-4.1"
AZURE_DEFAULT_API_VERSION = "2024-12-01-preview"


def _build_client() -> Optional[tuple[AzureOpenAI, str]]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", AZURE_DEFAULT_DEPLOYMENT)
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", AZURE_DEFAULT_API_VERSION)

    if not api_key or not endpoint or not deployment:
        return None

    try:
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    except Exception:
        return None
    return client, deployment


def generate_medical_coordination_notes(
    match_context: Dict[str, str],
    roster_snapshot: Dict[str, object],
    base_notes: List[str],
    temperature: float = 0.3,
) -> Optional[List[str]]:
    client_info = _build_client()
    if client_info is None:
        return None

    client, deployment = client_info

    team_a = match_context.get("team_a", "Team A")
    team_b = match_context.get("team_b", "Team B")
    venue = match_context.get("venue", "unknown venue")
    date = match_context.get("date", "TBD")

    high_risk_text = roster_snapshot.get("high_risk_report", "None")
    moderate_risk_text = roster_snapshot.get("moderate_risk_report", "None")
    readiness_mean = roster_snapshot.get("readiness_mean", 0)
    workload_trends = roster_snapshot.get("workload_flags", "None")
    per_team_high = cast(Dict[str, str], roster_snapshot.get("per_team_high") or {})
    per_team_moderate = cast(Dict[str, str], roster_snapshot.get("per_team_moderate") or {})
    per_team_workload = cast(Dict[str, str], roster_snapshot.get("per_team_workload") or {})

    # Build a match-specific prompt including per-team breakdowns where available
    team_breakdown_lines = []
    for t in (team_a, team_b):
        th = per_team_high.get(t, "None")
        tm = per_team_moderate.get(t, "None")
        tw = per_team_workload.get(t, "None")
        team_breakdown_lines.append(f"{t}: High-risk -> {th}; Moderate-risk -> {tm}; Workload -> {tw}.")

    user_prompt = (
        "You are the Digital Physio agent collaborating with the Tactical agent in a cricket analyst war-room. "
        "Produce 3 match-specific recommendations as JSON (no markdown).\n\n"
        f"Match: {team_a} vs {team_b} on {date} at {venue}.\n"
        f"Squad readiness avg: {readiness_mean:.2f}.\n"
        f"Global high-risk players: {high_risk_text}.\n"
        f"Global moderate-risk players: {moderate_risk_text}.\n"
        f"Global workload considerations: {workload_trends}.\n"
        "Team breakdown:\n"
        + "\n".join(team_breakdown_lines)
        + "\n\n"
        "Respond with a JSON array of exactly 3 objects. Each object must contain keys:"
        " team (string), focus (string describing risk insight), medical_action (string), tactical_adjustment (string)."
        "Keep language concise (<30 words per field) and reference specific players/roles."
        "Example: [{\"team\":\"England\",\"focus\":\"...\",\"medical_action\":\"...\",\"tactical_adjustment\":\"...\"}, ...]."
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CAIN's Digital Physio expert. Provide medically grounded, tactically actionable guidance. "
                        "Return only valid JSON (no Markdown, no code fences)."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=350,
            top_p=0.9,
        )
    except Exception:
        return None

    text = response.choices[0].message.content if response.choices else ""
    if not text:
        return None

    bullets: List[str] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload[:5]:
                if isinstance(item, dict):
                    team = str(item.get("team", "Team"))
                    focus = str(item.get("focus", "Key focus"))
                    medical = str(item.get("medical_action", "Medical action"))
                    tactical = str(item.get("tactical_adjustment", "Tactical adjustment"))
                    bullets.append(
                        f"[{team}] {focus} — Medical: {medical}. Tactical: {tactical}."
                    )
    except json.JSONDecodeError:
        bullets = []

    if bullets:
        return bullets

    # Fallback: split plain text into lines, merge with base notes for safety
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    return base_notes or None
