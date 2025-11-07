import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import pandas as pd
import streamlit as st

from src.analyzer import (
    compute_per_bowler_wicket_clusters,
    compute_phase_run_rates,
    compute_phase_run_rates_per_team,
    compute_wicket_clusters,
    load_deliveries_csv,
    load_matches_csv,
    load_players_csv,
    monte_carlo_simulation,
    plot_phase_run_rates_per_team,
    safe_int,
)

# Make Global Scout and Digital Physio pipelines importable when running from repo root
root_dir = Path(__file__).resolve().parent
# Attempt to load environment variables from a local .env file so Azure creds are available in Streamlit sessions
_env_path = root_dir / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=_env_path, override=False)
    except Exception:
        # Lightweight fallback: parse key=value pairs manually without external dependency
        try:
            for line in _env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            pass
score_candidates = None  # type: ignore
mitigate_and_rank = None  # type: ignore
load_wellness_data = None  # type: ignore
compute_wellness_risk = None  # type: ignore
summarize_wellness = None  # type: ignore
generate_wellness_guidance = None  # type: ignore
build_integrity_agent = None  # type: ignore
build_supervisor_agent = None  # type: ignore
try:
    # Preferred: import via package path so relative imports inside pipelines work
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    from global_scout.src.pipelines.rank_candidates import score_candidates as _sc, mitigate_and_rank as _mr  # type: ignore
    from global_scout.src.pipelines.wellness import (  # type: ignore
        load_wellness_data as _load_wellness,
        compute_injury_risk as _compute_wellness,
        summarize_risk as _summarize_wellness,
        generate_coach_guidance as _generate_guidance,
    )
    score_candidates, mitigate_and_rank = _sc, _mr
    load_wellness_data = _load_wellness
    compute_wellness_risk = _compute_wellness
    summarize_wellness = _summarize_wellness
    generate_wellness_guidance = _generate_guidance
    from global_scout.src.agents.integrity.adjudication_agent import (
        build_agent as _build_integrity_agent,
    )  # type: ignore
    build_integrity_agent = _build_integrity_agent
    from global_scout.src.agents.supervisor.supervisor_agent import (
        build_agent as _build_supervisor_agent,
    )  # type: ignore
    build_supervisor_agent = _build_supervisor_agent
except Exception:
    pass


st.set_page_config(
    page_title="CAIN Strategist Control Room",
    page_icon="🏏",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_data_cached(
    deliveries_mtime: float,
    matches_mtime: float,
    players_mtime: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    deliveries = load_deliveries_csv(Path("deliveries.csv"))
    matches = load_matches_csv(Path("matches.csv"))
    players = load_players_csv(Path("players.csv"))

    deliveries_df = pd.DataFrame(deliveries)
    if not deliveries_df.empty:
        deliveries_df["runs_off_bat"] = deliveries_df["runs_off_bat"].fillna(0)
        deliveries_df["extras"] = deliveries_df["extras"].fillna(0)
        deliveries_df["runs_total"] = deliveries_df["runs_off_bat"] + deliveries_df["extras"]
        deliveries_df["wicket"] = deliveries_df["wicket_type"].fillna("") != ""
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty and "match_id" in matches_df.columns:
        matches_df["match_id"] = matches_df["match_id"].apply(safe_int)

    players_map = {int(pid): name for pid, name in players.items()} if players else {}
    return deliveries_df, matches_df, players_map


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    deliveries_path = Path("deliveries.csv")
    matches_path = Path("matches.csv")
    players_path = Path("players.csv")

    deliveries_mtime = deliveries_path.stat().st_mtime if deliveries_path.exists() else 0.0
    matches_mtime = matches_path.stat().st_mtime if matches_path.exists() else 0.0
    players_mtime = players_path.stat().st_mtime if players_path.exists() else 0.0

    return _load_data_cached(deliveries_mtime, matches_mtime, players_mtime)


def compute_innings_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(columns=["innings", "runs", "wickets", "overs", "run_rate", "boundaries"])
    for innings in sorted(df["innings"].dropna().astype(int).unique()):
        inn_df = df[df["innings"] == innings]
        runs = float(inn_df["runs_total"].sum())
        wickets = int(inn_df["wicket"].sum())
        balls = int(inn_df["runs_total"].count())
        overs_completed = balls // 6
        balls_remaining = balls % 6
        overs_text = f"{overs_completed}.{balls_remaining}"
        run_rate = (runs / (balls / 6)) if balls else 0
        boundaries = int((inn_df["runs_off_bat"] == 4).sum() + (inn_df["runs_off_bat"] == 6).sum())
        records.append(
            {
                "innings": innings,
                "runs": int(runs),
                "wickets": wickets,
                "overs": overs_text,
                "run_rate": round(run_rate, 2),
                "boundaries": boundaries,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_over_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["innings", "over", "runs", "wickets"])
    grouped = (
        df.groupby(["innings", "over"])
        .agg(runs=("runs_total", "sum"), wickets=("wicket", "sum"))
        .reset_index()
        .sort_values(["innings", "over"])
    )
    grouped["runs"] = grouped["runs"].astype(int)
    grouped["wickets"] = grouped["wickets"].astype(int)
    return grouped


def compute_wicket_clusters_sliding(
    df: pd.DataFrame,
    window: int = 6,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compute sliding-window wicket clusters directly from ball-by-ball data."""
    if df.empty or "wicket" not in df.columns:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    working = df.dropna(subset=["innings", "over"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    working["innings"] = working["innings"].astype(int)
    working["over"] = working["over"].astype(int)
    wickets_by_over = (
        working.groupby(["innings", "over"])["wicket"].sum().reset_index()
    )

    clusters: List[Dict[str, int]] = []
    for innings_key, inn_df in wickets_by_over.groupby("innings"):
        innings_val = safe_int(innings_key)
        if innings_val is None:
            continue
        overs_series = inn_df["over"].astype(int)
        overs_list = sorted(overs_series.tolist())
        if not overs_list:
            continue
        min_over = int(overs_list[0])
        max_over = int(overs_list[-1])
        for start in range(min_over, max_over + 1):
            end = start + window - 1
            wickets_sum = inn_df[(inn_df["over"] >= start) & (inn_df["over"] <= end)]["wicket"].sum()
            wickets = int(wickets_sum) if wickets_sum else 0
            if wickets > 0:
                clusters.append(
                    {
                        "innings": int(innings_val),
                        "start_over": int(start),
                        "end_over": int(end),
                        "wickets": wickets,
                    }
                )

    if not clusters:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    clusters_df = pd.DataFrame(clusters).drop_duplicates()
    clusters_df = clusters_df.sort_values(
        ["wickets", "innings", "start_over"], ascending=[False, True, True]
    )
    return clusters_df.head(top_n)


def label_phase(over: int) -> str:
    if over <= 5:
        return "Powerplay"
    if over <= 14:
        return "Middle"
    return "Death"


def generate_tactical_recommendations(
    teams: List[str],
    phase_team: Dict[str, Dict[str, Dict[str, float]]],
    wicket_clusters: List[Dict[str, object]],
    bowler_clusters: List[Dict[str, object]],
    over_summary: pd.DataFrame,
    monte_stats: Dict[str, object],
) -> List[str]:
    notes: List[str] = []
    if teams:
        notes.append(
            f"**Supervisor Agent**: Coordinate strategy briefing for {teams[0]} vs {teams[1]} using the live dashboards below."
        )
    for team, phases in (phase_team or {}).items():
        for phase_name, stats in phases.items():
            balls = stats.get("balls", 0)
            rpo = stats.get("rpo", 0)
            if not balls:
                continue
            if phase_name.startswith("Powerplay") and rpo < 6:
                notes.append(
                    f"**Tactical Agent**: Powerplay run rate for {team} is {rpo:.2f}. Promote intent with pinch-hitter drills or staggered batting orders."
                )
            if phase_name.startswith("Middle") and rpo < 6:
                notes.append(
                    f"**Performance & Wellness Agent**: Schedule mid-innings batting scenarios for {team} to unlock rotation of strike; current run rate {rpo:.2f}."
                )
            if phase_name.startswith("Late") and rpo > 10:
                notes.append(
                    f"**Commercial Agent**: Highlight {team}'s finishing burst (run rate {rpo:.2f}) in fan engagement reels for the upcoming fixture."
                )
    if wicket_clusters:
        cluster = wicket_clusters[0]
        notes.append(
            f"**Tactical Agent**: Target wicket surge in innings {cluster['innings']} (overs {cluster['start_over']}-{cluster['end_over']}) with field traps; replicate successful plans."
        )
    if bowler_clusters:
        bow = bowler_clusters[0]
        notes.append(
            f"**Supervisor Agent**: Assign {bow.get('bowler_name')} focused overs {bow['start_over']}-{bow['end_over']} (innings {bow['innings']}); cluster yielded {bow['wickets']} wickets."
        )
    if not over_summary.empty:
        death_slump = over_summary[(over_summary["over"] >= 16) & (over_summary["runs"] < 6)]
        if not death_slump.empty:
            overs_list = ", ".join(str(int(o)) for o in sorted(death_slump["over"].unique()))
            notes.append(
                f"**Performance Agent**: Death overs {overs_list} leaked below-par runs; rehearse yorker execution and slower-ball variations."
            )
    if monte_stats and monte_stats.get("trials", 0):
        win1 = monte_stats.get("win_prob_innings1", 0)
        win2 = monte_stats.get("win_prob_innings2", 0)
        notes.append(
            f"**Simulation Agent**: Simulations suggest innings 1 win probability {win1:.0%} vs innings 2 {win2:.0%}; adjust bowling changes accordingly."
        )
    if not notes:
        notes.append(
            "**Supervisor Agent**: Leverage CAIN playbooks to synthesize scouting, performance, and tactical data for a unified match directive."
        )
    return notes


def enrich_ball_by_ball(df: pd.DataFrame, players: Dict[int, str]) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    for column, target in [("batter_id", "batter"), ("bowler_id", "bowler"), ("non_striker_id", "non_striker")]:
        enriched[target] = enriched[column].apply(lambda pid: players.get(int(pid), str(pid)) if not pd.isna(pid) else "")
    enriched["runs_total"] = enriched["runs_total"].astype(int)
    enriched["wicket_type"] = (
        enriched["wicket_type"].fillna("-").replace({"": "-", "None": "-", "none": "-"})
    )
    enriched["dismissed"] = enriched["dismissed_batter_id"].apply(
        lambda pid: players.get(int(pid), str(int(pid))) if not (pid is None or pd.isna(pid)) else "-"
    )
    enriched["phase"] = enriched["over"].apply(lambda o: label_phase(int(o)))
    columns = [
        "innings",
        "over",
        "ball",
        "phase",
        "batter",
        "bowler",
        "runs_off_bat",
        "extras",
        "runs_total",
        "wicket_type",
        "dismissed",
    ]
    return enriched[columns]


def format_match_title(row: pd.Series) -> str:
    if row is None or row.empty:
        return "Select a match"
    team_a = row.get("team_a", "Team A")
    team_b = row.get("team_b", "Team B")
    venue = row.get("venue", "")
    start_date = row.get("date", "")
    end_date = row.get("end_date", "")
    series_name = row.get("series_name")

    date_label = ""
    if start_date and end_date and str(end_date) != str(start_date):
        date_label = f"{start_date} to {end_date}"
    elif start_date:
        date_label = str(start_date)

    title = series_name if series_name else f"{team_a} vs {team_b}"
    if venue and date_label:
        return f"{title} — {venue} ({date_label})"
    if venue:
        return f"{title} — {venue}"
    if date_label:
        return f"{title} ({date_label})"
    return title


def main() -> None:
    # Sidebar view selector first so Global Scout can work even if cricket CSVs are missing
    view = st.sidebar.radio(
        "View",
        [
            "Supervisor Command Deck",
            "Tactical Command Center",
            "Global Scout",
            "Performance & Wellness",
            "Adjudication & Integrity",
        ],
        index=0,
        help="Switch between match analysis and scouting/recruitment",
    )

    if view == "Supervisor Command Deck":
        render_supervisor_command()
        return
    if view == "Global Scout":
        render_global_scout()
        return
    if view == "Performance & Wellness":
        render_performance_wellness()
        return
    if view == "Adjudication & Integrity":
        render_adjudication_integrity()
        return

    deliveries_df, matches_df, players_map = load_data()

    if matches_df.empty or deliveries_df.empty:
        st.error("Required CSV files are missing or empty. Ensure deliveries.csv and matches.csv are present.")
        return

    st.sidebar.header("Match Selector")
    match_options: List[Tuple[int, str]] = []
    for _, row in matches_df.iterrows():
        match_id_val = safe_int(row.get("match_id"))
        if match_id_val is None:
            continue
        match_options.append((match_id_val, format_match_title(row)))
    if not match_options:
        st.warning("No matches found in matches.csv")
        return

    default_match = match_options[0][0]
    match_labels = {mid: label for mid, label in match_options}
    selected_match = st.sidebar.selectbox(
        "Choose a match",
        options=[mid for mid, _ in match_options],
        format_func=lambda mid: match_labels.get(mid, str(mid)),
        index=0,
    )

    monte_trials = st.sidebar.slider("Simulation runs", min_value=200, max_value=2000, value=800, step=200,
                                      help="Number of simulated innings used for win-chance estimates.")

    match_innings = sorted(
        deliveries_df[deliveries_df["match_id"] == selected_match]["innings"].dropna().astype(int).unique()
    )
    innings_filter = st.sidebar.multiselect(
        "Show innings",
        options=match_innings,
        default=match_innings,
    )

    match_row = matches_df[matches_df["match_id"] == selected_match].iloc[0]
    match_title = format_match_title(match_row)
    st.title("CAIN Tactical & Simulation Command Center")
    st.caption(
        "A Streamlit control room for the Cricketing Agentic Intelligence Nexus (CAIN) pipeline."
    )
    st.subheader(match_title)

    match_deliveries = deliveries_df[deliveries_df["match_id"] == selected_match].copy()
    if innings_filter:
        match_deliveries = match_deliveries[match_deliveries["innings"].isin(innings_filter)]
    if match_deliveries.empty:
        st.warning("No deliveries for the selected filters.")
        return

    context = {
        "meta": {"match_id": selected_match},
        "info": {
            "teams": [match_row.get("team_a", "Team A"), match_row.get("team_b", "Team B")],
            "dates": [match_row.get("date")],
            "venue": match_row.get("venue"),
            "outcome": {"winner": match_row.get("winner")},
        },
    }

    deliveries_records: List[Dict[str, Any]] = match_deliveries.to_dict("records")  # type: ignore[assignment]
    phase_overall = compute_phase_run_rates(deliveries_records, context)
    phase_team = compute_phase_run_rates_per_team(deliveries_records, context)
    wicket_clusters_df = compute_wicket_clusters_sliding(match_deliveries)
    wicket_clusters: List[Dict[str, object]] = []
    for record in wicket_clusters_df.to_dict("records"):
        wicket_clusters.append(
            {
                "innings": int(record.get("innings", 0)),
                "start_over": int(record.get("start_over", 0)),
                "end_over": int(record.get("end_over", 0)),
                "wickets": int(record.get("wickets", 0)),
            }
        )
    bowler_clusters = compute_per_bowler_wicket_clusters(deliveries_records, context)
    monte_stats = monte_carlo_simulation(deliveries_records, context, trials=monte_trials)

    innings_summary = compute_innings_summary(match_deliveries)
    over_summary = compute_over_summary(match_deliveries)
    enriched_balls = enrich_ball_by_ball(match_deliveries, players_map)

    # --- Key metrics row ---
    col1, col2, col3, col4 = st.columns(4)
    winner = match_row.get("winner", "-")
    col1.metric("Winner", winner or "-", help="Result from matches.csv")
    col2.metric("Simulation Runs", monte_stats.get("trials", 0))
    col3.metric(
        "Avg Run Rate",
        f"{innings_summary['run_rate'].mean():.2f}" if not innings_summary.empty else "-",
    )

    # Compute prediction accuracy by innings rather than team labels (robust to naming)
    try:
        p1 = float(monte_stats.get("win_prob_innings1", 0) or 0.0)
        p2 = float(monte_stats.get("win_prob_innings2", 0) or 0.0)
        tie_p = float(monte_stats.get("tie_prob", 0) or 0.0)

        if p1 > p2:
            predicted_innings = 1
            predicted_label = "Innings 1"
            confidence = p1
        elif p2 > p1:
            predicted_innings = 2
            predicted_label = "Innings 2"
            confidence = p2
        else:
            predicted_innings = 0  # tie
            predicted_label = "Tie"
            confidence = tie_p

        # Actual winner by runs from deliveries
        runs_by_innings = match_deliveries.groupby("innings")["runs_total"].sum()
        runs1 = int(runs_by_innings.get(1, 0))
        runs2 = int(runs_by_innings.get(2, 0))
        if runs1 == 0 and runs2 == 0:
            actual_innings = None
        elif runs1 > runs2:
            actual_innings = 1
        elif runs2 > runs1:
            actual_innings = 2
        else:
            actual_innings = 0  # tie

        if predicted_innings in (1, 2) and actual_innings in (1, 2):
            acc = 1.0 if predicted_innings == actual_innings else 0.0
            accuracy_display = f"{int(acc * 100)}%"
        elif predicted_innings == 0 and actual_innings == 0:
            accuracy_display = "100%"
        else:
            accuracy_display = "N/A"

        actual_label = (
            f"Innings {actual_innings}" if actual_innings in (1, 2) else ("Tie" if actual_innings == 0 else "Unknown")
        )
        col4.metric(
            "Prediction Accuracy",
            accuracy_display,
            help=f"Predicted: {predicted_label} (confidence {confidence:.0%}) vs Actual: {actual_label}",
        )
    except Exception:
        col4.metric("Prediction Accuracy", "N/A")

    st.markdown("---")
    st.markdown("### Innings Overview")
    overview_cols = st.columns(len(innings_summary) or 1)
    for idx, (_, row) in enumerate(innings_summary.iterrows()):
        col = overview_cols[idx % len(overview_cols)]
        col.metric(
            f"Innings {int(row['innings'])}",
            f"{int(row['runs'])}/{int(row['wickets'])}",
            delta=f"RR {row['run_rate']}",
        )
        col.caption(f"Overs: {row['overs']} | Boundaries: {row['boundaries']}")

    with st.expander("Phase-wise Run Rates by Team", expanded=True):
        if phase_team:
            fig = plot_phase_run_rates_per_team(phase_team)
            if fig is not None:
                st.pyplot(fig)

            # Flatten the same per-team stats powering the chart so the table matches exactly
            phase_rows: List[Dict[str, object]] = []
            for team_name, phases in phase_team.items():
                for phase_name, stats in phases.items():
                    phase_rows.append(
                        {
                            "Team": team_name,
                            "Phase": phase_name,
                            "Runs": stats.get("runs", 0),
                            "Balls": stats.get("balls", 0),
                            "RPO": round(float(stats.get("rpo", 0.0)), 2),
                        }
                    )

            if phase_rows:
                phase_df = pd.DataFrame(phase_rows).sort_values(["Team", "Phase"])
                st.dataframe(phase_df, use_container_width=True, hide_index=True)
        else:
            st.info("No phase data available for this match.")

    st.markdown("### Run Distribution by Over")
    if not over_summary.empty:
        runs_pivot = over_summary.pivot(index="over", columns="innings", values="runs").fillna(0)
        st.bar_chart(runs_pivot)
    else:
        st.info("Run distribution unavailable.")

    top_overs = (
        over_summary.sort_values("runs", ascending=False)
        .head(5)
        .assign(phase=lambda df: df["over"].apply(lambda o: label_phase(int(o))))
    )
    top_overs["detail"] = top_overs.apply(
        lambda row: f"Innings {row['innings']} • Over {int(row['over'])} • Runs {row['runs']} • Wickets {row['wickets']}",
        axis=1,
    )
    st.markdown("#### High-Impact Overs")
    if not top_overs.empty:
        for detail in top_overs["detail"]:
            st.write(f"- {detail}")
    else:
        st.info("No standout overs detected.")

    st.markdown("### Wicket Clusters")
    if not wicket_clusters_df.empty:
        st.caption(
            "Shows overs where wickets bunched together inside a sliding six-over window. Useful for spotting collapse risk and pressure spikes."
        )
        clusters_display = wicket_clusters_df.copy()
        clusters_display["overs_window"] = clusters_display.apply(
            lambda row: f"{int(row['start_over'])}-{int(row['end_over'])}", axis=1
        )
        clusters_display = clusters_display.rename(
            columns={
                "innings": "Innings",
                "overs_window": "Overs Window",
                "wickets": "Wickets",
            }
        )
        display_columns = ["Innings", "Overs Window", "Wickets"]
        st.dataframe(
            clusters_display[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        if clusters_display["Wickets"].nunique() == 1:
            st.info(
                "All highlighted windows captured the same wicket count for this match — indicating wickets were evenly spread rather than multiple in quick succession."
            )
    else:
        st.info("No wicket clusters identified.")

    st.markdown("### Simulation Snapshot")
    if monte_stats and monte_stats.get("trials", 0):
        monte_df = pd.DataFrame(
            {
                "Metric": [
                    "Win Chance (Batting First)",
                    "Win Chance (Chasing)",
                    "Tie Chance",
                    "Projected Score (Batting First)",
                    "Projected Score (Chasing)",
                    "Median Score (Batting First)",
                    "Median Score (Chasing)",
                ],
                "Value": [
                    f"{monte_stats.get('win_prob_innings1', 0):.0%}",
                    f"{monte_stats.get('win_prob_innings2', 0):.0%}",
                    f"{monte_stats.get('tie_prob', 0):.0%}",
                    monte_stats.get("inn1_stats", {}).get("mean", 0),
                    monte_stats.get("inn2_stats", {}).get("mean", 0),
                    monte_stats.get("inn1_stats", {}).get("median", 0),
                    monte_stats.get("inn2_stats", {}).get("median", 0),
                ],
            }
        )
        st.table(monte_df)
    else:
        st.info("Simulation insufficient due to limited over data.")

    st.markdown("### Ball-by-Ball Ledger")
    st.dataframe(enriched_balls, use_container_width=True, hide_index=True)

    st.markdown("### CAIN Tactical Recommendations")
    teams = context["info"].get("teams", [])
    recommendations = generate_tactical_recommendations(
        teams,
        phase_team,
        wicket_clusters,
        bowler_clusters,
        over_summary,
        monte_stats,
    )
    for note in recommendations:
        st.markdown(f"- {note}")

    st.markdown("---")
    st.caption(
        "CAIN (Cricketing Agentic Intelligence Nexus) synthesizes multi-agent insights across tactical, performance, scouting, commercial, and integrity lanes."
    )


def render_global_scout() -> None:
    st.title("Global Scout — Objective, Bias-Aware Scouting")
    st.caption("Identify future talent objectively and without bias, aligned to CAIN's agentic loop (perceive → reason → act → learn).")

    if score_candidates is None or mitigate_and_rank is None:
        st.error("Global Scout pipeline is unavailable. Ensure the 'global_scout' module exists and dependencies are installed.")
        st.stop()

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("#### Data Source")
        source = st.radio("Choose candidate data", ["Demo data", "Upload CSV"], horizontal=True, index=0)
    with colB:
        st.markdown("#### Fairness & Shortlist")
        shortlist_k = st.slider("Shortlist size (approx)", min_value=5, max_value=50, value=10, step=5,
                                help="Top candidates above a quantile-based cutoff around this size.")

    df = None
    if source == "Demo data":
        demo_path = Path("global_scout/data/demo_players.csv")
        if not demo_path.exists():
            # Fallback: synthesize small demo dataset
            import random
            import pandas as pd
            rng = random.Random(42)
            roles = ["batter", "bowler", "allrounder", "keeper"]
            regions = ["North", "South", "East", "West"]
            rows = []
            for i in range(40):
                role = rng.choice(roles)
                rows.append({
                    "player_id": f"D{i+1:03d}",
                    "name": f"Demo Player {i+1}",
                    "age": rng.randint(18, 34),
                    "role": role,
                    "league_level": rng.randint(1, 5),
                    "matches": rng.randint(5, 70),
                    "batting_sr": round(rng.uniform(85, 185), 1) if role != "bowler" else None,
                    "batting_avg": round(rng.uniform(12, 62), 1) if role != "bowler" else None,
                    "bowling_eco": round(rng.uniform(4.5, 10.5), 2) if role != "batter" else None,
                    "bowling_avg": round(rng.uniform(12, 55), 1) if role != "batter" else None,
                    "fielding_eff": round(rng.uniform(0.6, 0.98), 2),
                    "recent_form": round(rng.uniform(0.2, 0.95), 2),
                    "region": rng.choice(regions),
                })
            df = pd.DataFrame(rows)
        else:
            import pandas as pd
            df = pd.read_csv(demo_path)
    else:
        import pandas as pd
        file = st.file_uploader("Upload candidates CSV", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)

    if df is None:
        st.info("Provide candidate data to proceed.")
        return

    # Allow user to pick protected attribute from columns
    protected_cols = [c for c in df.columns if c.lower() in ("region", "gender", "country", "state")]
    protected = st.selectbox("Protected attribute for fairness auditing", options=protected_cols or ["region"], index=0)

    with st.expander("Input Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Score and rank
    try:
        scored = score_candidates(df)
        ranked, audits = mitigate_and_rank(scored, protected=protected, shortlist_k=shortlist_k)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Explanation pane
    st.markdown("---")
    st.markdown("#### How candidates are evaluated")
    st.markdown(
        "- Performance score: role-aware blend (Batter: SR, AVG; Bowler: ECO, AVG; Allrounder/keeper: both) + recent form\n"
        "- Strategic fit: matches played, league level, recent form\n"
        "- Final score: fairness mitigation via within-group standardization on protected attribute, then calibrated to 0..1\n"
        "- Shortlist: top quantile around size K; audits compute demographic parity and equal opportunity diffs"
    )

    # Audits
    st.markdown("#### Fairness Audits")
    cols = st.columns(3)
    cols[0].metric("Protected Attribute", protected)
    cols[1].metric("Demographic Parity Δ", f"{audits.get('demographic_parity_diff', 0):.3f}")
    cols[2].metric("Equal Opportunity Δ", f"{audits.get('equal_opportunity_diff', 0):.3f}")

    group_stats_df = None
    if audits.get("group_stats"):
        group_stats_df = pd.DataFrame(audits["group_stats"])
        rename_map = {"count": "count", "mean": "avg_final_score", "std": "score_std"}
        group_stats_df = group_stats_df.rename(columns=rename_map)
        ordered_cols = [c for c in [protected, "count", "avg_final_score", "score_std"] if c in group_stats_df.columns]
        group_stats_df = group_stats_df[ordered_cols]
        st.dataframe(group_stats_df, use_container_width=True, hide_index=True)

    st.markdown("#### Fairness Group Stats Heatmap")
    if group_stats_df is not None and not group_stats_df.empty:
        shortlist_rates = (
            ranked.groupby(protected)["shortlisted"].mean().reset_index(name="shortlist_rate")
        )
        fairness_chart_df = group_stats_df.merge(shortlist_rates, on=protected, how="left")
        fairness_chart_df["shortlist_rate"] = fairness_chart_df["shortlist_rate"].fillna(0.0)
        heatmap_base = fairness_chart_df.set_index(protected)[["avg_final_score", "shortlist_rate"]]
        styled_heatmap = heatmap_base.style.format(
            {
                "avg_final_score": "{:.2f}",
                "shortlist_rate": "{:.1%}",
            }
        ).background_gradient(cmap="RdYlGn_r")
        st.dataframe(styled_heatmap, use_container_width=True)
        st.caption("Color-coded cells highlight demographic slices drifting high (green) or low (red) on final score and shortlist rate.")
    else:
        st.info("Fairness heatmap available once mitigation stats are computed.")

    st.markdown("#### Top-Ranked Candidate Slices")
    shortlist_df = ranked[ranked["shortlisted"] == 1]
    if shortlist_df.empty:
        st.info("No shortlisted candidates at the current threshold.")
    else:
        import matplotlib.pyplot as plt

        col_role, col_protected = st.columns(2)
        with col_role:
            if "role" in shortlist_df.columns:
                role_counts = (
                    shortlist_df["role"].fillna("Unknown Role").value_counts().sort_values(ascending=False)
                )
                fig, ax = plt.subplots(figsize=(4, 3))
                role_counts.plot(kind="bar", ax=ax, color="#1f77b4")
                ax.set_title("Shortlist by Role")
                ax.set_xlabel("Role")
                ax.set_ylabel("Players")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Role field unavailable in dataset.")

        with col_protected:
            protected_counts = (
                shortlist_df[protected].fillna("Unknown").astype(str).value_counts().sort_values(ascending=False)
            )
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            protected_counts.plot(kind="bar", ax=ax2, color="#ff7f0e")
            ax2.set_title(f"Shortlist by {protected.title()}")
            ax2.set_xlabel(protected.title())
            ax2.set_ylabel("Players")
            ax2.tick_params(axis="x", rotation=45, labelsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)

    # Ranked candidates
    st.markdown("#### Ranked Candidates")
    display_cols = [c for c in [
        "rank", "player_id", "name", "role", "age", "league_level", "matches",
        "performance_score", "fit_score", "final_score", protected, "shortlisted"
    ] if c in ranked.columns]
    st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)

    # Download
    import io
    csv_buf = io.StringIO()
    ranked.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Ranked Candidates (CSV)",
        data=csv_buf.getvalue(),
        file_name="candidate_rankings.csv",
        mime="text/csv",
    )


def render_performance_wellness() -> None:
    st.title("Digital Physio — Performance & Wellness Command")
    st.caption(
        "Fuse load management, readiness, and injury risk intelligence into CAIN's agentic feedback loop."
    )

    if not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
        st.info(
            "Set environment variables AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT "
            "to enable AI-personalised Medical & Tactical coordination guidance."
        )

    if None in (load_wellness_data, compute_wellness_risk, summarize_wellness, generate_wellness_guidance):
        st.error("Digital Physio pipeline unavailable. Ensure wellness module is present under global_scout/src/pipelines.")
        st.stop()

    source = st.radio("Data source", ["Demo wellness data", "Upload CSV"], index=0, horizontal=True)

    df = None
    if source == "Demo wellness data":
        df = load_wellness_data()  # type: ignore[operator]
    else:
        file = st.file_uploader("Upload wellness CSV", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)

    if df is None or df.empty:
        st.info("Provide wellness data to analyze readiness signals.")
        return

    required_cols = {
        "player_id",
        "name",
        "team",
        "role",
        "acute_load",
        "chronic_load",
        "acute_chronic_ratio",
        "wellness_score",
        "soreness",
        "sleep_hours",
        "injury_history",
        "recovery_index",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        st.error(
            "Missing required columns: " + ", ".join(missing_cols) + ". Refer to global_scout/data/wellness_demo.csv for schema."
        )
        return

    risk_df = compute_wellness_risk(df)  # type: ignore[misc]
    summary = summarize_wellness(risk_df)  # type: ignore[misc]

    teams = sorted(risk_df["team"].dropna().unique())
    roles = sorted(risk_df["role"].dropna().unique())

    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        team_filter = st.multiselect("Teams", options=teams, default=teams)
    with col_f2:
        role_filter = st.multiselect("Roles", options=roles, default=roles)
    with col_f3:
        readiness_cut = st.slider("Minimum readiness", 0.0, 1.0, 0.0, 0.05)

    filtered = risk_df.copy()
    if team_filter:
        filtered = filtered[filtered["team"].isin(team_filter)]
    if role_filter:
        filtered = filtered[filtered["role"].isin(role_filter)]
    if readiness_cut > 0:
        filtered = filtered[filtered["readiness"] >= readiness_cut]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Average Risk Score", f"{summary['mean_risk']:.2f}")
    col_m2.metric("Average Readiness", f"{summary['readiness_mean']:.2f}")
    col_m3.metric("High-Risk Players", f"{summary['high_risk_pct']:.0%}")
    col_m4.metric("Moderate-Risk Players", f"{summary['moderate_risk_pct']:.0%}")

    st.markdown("#### Readiness Snapshot")
    if filtered.empty:
        st.warning("No athletes match the current filters.")
    else:
        display_cols = [
            "player_id",
            "name",
            "team",
            "role",
            "risk_score",
            "readiness",
            "risk_level",
            "primary_driver",
            "recommendation",
        ]
        display_df = filtered[display_cols].copy()
        driver_friendly = {
            "acute_ratio": "Workload imbalance",
            "load": "Heavy recent workload",
            "soreness": "Muscle soreness",
            "wellness": "Low wellness score",
            "sleep": "Limited sleep",
            "injury_history": "Previous injuries",
            "recovery": "Slower recovery",
            "travel": "Travel fatigue",
            "bowling": "Bowling workload",
            "batting": "Batting workload",
            "sprint": "Sprint volume",
        }
        def map_driver(value: str) -> str:
            if pd.isna(value) or value == "":
                return "No single driver"
            return driver_friendly.get(value, str(value).replace("_", " ").title())

        display_df["primary_driver"] = display_df["primary_driver"].apply(map_driver)
        display_df = display_df.rename(
            columns={
                "player_id": "Player ID",
                "name": "Player",
                "team": "Team",
                "role": "Role",
                "risk_score": "Risk Score",
                "readiness": "Readiness Score",
                "risk_level": "Risk Level",
                "primary_driver": "Main Risk Driver",
                "recommendation": "Suggested Action",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        viz_cols = st.columns(2)
        with viz_cols[0]:
            chart_df = (
                filtered.set_index("name")["risk_score"]
                .sort_values(ascending=False)
                .rename("Injury Risk Score")
            )
            st.bar_chart(chart_df, height=260)
        with viz_cols[1]:
            scatter_df = filtered[["acute_chronic_ratio", "readiness", "name", "team"]]
            scatter_df = scatter_df.rename(
                columns={
                    "acute_chronic_ratio": "Workload Ratio",
                    "readiness": "Readiness Score",
                    "name": "Player",
                    "team": "Team",
                }
            )
            st.scatter_chart(scatter_df, x="Workload Ratio", y="Readiness Score", color="Team")

    st.markdown("#### Training Load Summary")
    workload_cols = ["name", "acute_load", "chronic_load", "bowling_overs_last_7d", "batting_balls_last_7d"]
    available_workload_cols = [c for c in workload_cols if c in risk_df.columns]
    if available_workload_cols:
        workload_display = risk_df[available_workload_cols].rename(
            columns={
                "name": "Player",
                "acute_load": "Recent Load",
                "chronic_load": "Baseline Load",
                "bowling_overs_last_7d": "Overs (7 days)",
                "batting_balls_last_7d": "Balls Faced (7 days)",
            }
        )
        st.dataframe(workload_display, use_container_width=True, hide_index=True)

    st.markdown("#### Coordinated Action Plan")
    match_df = pd.DataFrame(load_matches_csv(Path("matches.csv")))
    match_row = None
    if not match_df.empty and "match_id" in match_df.columns:
        match_df["match_id"] = match_df["match_id"].apply(safe_int)
        match_df = match_df.dropna(subset=["match_id"])
        match_options = []
        for _, row in match_df.iterrows():
            title = format_match_title(row)
            match_options.append((int(row["match_id"]), title))
        if match_options:
            match_label_map = dict(match_options)
            selected_match = st.selectbox(
                "Upcoming fixture context",
                options=[mid for mid, _ in match_options],
                format_func=lambda mid: match_label_map.get(mid, str(mid)),
                index=0,
            )
            match_row = match_df[match_df["match_id"] == selected_match].iloc[0]

    guidance = generate_wellness_guidance(risk_df, match_row=match_row)  # type: ignore[misc]
    for note in guidance:
        st.markdown(f"- {note}")

    st.markdown("---")
    st.caption("Digital Physio feeds assistant coach briefings, helping align tactical over plans with athlete well-being signals.")


def render_supervisor_command() -> None:
    st.title("Supervisor Command Deck — Agentic Orchestration")
    st.caption("Head Coach agent aligning Tactical, Performance, Integrity, and Scouting pipelines.")

    if build_supervisor_agent is None:
        st.error(
            "Supervisor agent pipeline unavailable. Ensure global_scout/src/agents/supervisor is present and dependencies installed."
        )
        st.stop()

    deliveries_df, matches_df, players_map = load_data()
    _ = players_map  # Supervisor view only needs match and delivery data
    if matches_df.empty or deliveries_df.empty:
        st.error("Required CSV files are missing or empty. Ensure deliveries.csv and matches.csv are present.")
        return

    st.sidebar.header("Supervisor Controls")
    match_options: List[Tuple[int, str]] = []
    for _, row in matches_df.iterrows():
        match_id_val = safe_int(row.get("match_id"))
        if match_id_val is None:
            continue
        match_options.append((match_id_val, format_match_title(row)))
    if not match_options:
        st.warning("No matches found in matches.csv")
        return

    match_labels = {mid: label for mid, label in match_options}
    selected_match = st.sidebar.selectbox(
        "Focus match",
        options=[mid for mid, _ in match_options],
        format_func=lambda mid: match_labels.get(mid, str(mid)),
        index=0,
    )

    monte_trials = st.sidebar.slider(
        "Simulation runs",
        min_value=200,
        max_value=2000,
        value=800,
        step=200,
        help="Number of simulated innings powering the tactical projections.",
    )

    match_row = matches_df[matches_df["match_id"] == selected_match].iloc[0]
    match_title = format_match_title(match_row)
    st.subheader(match_title)

    match_deliveries = deliveries_df[deliveries_df["match_id"] == selected_match].copy()
    if match_deliveries.empty:
        st.warning("No deliveries for the selected match.")
        return

    context = {
        "meta": {"match_id": selected_match},
        "info": {
            "teams": [match_row.get("team_a", "Team A"), match_row.get("team_b", "Team B")],
            "dates": [match_row.get("date")],
            "venue": match_row.get("venue"),
            "outcome": {"winner": match_row.get("winner")},
        },
    }

    deliveries_records: List[Dict[str, Any]] = match_deliveries.to_dict("records")  # type: ignore[assignment]
    phase_team = compute_phase_run_rates_per_team(deliveries_records, context)
    wicket_clusters_df = compute_wicket_clusters_sliding(match_deliveries)
    wicket_clusters: List[Dict[str, object]] = []
    for record in wicket_clusters_df.to_dict("records"):
        wicket_clusters.append(
            {
                "innings": int(record.get("innings", 0)),
                "start_over": int(record.get("start_over", 0)),
                "end_over": int(record.get("end_over", 0)),
                "wickets": int(record.get("wickets", 0)),
            }
        )
    bowler_clusters = compute_per_bowler_wicket_clusters(deliveries_records, context)
    monte_stats = monte_carlo_simulation(deliveries_records, context, trials=monte_trials)

    innings_summary = compute_innings_summary(match_deliveries)
    over_summary = compute_over_summary(match_deliveries)
    top_overs = (
        over_summary.sort_values("runs", ascending=False)
        .head(3)
        .assign(phase=lambda df: df["over"].apply(lambda o: label_phase(int(o))))
    )

    runs_by_innings = match_deliveries.groupby("innings")["runs_total"].sum().to_dict()
    wickets_by_innings = match_deliveries.groupby("innings")["wicket"].sum().to_dict()

    match_meta = {
        "match_id": selected_match,
        "title": match_title,
        "teams": context["info"].get("teams", []),
        "venue": match_row.get("venue"),
        "date": match_row.get("date"),
        "winner": match_row.get("winner"),
        "runs_by_innings": {str(k): int(v) for k, v in runs_by_innings.items()},
        "wickets_by_innings": {str(k): int(v) for k, v in wickets_by_innings.items()},
    }

    tactical_snapshot = {
        "phase_team": {
            team: {
                phase: {
                    "rpo": round(stats.get("rpo", 0.0), 2),
                    "balls": stats.get("balls", 0),
                }
                for phase, stats in phases.items()
            }
            for team, phases in (phase_team or {}).items()
        },
        "wicket_clusters": wicket_clusters[:3],
        "bowler_clusters": bowler_clusters[:3],
        "monte": {
            "win_prob_innings1": monte_stats.get("win_prob_innings1", 0) if monte_stats else 0,
            "win_prob_innings2": monte_stats.get("win_prob_innings2", 0) if monte_stats else 0,
            "tie_prob": monte_stats.get("tie_prob", 0) if monte_stats else 0,
        },
        "top_overs": top_overs.to_dict("records") if not top_overs.empty else [],
    }

    wellness_snapshot: Dict[str, Any] = {}
    wellness_chart_df: Optional[pd.DataFrame] = None
    if None not in (load_wellness_data, compute_wellness_risk, summarize_wellness):
        try:
            wellness_df = load_wellness_data()  # type: ignore[misc]
        except Exception:
            wellness_df = None
        if wellness_df is not None and not wellness_df.empty:
            risk_df = compute_wellness_risk(wellness_df)  # type: ignore[misc]
            match_teams = [team for team in context["info"].get("teams", []) if team]
            relevant = risk_df[risk_df["team"].isin(match_teams)] if match_teams else risk_df
            summary_df = relevant if not relevant.empty else risk_df
            summary = summarize_wellness(summary_df)  # type: ignore[misc]
            high_names = summary_df[summary_df["risk_level"] == "High"]["name"].head(3).tolist()
            top_driver = summary_df[summary_df["risk_level"] == "High"]["primary_driver"].mode()
            wellness_chart_df = summary_df.sort_values("risk_score", ascending=False).head(8)[
                ["name", "risk_score", "readiness"]
            ]
            wellness_snapshot = {
                "readiness_mean": summary.get("readiness_mean"),
                "high_risk_pct": summary.get("high_risk_pct"),
                "high_risk_names": high_names,
                "top_driver": top_driver.iloc[0] if not top_driver.empty else None,
            }

    integrity_snapshot: Dict[str, Any] = {}
    integrity_insights = None
    integrity_pressure_chart: Optional[pd.DataFrame] = None
    if build_integrity_agent is not None:
        try:
            integrity_agent = build_integrity_agent(match_deliveries, match_row, overs_window=3)
            integrity_insights = integrity_agent.build_insights()
            integrity_snapshot = {
                "alerts": integrity_insights.alerts,
                "hotspots": integrity_insights.review_hotspots.to_dict("records")
                if not integrity_insights.review_hotspots.empty
                else [],
                "pressure_windows": integrity_insights.high_verdict_windows,
            }
            if not integrity_insights.pressure_windows.empty:
                integrity_pressure_chart = (
                    integrity_insights.pressure_windows.sort_values("pressure_index", ascending=False)
                    .head(6)
                    .assign(
                        window=lambda df: df.apply(
                            lambda r: f"I{int(r['innings'])} {int(r['start_over'])}-{int(r['end_over'])}", axis=1
                        )
                    )
                )
        except Exception:
            integrity_snapshot = {}

    recruiting_snapshot: Dict[str, Any] = {}
    candidate_path = Path("global_scout/data/candidate_rankings.csv")
    if candidate_path.exists():
        try:
            cand_df = pd.read_csv(candidate_path)
            if not cand_df.empty:
                highlights = (
                    cand_df.head(3)
                    .apply(lambda row: f"{row.get('name', 'Candidate')} ({row.get('role', 'role')})", axis=1)
                    .tolist()
                )
                recruiting_snapshot["shortlist_highlights"] = highlights
        except Exception:
            recruiting_snapshot = {}

    supervisor_agent = build_supervisor_agent(
        match_meta=match_meta,
        tactical_snapshot=tactical_snapshot,
        wellness_snapshot=wellness_snapshot,
        integrity_snapshot=integrity_snapshot,
        recruiting_snapshot=recruiting_snapshot,
    )

    try:
        supervisor_output = supervisor_agent.build()
    except Exception as exc:
        st.exception(exc)
        return

    runs1 = int(runs_by_innings.get(1, 0))
    runs2 = int(runs_by_innings.get(2, 0))
    wickets1 = int(wickets_by_innings.get(1, 0))
    wickets2 = int(wickets_by_innings.get(2, 0))
    readiness = wellness_snapshot.get("readiness_mean") if wellness_snapshot else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winner", match_row.get("winner") or "-")
    col2.metric("Innings 1", f"{runs1}/{wickets1}")
    col3.metric("Innings 2", f"{runs2}/{wickets2}")
    monte_msg = "-"
    if monte_stats:
        win1 = monte_stats.get("win_prob_innings1", 0)
        win2 = monte_stats.get("win_prob_innings2", 0)
        monte_msg = f"1st {win1:.0%} / 2nd {win2:.0%}"
    col4.metric("Simulated Win Chance", monte_msg)
    if readiness is not None:
        col4.caption(f"Readiness mean {readiness:.2f}")

    st.markdown("### Mission Brief")
    if supervisor_output.plan_markdown:
        st.markdown(supervisor_output.plan_markdown)
    else:
        st.info(
            "Supervisor plan requires Azure OpenAI credentials (GPT-4.1). Configure AZURE_OPENAI_* settings to unlock the briefing."
        )

    st.markdown("### Strategic Visuals")
    chart_col1, chart_col2 = st.columns(2)

    phase_rows: List[Dict[str, Any]] = []
    for team, phases in (phase_team or {}).items():
        for phase_name, stats in phases.items():
            phase_rows.append(
                {
                    "Phase": phase_name,
                    "Team": team,
                    "RPO": round(stats.get("rpo", 0.0), 2),
                }
            )
    phase_df = pd.DataFrame(phase_rows)
    if not phase_df.empty:
        pivot = phase_df.pivot(index="Phase", columns="Team", values="RPO").fillna(0)
        chart_col1.bar_chart(pivot, height=280)
    else:
        chart_col1.info("Phase tempo data unavailable for this fixture.")

    if not over_summary.empty:
        cumulative_df = over_summary.sort_values(["innings", "over"]).copy()
        cumulative_df["cumulative_runs"] = cumulative_df.groupby("innings")["runs"].cumsum()
        cum_pivot = cumulative_df.pivot(index="over", columns="innings", values="cumulative_runs").ffill()
        chart_col2.line_chart(cum_pivot, height=280)
    else:
        chart_col2.info("Over-by-over scoring unavailable.")

    st.markdown("### Human & Integrity Signals")
    signal_col1, signal_col2 = st.columns(2)

    if wellness_chart_df is not None and not wellness_chart_df.empty:
        risk_series = wellness_chart_df.set_index("name")["risk_score"]
        signal_col1.bar_chart(risk_series, height=260)
        signal_col1.caption("Top wellness risk scores (lower is safer).")
    else:
        signal_col1.info("Wellness risk insights unavailable or not configured.")

    if integrity_pressure_chart is not None and not integrity_pressure_chart.empty:
        pressure_series = integrity_pressure_chart.set_index("window")["pressure_index"]
        signal_col2.bar_chart(pressure_series, height=260)
        signal_col2.caption("Decision-pressure index across key windows.")
    else:
        signal_col2.info("Integrity pressure map unavailable for this match.")

    st.markdown("### Agent Callouts")
    tact_col, perf_col, integ_col, scout_col = st.columns(4)
    tact_col.subheader("Tactical")
    for note in supervisor_output.tactical_callouts:
        tact_col.write(f"- {note}")
    perf_col.subheader("Performance")
    if supervisor_output.wellness_callouts:
        for note in supervisor_output.wellness_callouts:
            perf_col.write(f"- {note}")
    else:
        perf_col.write("- Wellness data unavailable")
    integ_col.subheader("Integrity")
    for note in supervisor_output.integrity_callouts:
        integ_col.write(f"- {note}")
    scout_col.subheader("Scouting")
    for note in supervisor_output.recruiting_prompts:
        scout_col.write(f"- {note}")

    st.markdown("### Coordination Queue")
    if supervisor_output.coordination_queue:
        for item in supervisor_output.coordination_queue:
            st.write(f"- {item}")
    else:
        st.info("No cross-agent escalations flagged.")

    with st.expander("Phase Run Rate Snapshot", expanded=False):
        if phase_team:
            for team, phases in phase_team.items():
                team_df = (
                    pd.DataFrame.from_dict(phases, orient="index")
                    .reset_index()
                    .rename(columns={"index": "Phase"})
                )
                team_df["rpo"] = team_df["rpo"].round(2)
                st.markdown(f"**{team}**")
                st.dataframe(team_df, use_container_width=True, hide_index=True)
        else:
            st.info("Phase data unavailable for this match.")

    if integrity_insights is not None:
        with st.expander("Integrity Hotspots", expanded=False):
            if not integrity_insights.review_hotspots.empty:
                st.dataframe(integrity_insights.review_hotspots, use_container_width=True, hide_index=True)
            else:
                st.info("No concentrated review zones detected.")

    if not innings_summary.empty:
        with st.expander("Innings Overview", expanded=False):
            st.dataframe(innings_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Supervisor Agent decomposes goals, assigns agent playbooks, and synchronises CAIN's perceive→reason→act→learn loop."
    )


def render_adjudication_integrity() -> None:
    st.title("Adjudication & Integrity — Umpire's Assistant")
    st.caption("Objective adjudication intelligence synced with Tactical and Digital Physio agents.")

    if build_integrity_agent is None:
        st.error(
            "Integrity agent pipeline unavailable. Ensure global_scout/src/agents/integrity is installed and dependencies configured."
        )
        st.stop()

    deliveries_df, matches_df, players_map = load_data()
    if matches_df.empty or deliveries_df.empty:
        st.error("Required CSV files are missing or empty. Ensure deliveries.csv and matches.csv are present.")
        return

    st.sidebar.header("Match Selector")
    match_options: List[Tuple[int, str]] = []
    for _, row in matches_df.iterrows():
        match_id_val = safe_int(row.get("match_id"))
        if match_id_val is None:
            continue
        match_options.append((match_id_val, format_match_title(row)))
    if not match_options:
        st.warning("No matches found in matches.csv")
        return

    match_labels = {mid: label for mid, label in match_options}
    selected_match = st.sidebar.selectbox(
        "Choose a match",
        options=[mid for mid, _ in match_options],
        format_func=lambda mid: match_labels.get(mid, str(mid)),
        index=0,
    )

    overs_window = st.sidebar.slider(
        "Integrity window (overs)",
        min_value=2,
        max_value=6,
        value=3,
        help="Rolling window used to surface decision-pressure clusters.",
    )

    match_row = matches_df[matches_df["match_id"] == selected_match].iloc[0]
    match_title = format_match_title(match_row)
    st.subheader(match_title)

    match_deliveries = deliveries_df[deliveries_df["match_id"] == selected_match].copy()
    if match_deliveries.empty:
        st.warning("No deliveries for the selected match.")
        return

    # Normalise wicket type text for downstream logic
    match_deliveries["wicket_type"] = match_deliveries["wicket_type"].fillna("")

    try:
        agent = build_integrity_agent(match_deliveries, match_row, overs_window=overs_window)
        insights = agent.build_insights()
    except Exception as exc:
        st.exception(exc)
        return

    enriched = enrich_ball_by_ball(match_deliveries, players_map)
    enriched["appeal"] = (enriched["wicket_type"].str.strip() != "-").astype(int)
    enriched["lbw"] = enriched["wicket_type"].str.contains("lbw", case=False).astype(int)
    enriched["runout"] = enriched["wicket_type"].str.contains("run out", case=False).astype(int)

    total_appeals = int(enriched["appeal"].sum())
    lbw_total = int(enriched["lbw"].sum())
    runout_total = int(enriched["runout"].sum())
    peak_pressure = float(insights.pressure_windows["pressure_index"].max()) if not insights.pressure_windows.empty else 0.0

    other_appeals = max(total_appeals - lbw_total - runout_total, 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Appeals", total_appeals)
    col2.metric("LBW Decisions", lbw_total)
    col3.metric("Run-out Decisions", runout_total)
    col4.metric("Other Appeals", other_appeals)
    col5.metric("Peak Pressure Index", f"{peak_pressure:.2f}")

    other_breakdown = enriched[(enriched["appeal"] == 1) & (enriched["lbw"] == 0) & (enriched["runout"] == 0)].copy()
    if not other_breakdown.empty:
        other_breakdown["wicket_label"] = (
            other_breakdown["wicket_type"].astype(str).str.replace("_", " ", regex=False).str.title()
        )
        other_breakdown["wicket_label"] = other_breakdown["wicket_label"].replace({"-": "Unspecified Appeal"})
        breakdown_counts = other_breakdown["wicket_label"].value_counts().to_dict()
        breakdown_text = ", ".join(
            f"{label} – {count}"
            for label, count in breakdown_counts.items()
        )
        st.markdown(f"**Other appeals mix:** {breakdown_text}.")

    appeals_by_innings = (
        enriched.groupby("innings")["appeal"].sum().astype(int).to_dict()
    )
    if appeals_by_innings:
        st.markdown("#### Appeals by Innings")
        appeal_cols = st.columns(len(appeals_by_innings))
        for idx, (inn, count) in enumerate(sorted(appeals_by_innings.items())):
            appeal_cols[idx % len(appeal_cols)].metric(f"Innings {inn}", count)

    st.markdown("### Appeal Density by Over")
    per_over = (
        enriched.groupby(["innings", "over"])
        .agg(appeals=("appeal", "sum"), lbw=("lbw", "sum"), runout=("runout", "sum"))
        .reset_index()
    )
    if not per_over.empty:
        appeal_chart = per_over.pivot(index="over", columns="innings", values="appeals").fillna(0)
        appeal_matrix = appeal_chart.to_numpy()
        if appeal_matrix.size > 0 and (appeal_matrix.max() - appeal_matrix.min()) > 0:
            st.bar_chart(appeal_chart)
        else:
            st.info(
                "Appeal counts are uniform across overs for this match; density chart omitted to avoid duplication."
            )
        lbw_chart = per_over.pivot(index="over", columns="innings", values="lbw").fillna(0)
        if lbw_chart.to_numpy().sum() > 0:
            st.line_chart(lbw_chart, height=260)
    else:
        st.info("No appeal events recorded for this fixture.")

    st.markdown("### Decision-Pressure Windows")
    window_block = max(int(overs_window), 1)
    agg_frame = enriched.copy()
    agg_frame["over_int"] = agg_frame["over"].astype(int)
    agg_frame["window_start"] = (agg_frame["over_int"] // window_block) * window_block
    agg_frame["window_end"] = agg_frame["window_start"] + window_block - 1
    window_summary = (
        agg_frame.groupby(["innings", "window_start"])
        .agg(
            window_end=("window_end", "max"),
            total_appeals=("appeal", "sum"),
            lbw=("lbw", "sum"),
            runout=("runout", "sum"),
        )
        .reset_index()
    )
    window_summary["window_span"] = window_summary["window_end"] - window_summary["window_start"] + 1
    window_summary = window_summary[window_summary["window_span"] > 0]
    window_summary["other_appeals"] = (
        window_summary["total_appeals"] - window_summary["lbw"] - window_summary["runout"]
    ).clip(lower=0)
    window_summary["pressure_index"] = (
        (window_summary["total_appeals"] * 1.5 + window_summary["lbw"] * 2.0 + window_summary["runout"] * 1.0)
        / window_summary["window_span"].clip(lower=1)
    )
    window_summary = window_summary[window_summary["total_appeals"] > 0]

    if not window_summary.empty:
        st.markdown(
            f"_Label guide: totals above reflect the full match; table counts show events inside each continuous {window_block}-over window (LBW = leg-before-wicket appeals, Run-out = run-out appeals, Other Appeals = remaining adjudications such as catches, stumpings, hit wicket; 0 means none, 1 one appeal, 2 two appeals, etc.)._"
        )
        window_summary["overs_window"] = window_summary.apply(
            lambda row: f"{int(row['window_start'])}-{int(row['window_end'])}", axis=1
        )
        if insights.review_hotspots.empty:
            hotspots_df = pd.DataFrame(columns=["innings", "overs_window", "integrity_note"])
        else:
            hotspots_df = insights.review_hotspots[["innings", "over", "reason"]].rename(
                columns={"over": "overs_window", "reason": "integrity_note"}
            )

            def normalise_window(window: str) -> str:
                try:
                    start_str, end_str = window.split("-", 1)
                    start_val = int(start_str.strip())
                    end_val = int(end_str.strip())
                except Exception:
                    return window
                aligned_start = (start_val // window_block) * window_block
                aligned_end = max(aligned_start, min(end_val, aligned_start + window_block - 1))
                return f"{aligned_start}-{aligned_end}"

            hotspots_df["overs_window"] = hotspots_df["overs_window"].apply(normalise_window)
            hotspots_df["integrity_note"] = hotspots_df["integrity_note"].fillna("")
            hotspots_df = (
                hotspots_df.groupby(["innings", "overs_window"])["integrity_note"]
                .agg(lambda notes: "; ".join(sorted({note for note in notes if note})))
                .reset_index()
            )

        combined_df = window_summary.merge(hotspots_df, on=["innings", "overs_window"], how="left")
        combined_df = combined_df.rename(
            columns={
                "innings": "Innings",
                "overs_window": "Overs Window",
                "total_appeals": "Total Appeals",
                "lbw": "LBW Appeals",
                "runout": "Run-out Appeals",
                "other_appeals": "Other Appeals",
                "pressure_index": "Pressure Index",
                "integrity_note": "Integrity Note",
            }
        )
        combined_df["Pressure Index"] = combined_df["Pressure Index"].astype(float).round(2)
        combined_df["Integrity Note"] = combined_df["Integrity Note"].fillna("No specific review flag")
        combined_df = combined_df.sort_values(by=["Pressure Index", "Total Appeals"], ascending=[False, False]).head(12)
        st.dataframe(
            combined_df[
                [
                    "Innings",
                    "Overs Window",
                    "Total Appeals",
                    "LBW Appeals",
                    "Run-out Appeals",
                    "Other Appeals",
                    "Pressure Index",
                    "Integrity Note",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Counts in each row are per overs window and align with the match-wide totals shown in the metrics above."
        )
    else:
        st.info("No pressure windows detected.")

    st.markdown("### Integrity Narrative")
    if insights.narrative:
        st.markdown(insights.narrative)
    else:
        st.info(
            "Integrity brief unavailable (Azure OpenAI not configured). Configure AZURE_OPENAI_ENDPOINT/API_KEY for GPT-4.1 narratives."
        )

    st.markdown("### Coordination Signals")
    for alert in insights.alerts:
        st.write(f"- {alert}")

    coordination_notes: List[str] = []
    if lbw_total >= 3:
        coordination_notes.append(
            "Flag Tactical Agent to allocate DRS judiciously against left-hand matchups during highlighted LBW windows."
        )
    if runout_total >= 2:
        coordination_notes.append(
            "Alert Digital Physio to monitor sprint fatigue for batters involved in repeated run-out scenarios."
        )
    if peak_pressure >= 2.0:
        coordination_notes.append(
            "Coordinate with Tactical bench to steady over-rate and reduce appeal clustering in death overs."
        )

    if coordination_notes:
        st.markdown("**Agent Sync Suggestions**")
        for note in coordination_notes:
            st.write(f"- {note}")

    st.markdown("### Appeal Ledger")
    appeals_log = enriched[enriched["appeal"] == 1][
        ["innings", "over", "ball", "phase", "batter", "bowler", "wicket_type", "dismissed"]
    ]
    if not appeals_log.empty:
        st.dataframe(appeals_log, use_container_width=True, hide_index=True)
    else:
        st.info("No wicket appeals registered in this match snapshot.")

    st.markdown("---")
    st.caption(
        "The Umpire's Assistant cross-references Tactical simulations and Digital Physio risk alerts to maintain integrity with human oversight."
    )


if __name__ == "__main__":
    main()
