import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
score_candidates = None  # type: ignore
mitigate_and_rank = None  # type: ignore
load_wellness_data = None  # type: ignore
compute_wellness_risk = None  # type: ignore
summarize_wellness = None  # type: ignore
generate_wellness_guidance = None  # type: ignore
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
except Exception:
    pass


st.set_page_config(
    page_title="CAIN Strategist Control Room",
    page_icon="🏏",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
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
                    f"**Tactical Agent**: Powerplay strike rate for {team} is {rpo:.2f}. Promote intent with pinch-hitter drills or staggered batting orders."
                )
            if phase_name.startswith("Middle") and rpo < 6:
                notes.append(
                    f"**Performance & Wellness Agent**: Schedule mid-innings batting scenarios for {team} to unlock rotation of strike; current RPO {rpo:.2f}."
                )
            if phase_name.startswith("Late") and rpo > 10:
                notes.append(
                    f"**Commercial Agent**: Highlight {team}'s finishing burst (RPO {rpo:.2f}) in fan engagement reels for the upcoming fixture."
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
            f"**Simulation Agent**: Monte Carlo suggests innings 1 win probability {win1:.0%} vs innings 2 {win2:.0%}; adjust bowling changes accordingly."
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
    enriched["wicket_type"] = enriched["wicket_type"].fillna("-")
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
    date = row.get("date", "")
    return f"{team_a} vs {team_b} — {venue} ({date})"


def main() -> None:
    # Sidebar view selector first so Global Scout can work even if cricket CSVs are missing
    view = st.sidebar.radio(
        "View",
        [
            "Tactical Command Center",
            "Global Scout",
            "Performance & Wellness",
        ],
        index=0,
        help="Switch between match analysis and scouting/recruitment",
    )

    if view == "Global Scout":
        render_global_scout()
        return
    if view == "Performance & Wellness":
        render_performance_wellness()
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

    monte_trials = st.sidebar.slider("Monte Carlo trials", min_value=200, max_value=2000, value=800, step=200)

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
    wicket_clusters = compute_wicket_clusters(deliveries_records, context)
    bowler_clusters = compute_per_bowler_wicket_clusters(deliveries_records, context)
    monte_stats = monte_carlo_simulation(deliveries_records, context, trials=monte_trials)

    innings_summary = compute_innings_summary(match_deliveries)
    over_summary = compute_over_summary(match_deliveries)
    enriched_balls = enrich_ball_by_ball(match_deliveries, players_map)

    # --- Key metrics row ---
    col1, col2, col3, col4 = st.columns(4)
    winner = match_row.get("winner", "-")
    col1.metric("Winner", winner or "-", help="Result from matches.csv")
    col2.metric("Monte Carlo Trials", monte_stats.get("trials", 0))
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
        else:
            st.info("No phase data available for this match.")

        if phase_overall:
            phase_df = (
                pd.DataFrame.from_dict(phase_overall, orient="index")
                .reset_index()
                .rename(columns={"index": "Phase"})
            )
            phase_df["rpo"] = phase_df["rpo"].round(2)
            st.dataframe(phase_df, use_container_width=True)

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
    if wicket_clusters:
        clusters_df = pd.DataFrame(wicket_clusters)[["innings", "start_over", "end_over", "wickets"]]
        st.dataframe(clusters_df, use_container_width=True)
    else:
        st.info("No wicket clusters identified.")

    st.markdown("### Bowler Hot Zones")
    if bowler_clusters:
        bowler_df = pd.DataFrame(bowler_clusters)[["bowler_name", "innings", "start_over", "end_over", "wickets"]]
        st.dataframe(bowler_df, use_container_width=True)
    else:
        st.info("No bowler hot zones detected.")

    st.markdown("### Monte Carlo Simulation Snapshot")
    if monte_stats and monte_stats.get("trials", 0):
        monte_df = pd.DataFrame(
            {
                "Metric": [
                    "Win Prob Innings 1",
                    "Win Prob Innings 2",
                    "Tie Probability",
                    "Innings 1 Mean",
                    "Innings 2 Mean",
                    "Innings 1 Median",
                    "Innings 2 Median",
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
    if audits.get("group_stats"):
        st.dataframe(pd.DataFrame(audits["group_stats"]), use_container_width=True, hide_index=True)

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
    col_m1.metric("Mean Risk", f"{summary['mean_risk']:.2f}")
    col_m2.metric("Average Readiness", f"{summary['readiness_mean']:.2f}")
    col_m3.metric("% High Risk", f"{summary['high_risk_pct']:.0%}")
    col_m4.metric("% Moderate Risk", f"{summary['moderate_risk_pct']:.0%}")

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
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        viz_cols = st.columns(2)
        with viz_cols[0]:
            chart_df = filtered.set_index("name")["risk_score"].sort_values(ascending=False)
            st.bar_chart(chart_df, height=260)
        with viz_cols[1]:
            scatter_df = filtered[["acute_chronic_ratio", "readiness", "name", "team"]]
            scatter_df = scatter_df.rename(columns={"acute_chronic_ratio": "AC Ratio"})
            st.scatter_chart(scatter_df, x="AC Ratio", y="readiness", color="team", size=None)

    st.markdown("#### Workload Lens")
    workload_cols = ["name", "acute_load", "chronic_load", "bowling_overs_last_7d", "batting_balls_last_7d"]
    available_workload_cols = [c for c in workload_cols if c in risk_df.columns]
    if available_workload_cols:
        st.dataframe(risk_df[available_workload_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Medical & Tactical Coordination")
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


if __name__ == "__main__":
    main()
