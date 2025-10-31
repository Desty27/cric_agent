# CAIN Agent Command Deck — Detailed README

This README distils the **Cricketing Agentic Intelligence Nexus (CAIN)** vision from `pipeline.txt` into the practical implementation that lives inside this repository. It walks through the five Streamlit agents that currently compose the control room, shows how the code stitches data and models together, explains every metric/visual in the UI, and clarifies how the agents collaborate to automate high-value workflows while keeping humans in the loop.

> **Agents covered**
>
> 1. Supervisor Command Deck (Head Coach)  
> 2. Tactical & Simulation Command Center (Strategist)  
> 3. Performance & Wellness Command (Digital Physio)  
> 4. Global Scout (Chief Scout)  
> 5. Adjudication & Integrity (Umpire’s Assistant)

All code references below point to the live implementation in `streamlit_app.py` and the supporting modules under `global_scout/src` and `src/`.

---

## 1. Supervisor Command Deck — “Head Coach” Agent

**Primary role.** Decomposes the coach’s high-level objective into tactical, performance, integrity, and scouting actions. It synthesises all downstream agent outputs into a single mission briefing and coordination queue.

**Key modules.**

- `streamlit_app.py::render_supervisor_command()` — orchestrates data fetch, feature extraction, charts, and UI layout.
- `global_scout/src/agents/supervisor/supervisor_agent.py` — packages structured snapshots and calls Azure OpenAI (GPT‑4.1) via `generate_supervisor_plan()` in `global_scout/src/services/azure_openai.py`.

### 1.1 Data & modelling pipeline

1. **Match context ingestion.** Uses `load_data()` to cache `deliveries.csv`, `matches.csv`, `players.csv`. The selected fixture is converted into an internal `context` dict with teams, venue, and historical outcome.
2. **Tactical snapshot.** Calls `compute_phase_run_rates_per_team`, `compute_wicket_clusters`, `compute_per_bowler_wicket_clusters`, and `monte_carlo_simulation` from `src/analyzer.py` to summarise scoring tempo, collapse windows of wickets, and estimate win probabilities (Monte Carlo simulation uses weighted random sampling across remaining overs).
3. **Performance snapshot.** If wellness data are available, `global_scout/src/pipelines/wellness.py::compute_injury_risk` produces a risk/readiness score. This function emulates an XGBoost-style ensemble by mixing weighted features (acute:chronic load, soreness, sleep, injury history, etc.).
4. **Integrity snapshot.** `global_scout/src/agents/integrity/adjudication_agent.py` (the same engine used in the dedicated integrity view) computes a pressure index by sliding three-over windows and tagging LBW/run-out clusters.
5. **Scouting snapshot.** If the Global Scout tab exported `candidate_rankings.csv`, the Supervisor reads the top three names as recruitment prompts.
6. **Mission plan (LLM).** `generate_supervisor_plan()` sends the above context to Azure OpenAI (default deployment `gpt-4.1`, `2024-12-01-preview`). The returned Markdown follows the requested headings: Goal Alignment, Tactical Priorities, Human Performance, Integrity Watch, Recruitment Lens.

### 1.2 Streamlit UI walkthrough

| Section | What you see | How to read it |
| --- | --- | --- |
| **Header Metrics** (`col1`–`col4`) | Winner, Innings 1/2 scorecards, Sim Win Prob | Summaries drawn from `matches.csv` and Monte Carlo output. Sim Win Prob shows *“1st x% / 2nd y%”* confidence for each batting order. If wellness data exists, the caption displays squad readiness (0–1).
| **Mission Brief** | Markdown directly from GPT‑4.1 | The Supervisor fuses all agent signals here. Bullet counts are capped (≤2 per heading) for quick briefing.
| **Strategic Visuals (left)** | Bar chart of RPO vs Phase vs Team | X-axis: phases (`Powerplay`, `Middle`, `Death`). Y-axis: runs per over. Bars are grouped by team, highlighting which innings-phase combinations require attention.
| **Strategic Visuals (right)** | Line chart of Cumulative Runs vs Over per innings | Shows scoreboard acceleration. X-axis: over number; Y-axis: cumulative runs. Each line represents an innings, enabling cross-innings tempo comparison.
| **Human & Integrity Signals (left)** | Bar chart of top eight `risk_score`s | X-axis: player names; Y-axis: injury risk score (0 “safe” → 1 “critical”). Derived from the weighted risk model. Highlights who requires load management.
| **Human & Integrity Signals (right)** | Bar chart of integrity pressure indices | X-axis: window label (`I<innings> start-end`). Y-axis: pressure index (appeals × weight). Helps the head coach partner with officiating staff.
| **Agent Callouts** | Four sub-columns (Tactical, Performance, Integrity, Scouting) | Short bullet lists computed in `SupervisorAgent._derive_*`. They surface the single highest-leverage note per discipline.
| **Coordination Queue** | Ordered bullet list | Indicates the immediate follow-up conversations the head coach should trigger (e.g., “Synchronise Tactical Agent on …”).
| **Expanders** | Phase run rate tables, Integrity hotspot table, Innings overview | Provide raw data behind the visuals so analysts can audit the LLM-level synthesis.

### 1.3 Automation & recommendation flow

1. **Goal input** happens implicitly when the coach selects the match and Monte Carlo intensity.  
2. Data snapshots are computed locally (pandas + helper functions).  
3. The Supervisor Agent class merges heuristics (e.g., lowest RPO phase) with LLM reasoning to contextualise numbers.  
4. Output surfaces as the mission brief, callouts, and coordination queue — the precise form of the “delegate → supervise” loop described in `pipeline.txt`.

Fail-safes: if OpenAI credentials are missing, Streamlit highlights which sections degrade (“Supervisor plan requires Azure OpenAI credentials”). Local tables remain available, so the human can still act manually.

---

## 2. Tactical & Simulation Command Center — “Strategist” Agent

**Primary role.** Deliver ball-by-ball tactical intelligence: tempo trends, wicket clusters, Monte Carlo scenarios, and auto-generated recommendations.

**Key modules.**

- `streamlit_app.py::main()` (default view)  
- `src/analyzer.py` for all statistical computations.

### 2.1 Data & analytics stack

1. **Phase run rates.** `compute_phase_run_rates` and `compute_phase_run_rates_per_team` bucket overs into Powerplay (0–5), Middle (6–14), Death (15+). Each entry stores runs, balls, and RPO.
2. **Wicket clusters.** `compute_wicket_clusters` slides a six-over window for each innings and tallies dismissals; results sorted by wickets descending.
3. **Bowler hot zones.** `compute_per_bowler_wicket_clusters` replicates cluster logic per bowler to identify spells that generated wickets.
4. **Monte Carlo simulation.** `monte_carlo_simulation` performs hundreds to thousands of trials (configurable via sidebar slider). For each trial it randomises remaining overs using historical run distributions to produce win probabilities and score distributions.
5. **Player enrichment.** `enrich_ball_by_ball` maps IDs to names (via `players.csv`), labels phases, and normalises wicket text.

### 2.2 Streamlit UI walkthrough

- **Top metrics row.**
  - *Winner:* from `matches.csv`.  
  - *Monte Carlo Trials:* direct reflection of the slider.  
  - *Avg Run Rate:* arithmetic mean of innings run rates.  
  - *Prediction Accuracy:* compares Monte Carlo favourite (innings 1 vs 2) with actual winner; shows `N/A` if insufficient data.
- **Innings Overview cards.** Each card displays `runs/wickets`, `RR` delta, overs text (e.g., `19.5`), and boundary count.
- **Phase-wise Run Rates (expander).** First panel renders a Matplotlib bar chart (`plot_phase_run_rates_per_team`); second panel lists raw numbers with columns: `Phase`, `runs`, `balls`, `rpo`.
- **Run Distribution by Over (bar chart).** X-axis: over number. Y-axis: runs scored in that over. Bars coloured by innings (Streamlit automatically uses the multi-index pivot).
- **High-Impact Overs list.** Textual bullet per top-5 `runs` entries, including wickets in that over.
- **Wicket Clusters table.** Columns: `innings`, `start_over`, `end_over`, `wickets`. Highlight windows to replicate or avoid.
- **Bowler Hot Zones table.** Adds `bowler_name` field, enabling captain to plan spells around proven wicket bursts.
- **Monte Carlo Snapshot table.** Displays win probabilities (percentage strings) and summary stats (mean/median scores) for both innings.
- **Ball-by-Ball Ledger.** 11 columns from `enrich_ball_by_ball` including `phase`, `batter`, `bowler`, `runs_total`, `wicket_type`, `dismissed`.
- **CAIN Tactical Recommendations.** Generated by `generate_tactical_recommendations()`, mixing heuristics (low RPO triggers, wicket clusters) with pipeline-inspired phrasing (“**Tactical Agent**: …”).

### 2.3 Recommendation logic & automation

- When RPO < 6 in Powerplay, the agent recommends aggressive tactics (pinch hitters).  
- Low Middle overs RPO prompts practice suggestions from the Performance agent (“Schedule mid-innings batting scenarios…”).  
- Wicket clusters produce fielding adjustments.  
- Monte Carlo output informs bowling changes (favour innings with higher win probability).  

The Strategist runs continuously in the Streamlit session, so coaches can adjust filters (innings, Monte Carlo trials) and immediately see how the outputs — and thus the tactical note stack — evolve.

---

## 3. Performance & Wellness Command — “Digital Physio” Agent

**Primary role.** Translate raw wellness metrics into actionable readiness guidance and cross-agent coordination cues.

**Key modules.**

- `streamlit_app.py::render_performance_wellness()`  
- `global_scout/src/pipelines/wellness.py`

### 3.1 Data features & models

`compute_injury_risk()` assembles a composite risk score (0–1) using weighted components:

- **Load-based features:** acute load, chronic load, acute:chronic ratio.  
- **Wellness signals:** soreness, sleep hours, recovery index, wellness survey.  
- **Exposure metrics:** bowling overs, batting balls, sprint sessions, travel hours.  
- **History:** prior injuries.  

Weights mimic feature importances from ensemble models like XGBoost (documented in `pipeline.txt`). `summarize_wellness()` aggregates average readiness and high/moderate risk percentages.  `generate_coach_guidance()` bundles a roster snapshot and calls Azure OpenAI via `generate_medical_coordination_notes()` to produce three JSON instructions (team, focus, medical action, tactical adjustment).

### 3.2 Streamlit UI walkthrough

1. **Data source selector.** Choose between bundled demo data (`global_scout/data/wellness_demo.csv`) or a custom CSV. The schema checker validates required columns before processing.
2. **Filters.** Multi-select for `team`, `role`, and a readiness threshold slider. These filters propagate to the tables and charts.
3. **Metric strip.**
	- *Mean Risk:* average `risk_score` for the filtered set.  
	- *Average Readiness:* inverse (1 − risk).  
	- *% High Risk / % Moderate Risk:* share of players with scores ≥0.7 or ≥0.45 respectively.
4. **Readiness Snapshot table.** Columns include `risk_level`, `primary_driver` (top contributing feature), and recommended intervention (e.g., “Flag for medical review; cap spells at 2 overs”).
5. **Charts.**
	- Left bar chart: X-axis player names, Y-axis risk score.  
	- Right scatter: X-axis AC Ratio, Y-axis readiness, colour-coded by team; reveals overtraining clusters.
6. **Workload Lens table.** Shows underlying load metrics enabling S&C staff to verify the risk classification.
7. **Medical & Tactical Coordination bullets.** LLM-generated lines referencing match context (if selected). Example output: `[England] Manage Ben Stokes — Medical: limit neuromuscular work. Tactical: adjust death overs rotation.`

### 3.3 Automation

- Data ingestion can be scheduled (CSV drop or API feed).  
- Risk computation is deterministic; the guidance layer integrates with GPT‑4.1 for crisp game-plan language.  
- The agent automatically highlights top high-risk names and systemic drivers (“primary strain driver: acute_ratio”).  
- Supervisor consumes these outputs to balance tactical aggression with player health.

---

## 4. Global Scout — “Chief Scout” Agent

**Primary role.** Objectively rank talent while enforcing fairness constraints, so recruitment decisions align with long-term strategy.

**Key modules.**

- `streamlit_app.py::render_global_scout()`  
- `global_scout/src/pipelines/rank_candidates.py` (imported as `score_candidates`, `mitigate_and_rank`)

### 4.1 Data & algorithms

1. **Scoring.** `score_candidates()` computes role-aware performance scores (e.g., batters: strike rate & average; bowlers: economy & average) plus contextual factors (league level, matches played, recent form).
2. **Fairness mitigation.** `mitigate_and_rank()` normalises within protected groups (default `region`, optionally any categorical column) and applies a shortlist quantile filter. It also calculates demographic parity and equal opportunity gaps, aligning with fairness techniques described in `pipeline.txt`.
3. **Ranking.** Final score scaled 0–1 with `shortlisted` flag for top candidates.

### 4.2 Streamlit UI walkthrough

- **Data source block.** Toggle between bundled demo data or upload. The preview expander shows the first 20 rows to confirm mapping.
- **Fairness Audits.** Metrics display the protected attribute name, demographic parity delta, and equal opportunity delta; both aim for 0. The optional group stats table lists pass rates per subgroup.
- **Ranked Candidates table.** Columns include `rank`, `final_score`, `performance_score`, `fit_score`, and the protected attribute. Sorting lets scouts adjust for specific traits.
- **Download button.** Exports the exact ranking used elsewhere (i.e., by Supervisor in the Recruiting Lens section).

### 4.3 Automation & recommendations

- Once connected to live feeds (league APIs, video-derived metrics), the agent can re-run scoring nightly.  
- Fairness guardrails ensure the shortlist stays compliant.  
- Supervisor reads top-three names to integrate with tactical/physio needs (e.g., avoid signing a high-risk bowler when physio flags workload issues).

---

## 5. Adjudication & Integrity — “Umpire’s Assistant” Agent

**Primary role.** Support officials with objective data and identify integrity red flags during a match.

**Key modules.**

- `streamlit_app.py::render_adjudication_integrity()`  
- `global_scout/src/agents/integrity/adjudication_agent.py`

### 5.1 Data features & models

1. **Appeal classification.** Every delivery is enriched with boolean flags for `appeal`, `lbw`, `runout` using string checks on `wicket_type`.
2. **Pressure windows.** `_compute_pressure_windows()` slides a three-over window, summing `runs_total`, `appeals`, `lbw`, `runout`. Pressure index = `(1.5 × appeals + 2 × lbw + 1 × runout) / window_size`.
3. **Review hotspots.** `_compute_review_hotspots()` focuses on top five pressure windows, annotates reason (“LBW concentration” vs “Run-out pressure”).
4. **Alerts.** `_identify_alerts()` emits human-readable messages for thresholds (≥3 LBWs, ≥2 run-outs, ≥4 appeals per window).
5. **Integrity brief (LLM).** `generate_integrity_brief()` sends match meta + JSON payloads to GPT‑4.1 for a Markdown summary with sections: Situation Snapshot, High-Risk Windows, Coordination Signals.

### 5.2 Streamlit UI walkthrough

| Component | Meaning |
| --- | --- |
| **Metric strip** | `Total Appeals`, `LBW Decisions`, `Run-out Decisions`, `Peak Pressure Index`. Pressure Index > 2 indicates stacked reviews needing umpire awareness.
| **Appeals by Innings** | Small metrics per innings to separate fielding vs batting side pressure.
| **Appeal Density by Over (bar chart)** | X-axis: over; Y-axis: total appeals in that over. Chart is suppressed if counts are uniform to avoid false signals.
| **LBW trend (line chart)** | Plots LBW counts per over per innings (only shown when data exist). Helps track whether LBW pressure rises in the death overs.
| **Decision-Pressure Windows table** | Columns: `innings`, `start_over`, `end_over`, `appeals`, `lbw`, `runout`, `pressure_index`. Values stem from the sliding window logic; units are counts.
| **Review Hotspots table** | Adds `over` range string, LBW/run-out counts, total appeals, and dominant reason. Guides third-umpire reviews.
| **Integrity Narrative** | GPT‑4.1 Markdown summarising the above data in plain language, ready to send to officiating staff.
| **Coordination Signals & Agent Sync Suggestions** | Combines deterministic alerts with cross-agent hooks (e.g., inform Physio if run-out fatigue detected).
| **Appeal Ledger table** | Lists every appealed delivery with columns `innings`, `over`, `ball`, `phase`, `batter`, `bowler`, `wicket_type`, `dismissed`.

### 5.3 Automation

- Integrates the same match data as Tactical/Physio to avoid silos.  
- Alerts trigger automatically when thresholds trip.  
- The integrity brief uses GPT‑4.1 so the officiating summary remains concise and human-readable.  
- Supervisor ingests alerts (`SupervisorOutputs.integrity_callouts`) to balance tactical aggression with sportsmanship.

---

## Frequently Asked Questions

**What machine-learning models are in play?**

- Injury risk: weighted ensemble approximating XGBoost features (see `compute_injury_risk`).
- Tactical Monte Carlo: stochastic simulation of innings built on historic run distributions; while not a classifier, it mirrors the Monte Carlo approach described in `pipeline.txt`.
- Fairness mitigation: statistical parity metrics (demographic parity, equal opportunity) align with algorithmic auditing best practices.
- GPT‑4.1 (Azure OpenAI) provides natural-language reasoning for Supervisor, Physio, and Integrity agents, turning numeric insights into actionable guidance.

**How do the agents collaborate?**

- Supervisor consumes outputs from Tactical (phase RPO, Monte Carlo), Performance (risk summaries + LLM notes), Integrity (alerts + narrative), and Global Scout (shortlist names).  
- Tactical recommendations explicitly call on Performance and Commercial agents when thresholds breach.  
- Integrity agent feeds alerts into Supervisor coordination queue, prompting conversations with Tactical/Physio.  
- Physio guidance references tactical adjustments (“limit death-over spell to 2 overs”), closing the loop between human performance and match strategy.

**What remains manual?**

- Configuring Azure OpenAI credentials (set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` in `.env`).
- Uploading bespoke wellness or scouting datasets.  
- Deciding which agent recommendations to action in the real world — CAIN is decision-support, not decision-replacement.

---

## Next Steps

- **Data expansion.** Connect live telemetry (Hawk-Eye, wearables) to replace CSV ingestion.  
- **Model refinement.** Swap heuristic weights for trained gradient boosting models, extend Monte Carlo with bowler–batter matchup matrices, and embed anomaly detection for integrity (e.g., Isolation Forest).  
- **Fan & Commercial agent.** Not yet surfaced in the Streamlit UI; future work will extend the sidebar with personalised fan analytics per the blueprint in `pipeline.txt`.

By aligning the codebase with the architectural principles in `pipeline.txt`, this command deck demonstrates how agentic AI can perceive, reason, act, and learn in cricket without overwhelming coaches with raw numbers.
