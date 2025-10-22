# CAIN Agents Overview

This summary aligns with the provided pipeline description and clarifies each agent's role for the PoC.

## Supervisor Agent (Head Coach)
- Orchestrates goals and delegates tasks to specialists.
- Synthesizes cross-agent outputs into a single plan.

## Performance & Wellness Agent (Physio & Skills Coach)
- Predictive injury prevention (e.g., XGBoost), biomechanical analysis (CNNs).
- Live biometric monitoring for dynamic workload management.

## Tactical & Simulation Agent (Strategist)
- Pre/post-match simulations, dynamic win probability during matches.
- Provides tactical recommendations (field, bowling changes, batting order).
- This PoC implements a minimal Strategist with local analysis and optional AutoGen demo.

## Scouting & Recruitment Agent (Chief Scout)
- Global data/video scanning; predictive performance and strategic fit.

## Fan & Commercial Agent (Marketing Director)
- Personalization, dynamic pricing, and stadium operations optimization.

## Adjudication & Integrity Agent (Third Umpire's Assistant)
- Real-time CV for officiating decisions and anomaly detection for integrity.

## Roadmap from PoC
- Extend Strategist with Monte Carlo simulations and matchup models.
- Integrate Performance agent for fatigue-aware tactical planning.
- Supervisor to coordinate multi-agent workflows.
