#!/usr/bin/env python3
"""
Generate sample cricket data: players.csv, matches.csv, deliveries.csv and biometric_data.csv
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
NUM_PLAYERS = 50
NUM_MATCHES = 10
DELIVERIES_PER_MATCH = 120 * 2  # approx 120 balls per innings * 2 innings

TEAMS = ['Australia', 'England', 'India', 'New Zealand', 'South Africa', 'West Indies']
PLAYER_NAMES = ['Smith', 'Warner', 'Labuschagne', 'Cummins', 'Starc', 'Root', 'Bairstow', 'Stokes', 'Archer', 'Buttler',
                'Kohli', 'Sharma', 'Bumrah', 'Pant', 'Ashwin', 'Williamson', 'Boult', 'Southee', 'Conway', 'Rabada',
                'De Kock', 'Maharaj', 'Ngidi', 'Pollard', 'Hope', 'Holder', 'Russell', 'Gayle', 'Finch', 'Carey',
                'Paine', 'Sams', 'Marsh', 'Green', 'Agar', 'Khawaja', 'Billings', 'Ali', 'Woakes', 'Wood', 'Malan',
                'Curran', 'Vihari', 'Iyer', 'Kishan', 'Siraj', 'Shami', 'Jadeja', 'Chahal', 'Kuldeep']
BATTING_STYLES = ['Right-handed', 'Left-handed']
BOWLING_STYLES = ['Right-arm fast', 'Right-arm medium', 'Left-arm fast', 'Left-arm orthodox', 'Leg spin', 'Off spin', None]
VENUES = ['MCG', 'SCG', 'The Oval', 'Lords', 'Eden Gardens', 'Wankhede', 'Basin Reserve', 'Newlands']
WICKET_TYPES = ['bowled', 'caught', 'lbw', 'run out', 'stumped', 'hit wicket', None]

random.seed(42)

# --- Generate Players ---
players = []
for i in range(NUM_PLAYERS):
    player_id = i + 1
    team = random.choice(TEAMS)
    if i < 10:
        team = TEAMS[0]
    elif i < 20:
        team = TEAMS[1]
    players.append({
        'player_id': player_id,
        'full_name': f"{random.choice(PLAYER_NAMES)} P{player_id}",
        'team': team,
        'batting_style': random.choice(BATTING_STYLES),
        'bowling_style': random.choice(BOWLING_STYLES)
    })

players_path = OUT_DIR / 'players.csv'
with open(players_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['player_id', 'full_name', 'team', 'batting_style', 'bowling_style']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(players)
print(f"Generated {len(players)} players -> {players_path}")

# --- Generate Matches ---
matches = []
start_date = datetime(2023, 1, 1)
for i in range(NUM_MATCHES):
    match_id = i + 1
    date = start_date + timedelta(days=i*7)
    team_a, team_b = random.sample(TEAMS, 2)
    toss_winner = random.choice([team_a, team_b])
    toss_decision = random.choice(['bat', 'field'])
    winner = toss_winner if random.random() < 0.55 else (team_b if toss_winner == team_a else team_a)
    matches.append({
        'match_id': match_id,
        'season': 2023,
        'date': date.strftime('%Y-%m-%d'),
        'venue': random.choice(VENUES),
        'team_a': team_a,
        'team_b': team_b,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'winner': winner
    })

matches_path = OUT_DIR / 'matches.csv'
with open(matches_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['match_id', 'season', 'date', 'venue', 'team_a', 'team_b', 'toss_winner', 'toss_decision', 'winner']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(matches)
print(f"Generated {len(matches)} matches -> {matches_path}")

# --- Generate Deliveries ---
deliveries = []
delivery_id = 1

for match in matches:
    match_id = match['match_id']
    team_a_players = [p for p in players if p['team'] == match['team_a']]
    team_b_players = [p for p in players if p['team'] == match['team_b']]

    # If a team has fewer than 6 players (unlikely), fallback to any players
    if len(team_a_players) < 6:
        team_a_players = random.sample(players, 11)
    if len(team_b_players) < 6:
        team_b_players = random.sample(players, 11)

    for innings in [1, 2]:
        balls_per_innings = DELIVERIES_PER_MATCH // 2
        team_batting = team_a_players if innings == 1 else team_b_players
        team_bowling = team_b_players if innings == 1 else team_a_players

        batter_id = random.choice(team_batting)['player_id']
        non_striker_id = random.choice([p for p in team_batting if p['player_id'] != batter_id])['player_id']
        bowler_id = random.choice(team_bowling)['player_id']

        for ball in range(balls_per_innings):
            over = ball // 6
            ball_in_over = (ball % 6) + 1

            runs_off_bat = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[60, 20, 10, 2, 15, 1, 5], k=1)[0]
            extras = random.choices([0, 1, 2], weights=[90, 7, 3], k=1)[0]
            is_wicket = random.random() < 0.05

            wicket_type = None
            dismissed_batter_id = None

            if is_wicket:
                wicket_type = random.choice(WICKET_TYPES)
                if wicket_type:
                    dismissed_batter_id = batter_id
                    new_batter_candidates = [p for p in team_batting if p['player_id'] not in [batter_id, non_striker_id]]
                    if new_batter_candidates:
                        batter_id = random.choice(new_batter_candidates)['player_id']

            if ball % 36 == 0 and ball > 0:
                new_bowler_candidates = [p for p in team_bowling if p['player_id'] != bowler_id and p['bowling_style'] is not None]
                if new_bowler_candidates:
                    bowler_id = random.choice(new_bowler_candidates)['player_id']

            if runs_off_bat in [1, 3, 5]:
                batter_id, non_striker_id = non_striker_id, batter_id

            deliveries.append({
                'delivery_id': delivery_id,
                'match_id': match_id,
                'innings': innings,
                'over': over,
                'ball': ball_in_over,
                'batter_id': batter_id,
                'bowler_id': bowler_id,
                'non_striker_id': non_striker_id,
                'runs_off_bat': runs_off_bat,
                'extras': extras,
                'wicket_type': wicket_type,
                'dismissed_batter_id': dismissed_batter_id
            })
            delivery_id += 1

deliveries_path = OUT_DIR / 'deliveries.csv'
with open(deliveries_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['delivery_id', 'match_id', 'innings', 'over', 'ball', 'batter_id', 'bowler_id', 'non_striker_id', 'runs_off_bat', 'extras', 'wicket_type', 'dismissed_batter_id']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deliveries)
print(f"Generated {len(deliveries)} deliveries -> {deliveries_path}")

# --- Generate Biometric Data (sample) ---
biometric_data = []
timestamp = datetime(2023, 5, 15, 13, 0, 0)
for delivery in deliveries[:500]:
    for player_id in [delivery['batter_id'], delivery['non_striker_id'], delivery['bowler_id']]:
        timestamp += timedelta(seconds=random.uniform(0.5, 3.0))
        biometric_data.append({
            'timestamp': timestamp.isoformat() + 'Z',
            'player_id': player_id,
            'heart_rate_bpm': random.randint(80, 170),
            'speed_kmh': round(random.uniform(0, 35), 1),
            'total_distance_m': random.randint(0, 20000),
        })

biometric_path = OUT_DIR / 'biometric_data.csv'
with open(biometric_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['timestamp', 'player_id', 'heart_rate_bpm', 'speed_kmh', 'total_distance_m']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(biometric_data)
print(f"Generated {len(biometric_data)} biometric data points -> {biometric_path}")

print("\nSample Data Generation Complete!")
print(f"Files created: '{players_path.name}', '{matches_path.name}', '{deliveries_path.name}', '{biometric_path.name}'")
print(f"Total deliveries: {len(deliveries)}")
