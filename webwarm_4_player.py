import requests
import pandas as pd
import os
from tqdm import tqdm

# --- Config ---
START_YEAR = 2016
END_YEAR = 2025
MAX_PLAYERS = 2000
API_URL = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"

# --- Fetch API Data ---
def fetch_stats(season, group):
    params = {
        "env": "prod",
        "season": str(season),
        "stats": "season",
        "group": group,
        "gameType": "R",
        "playerPool": "ALL",
        "limit": MAX_PLAYERS,
        "offset": 0,
        "sortStat": "battingAverage" if group == "hitting" else "era",
        "order": "desc"
    }
    res = requests.get(API_URL, params=params)
    res.raise_for_status()
    return pd.DataFrame(res.json()["stats"])

# --- Main Loop ---
for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="📅 Processing seasons"):
    try:
        # Fetch hitters & pitchers
        hit_df = fetch_stats(year, "hitting")
        pitch_df = fetch_stats(year, "pitching")

        # Clean data
        hit_df["atBats"] = pd.to_numeric(hit_df["atBats"], errors="coerce")
        pitch_df["inningsPitched"] = pd.to_numeric(pitch_df["inningsPitched"], errors="coerce")
        pitch_df["strikeOuts"] = pd.to_numeric(pitch_df["strikeOuts"], errors="coerce")

        hit_df.dropna(subset=["teamAbbrev", "atBats"], inplace=True)
        pitch_df.dropna(subset=["teamAbbrev"], inplace=True)

        # Initialize team_data dict (optional for in-memory use)
        team_data = {}

        all_teams = set(hit_df["teamAbbrev"].unique()) | set(pitch_df["teamAbbrev"].unique())

        for team in all_teams:
            # --- Top 9 Hitters ---
            team_hit = hit_df[hit_df["teamAbbrev"] == team]
            top_hit = team_hit.sort_values(by="atBats", ascending=False).head(9)[[
                "playerFullName", "teamAbbrev", "atBats", "avg", "obp", "slg", "ops"
            ]].rename(columns={
                "avg": "AVG", "obp": "OBP", "slg": "SLG", "ops": "OPS"
            })

            # --- Top 9 Pitchers ---
            team_pitch = pitch_df[pitch_df["teamAbbrev"] == team]
            top_pitch = team_pitch.sort_values(by="strikeOuts", ascending=False).head(10)[[
                "playerFullName", "teamAbbrev", "inningsPitched", "era", "whip", "strikeOuts", "wins"
            ]].rename(columns={
                "inningsPitched": "IP", "era": "ERA", "whip": "WHIP",
                "strikeOuts": "SO", "wins": "W"
            })

            # Save to ./mlb_stats_full/{year}/{team}/
            team_dir = f"mlb_stats_full/{year}/{team}"
            os.makedirs(team_dir, exist_ok=True)

            top_hit.to_csv(f"{team_dir}/hitters.csv", index=False)
            top_pitch.to_csv(f"{team_dir}/pitchers.csv", index=False)

            # Store in memory (optional)
            team_data[team] = {
                "hitters": top_hit,
                "pitchers": top_pitch
            }

            print(f"✅ Saved: {year}/{team}/hitters.csv & pitchers.csv")

    except Exception as e:
        print(f"❌ Error in {year}: {e}")

print("🎯 All data saved by year and team in separate folders.")








