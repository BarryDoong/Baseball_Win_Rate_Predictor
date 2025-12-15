import requests
import pandas as pd
import os
from tqdm import tqdm

# ---------- CONFIG ----------
START_YEAR = 2000
END_YEAR = 2025
SAVE_DIR = "mlb_stats_full"
OUTPUT_FILE = f"{SAVE_DIR}/team_win_loss.csv"

STANDINGS_API = "https://statsapi.mlb.com/api/v1/standings"

# ---------- Transform ---------
TEAM_NAME_TO_ABBREV = {
    "D-backs": "AZ", "Braves": "ATL", "Orioles": "BAL",
    "Red Sox": "BOS", "White Sox": "CWS", "Cubs": "CHC",
    "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE",
    "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Yankees": "NYY", "Mets": "NYM", "Athletics": "OAK",
    "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD",
    "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL",
    "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR",
    "Nationals": "WSH"
}

def get_abbrev(name):
    return TEAM_NAME_TO_ABBREV.get(name, "UNK")

# ---------- MAIN ----------
all_rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="📅 Fetching win/loss records"):
    params = {
        "season": str(year),
        "leagueId": "103,104",  # AL + NL
        "standingsTypes": "regularSeason"
    }

    res = requests.get(STANDINGS_API, params=params)
    res.raise_for_status()
    data = res.json()

    for league in data["records"]:
        for team in league["teamRecords"]:
            team_name = team["team"]["name"]
            wins = team["wins"]
            losses = team["losses"]
            games = wins + losses
            win_rate = round(wins / games, 4) if games > 0 else None
            abbrev = get_abbrev(team_name)

            # ⏬ Append to main table
            all_rows.append({
                "year": year,
                "teamName": abbrev,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate
            })

            # 📁 Save win_rate to team folder
            team_dir = os.path.join(SAVE_DIR, str(year), abbrev)
            os.makedirs(team_dir, exist_ok=True)
            with open(os.path.join(team_dir, "win_rate.csv"), "w") as f:
                f.write(f"{win_rate:.4f}")

# ---------- SAVE MASTER CSV ----------
df = pd.DataFrame(all_rows)
os.makedirs(SAVE_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved team_win_loss master CSV to: {OUTPUT_FILE}")










