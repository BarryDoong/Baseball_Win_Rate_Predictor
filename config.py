import torch
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path("./mlb_stats_full")

HITTER_FEATS = ["atBats", "AVG", "OBP", "SLG", "OPS"]
PITCHER_FEATS = ["IP", "ERA", "WHIP", "SO", "W"]

N_BATTERS = 9
N_PITCHERS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE = 20
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-4

LOSS_GAMMA1 = 5.0
LOSS_GAMMA2 = 3.0