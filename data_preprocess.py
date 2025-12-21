from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import N_BATTERS, N_PITCHERS, HITTER_FEATS, PITCHER_FEATS
import numpy as np
import torch
from pathlib import Path

# ----------------------------
# Standardizer
# ----------------------------
class Standardizer:
    def __init__(self):
        self.h_mean = None
        self.h_std = None
        self.p_mean = None
        self.p_std = None

    def fit(self, hitter_list, pitcher_list):
        H = torch.cat([h.reshape(-1, h.shape[-1]) for h in hitter_list], dim=0)  # (num*9, Hf)
        P = torch.cat([p.reshape(-1, p.shape[-1]) for p in pitcher_list], dim=0)  # (num*10, Pf)

        self.h_mean = H.mean(dim=0, keepdim=True)
        self.h_std = H.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.p_mean = P.mean(dim=0, keepdim=True)
        self.p_std = P.std(dim=0, keepdim=True).clamp_min(1e-6)

    def transform_hitters(self, h):
        return (h - self.h_mean) / self.h_std

    def transform_pitchers(self, p):
        return (p - self.p_mean) / self.p_std

# ----------------------------
# Dataset
# ----------------------------
class MLBWinRateDataset(Dataset):
    """
    Assumptions (as you stated):
    - exactly 9 hitters, already sorted by atBats in CSV
    - exactly 10 pitchers, already sorted by IP ranking in CSV
    - win_rate is a single scalar
    """
    def __init__(self, base_dir: Path):
        self.samples = []
        base_dir = Path(base_dir)

        for year_dir in sorted(base_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for team_dir in sorted(year_dir.iterdir()):
                if not team_dir.is_dir():
                    continue

                h_path = team_dir / "hitters.csv"
                p_path = team_dir / "pitchers.csv"
                y_path = team_dir / "win_rate.csv"

                if h_path.exists() and p_path.exists() and y_path.exists():
                    self.samples.append((int(year_dir.name), team_dir.name, h_path, p_path, y_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No samples found under {base_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        year, team, h_path, p_path, y_path = self.samples[idx]

        hitters = pd.read_csv(h_path)
        pitchers = pd.read_csv(p_path)

        # Take features in given order (already sorted in CSV)
        h_mat = hitters[HITTER_FEATS].to_numpy(dtype=np.float32)
        p_mat = pitchers[PITCHER_FEATS].to_numpy(dtype=np.float32)

        # sanity check
        if h_mat.shape[0] != N_BATTERS:
            raise ValueError(f"{year}/{team}: hitters rows={h_mat.shape[0]} != {N_BATTERS}")
        if p_mat.shape[0] != N_PITCHERS:
            raise ValueError(f"{year}/{team}: pitchers rows={p_mat.shape[0]} != {N_PITCHERS}")

        # ---- robust win_rate read
        try:
            y_df = pd.read_csv(y_path)
        except Exception:
            y_df = pd.DataFrame()

        y = None

        # 1) If dataframe has data, try to find a numeric value
        if y_df is not None and not y_df.empty:
            # try column named win_rate first
            if "win_rate" in y_df.columns:
                s = pd.to_numeric(y_df["win_rate"], errors="coerce").dropna()
                if len(s) > 0:
                    y = float(s.iloc[0])
            # else try first numeric anywhere
            if y is None:
                vals = pd.to_numeric(y_df.to_numpy().reshape(-1), errors="coerce")
                vals = vals[~pd.isna(vals)]
                if len(vals) > 0:
                    y = float(vals[0])

        # 2) Fallback: parse raw text and grab the first float-like number
        if y is None:
            text = Path(y_path).read_text(encoding="utf-8", errors="ignore")
            import re
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if m:
                y = float(m.group(0))

        if y is None:
            raise ValueError(f"win_rate.csv has no numeric value: {y_path}")


        return (
            torch.tensor(h_mat, dtype=torch.float32),
            torch.tensor(p_mat, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            year,
            team
        )
    
def split_train_val(n, val_ratio=0.3, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    v = int(n * val_ratio)
    return idx[v:].tolist(), idx[:v].tolist()