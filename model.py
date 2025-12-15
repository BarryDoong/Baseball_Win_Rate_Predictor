from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

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

LOSS_GAMMA = 5.0


# ----------------------------
# Window builders
# ----------------------------
def circular_windows(x: torch.Tensor, window: int = 4) -> torch.Tensor:
    """
    x: (N, F) -> (N, window, F) circular wrap
    """
    N, _ = x.shape
    out = []
    for i in range(N):
        idx = [(i + k) % N for k in range(window)]
        out.append(x[idx])
    return torch.stack(out, dim=0)  # (N, window, F)


def pitcher_windows_top10(p: torch.Tensor) -> torch.Tensor:
    """
    p: (10, F) sorted by IP desc already (rank 1..10)
    windows:
      [0,5,6,7,8,9],
      [1,5,6,7,8,9],
      ...
      [4,5,6,7,8,9]
    returns: (5, 6, F)
    """
    fixed = [5, 6, 7, 8, 9]
    out = []
    for i in range(5):
        idx = [i] + fixed
        out.append(p[idx])
    return torch.stack(out, dim=0)  # (5, 6, F)


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


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WinRateNet(nn.Module):
    def __init__(self, hitter_feat_dim, pitcher_feat_dim, emb_dim=32):
        super().__init__()
        # Batter window MLP: (4 * Hf) -> emb_dim
        self.batter_win = MLP(in_dim=4 * hitter_feat_dim, hidden_dims=[128, 64], out_dim=emb_dim, dropout=0.1)
        # Pitcher window MLP: (6 * Pf) -> emb_dim
        self.pitcher_win = MLP(in_dim=6 * pitcher_feat_dim, hidden_dims=[128, 64], out_dim=emb_dim, dropout=0.1)

        # Final head: concat -> ... -> ReLU -> pred_win_rate
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # enforce non-negative prediction as you requested
        )

    def forward(self, hitters, pitchers):
        """
        hitters:  (B, 9, Hf)
        pitchers: (B, 10, Pf)
        """
        B, _, Hf = hitters.shape
        _, _, Pf = pitchers.shape

        # Batter: (B, 9, 4, Hf) -> (B, 9, 4*Hf) -> (B, 9, emb) -> mean -> (B, emb)
        b_windows = torch.stack([circular_windows(hitters[b], window=4) for b in range(B)], dim=0)
        b_flat = b_windows.reshape(B, N_BATTERS, 4 * Hf)
        b_emb = self.batter_win(b_flat).mean(dim=1)

        # Pitcher: (B, 5, 6, Pf) -> (B, 5, 6*Pf) -> (B, 5, emb) -> mean -> (B, emb)
        p_windows = torch.stack([pitcher_windows_top10(pitchers[b]) for b in range(B)], dim=0)
        p_flat = p_windows.reshape(B, 5, 6 * Pf)
        p_emb = self.pitcher_win(p_flat).mean(dim=1)

        x = torch.cat([b_emb, p_emb], dim=-1)
        yhat = self.head(x).squeeze(-1)
        return yhat


# ----------------------------
# Train / Eval
# ----------------------------
def split_train_val(n, val_ratio=0.3, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    v = int(n * val_ratio)
    return idx[v:].tolist(), idx[:v].tolist()


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pred_vs_gt_annotated(
    model,
    dataloader,
    save_path="pred_vs_gt_annotated.png",
    device=DEVICE,
    coverage=0.8,              # 80% accuracy interval
    extra_thresholds=(0.01, 0.02, 0.05, 0.10),
):
    model.eval()

    all_pred, all_gt = [], []

    with torch.no_grad():
        for hitters, pitchers, y, years, teams in dataloader:
            hitters = hitters.to(device)
            pitchers = pitchers.to(device)
            y = y.to(device)

            pred = model(hitters, pitchers)

            all_pred.append(pred.detach().cpu().numpy())
            all_gt.append(y.detach().cpu().numpy())

    # Flatten
    x = np.concatenate(all_pred, axis=0).reshape(-1)  # pred
    y = np.concatenate(all_gt, axis=0).reshape(-1)    # gt

    # Absolute error
    abs_err = np.abs(x - y)

    # Coverage-based error radius
    X = float(np.quantile(abs_err, coverage))

    # Metrics
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((x - y) ** 2))

    fixed = {t: float(np.mean(abs_err <= t)) for t in extra_thresholds}

    # Plot range
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    pad = 0.02 * (mx - mn + 1e-9)
    lo, hi = mn - pad, mx + pad

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=30, alpha=0.8)

    # y = x
    plt.plot([lo, hi], [lo, hi], linewidth=1)

    # ±X band around y=x
    plt.plot([lo, hi], [lo + X, hi + X], linestyle="--", linewidth=1)
    plt.plot([lo, hi], [lo - X, hi - X], linestyle="--", linewidth=1)

    # Explicit ±X reference lines
    plt.axhline(+X, linestyle=":", linewidth=1)
    plt.axhline(-X, linestyle=":", linewidth=1)
    plt.axvline(+X, linestyle=":", linewidth=1)
    plt.axvline(-X, linestyle=":", linewidth=1)

    # Text box
    lines = [
        f"{int(coverage * 100)}% within ±{X:.4f}",
        f"MAE: {mae:.4f}",
        f"RMSE: {rmse:.4f}",
    ]
    for t, v in fixed.items():
        lines.append(f"within ±{t:.2f}: {v*100:.1f}%")

    plt.gca().text(
        0.02, 0.98,
        "\n".join(lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15, pad=0.4)
    )

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("pred_win_rate (x)")
    plt.ylabel("ground_truth (y)")
    plt.title("Pred vs Ground Truth with Accuracy Interval")

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()

    print(f"Saved: {save_path}")
    print(f"{int(coverage * 100)}% of predictions are within ±{X:.6f} win_rate")



def main(epochs=10, batch_size=5, lr=1e-3, patience=20):
    ds = MLBWinRateDataset(BASE_DIR)
    train_idx, val_idx = split_train_val(len(ds), val_ratio=0.2)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    patience_counter = 1

    # Fit standardizer on train set only
    h_list, p_list = [], []
    for i in range(len(train_ds)):
        h, p, y, *_ = train_ds[i]
        h_list.append(h)
        p_list.append(p)
    scaler = Standardizer()
    scaler.fit(h_list, p_list)

    def loss_func(pred, y):
        
        error = (pred -y).pow(2)
        mask = ((y>=0.4) & (y<= 0.7)).float()
        
        weight = 1.0 + LOSS_GAMMA * mask

        loss = (weight * error).mean()
        
        return loss

    def collate(batch):
        hitters, pitchers, ys, years, teams = [], [], [], [], []
        for h, p, y, year, team in batch:
            hitters.append(scaler.transform_hitters(h))
            pitchers.append(scaler.transform_pitchers(p))
            ys.append(y)
            years.append(year)
            teams.append(team)
        return (
            torch.stack(hitters, 0),
            torch.stack(pitchers, 0),
            torch.stack(ys, 0),
            years,
            teams,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = WinRateNet(len(HITTER_FEATS), len(PITCHER_FEATS), emb_dim=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # ---- train
        model.train()
        tr_sum = 0.0
        for hitters, pitchers, y, *_ in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]"):
            hitters, pitchers, y = hitters.to(DEVICE), pitchers.to(DEVICE), y.to(DEVICE)
            pred = model(hitters, pitchers)
            loss = loss_func(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_sum += loss.item() * y.size(0)
        tr_loss = tr_sum / len(train_ds)
        train_losses.append(tr_loss)

        # ---- val
        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for hitters, pitchers, y, *_ in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]  "):
                hitters, pitchers, y = hitters.to(DEVICE), pitchers.to(DEVICE), y.to(DEVICE)
                pred = model(hitters, pitchers)
                loss = loss_func(pred, y)
                va_sum += loss.item() * y.size(0)

        va_loss = va_sum / max(1, len(val_ds))
        val_losses.append(va_loss)

        if va_loss < best_val:
            patience_counter = 1
            best_val = va_loss
            torch.save(model.state_dict(), "best_winrate_model.pt")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {ep} (patience {patience} exceeded)")
                break

        print(f"Epoch {ep}: train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}  best_val={best_val:.6f}")

    # plot + save
    plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png")
    print("Saved: best_winrate_model.pt")
    print("Saved: loss_curve.png")

    # ---- load best model
    model.load_state_dict(torch.load("best_winrate_model.pt", map_location=DEVICE))

    # ---- print predictions on validation set
    plot_pred_vs_gt_annotated(model, val_loader, save_path="pred_vs_gt_annotated.png")



if __name__ == "__main__":
    main(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)




