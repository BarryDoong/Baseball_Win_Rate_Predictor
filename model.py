import torch.nn as nn
import torch
from config import N_BATTERS, N_PITCHERS

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
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.25):
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
        self.batter_win = MLP(in_dim=4 * hitter_feat_dim, hidden_dims=[128, 64], out_dim=emb_dim, dropout=0.15)
        # Pitcher window MLP: (6 * Pf) -> emb_dim
        self.pitcher_win = MLP(in_dim=6 * pitcher_feat_dim, hidden_dims=[128, 64], out_dim=emb_dim, dropout=0.15)

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