import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from visualization import plot_loss_curve, plot_pred_vs_gt_annotated
from pathlib import Path
from data_preprocess import MLBWinRateDataset, Standardizer, split_train_val
from model import WinRateNet, WinRateNet_without_Window
from config import BASE_DIR, HITTER_FEATS, PITCHER_FEATS, DEVICE, EPOCHS, BATCH_SIZE, LR, PATIENCE, LOSS_GAMMA1, LOSS_GAMMA2


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
        mask2 = (error > 0.025).float()

        weight = 1.0 + LOSS_GAMMA1 * mask + LOSS_GAMMA2 * mask2

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

    model = WinRateNet_without_Window(len(HITTER_FEATS), len(PITCHER_FEATS), emb_dim=64).to(DEVICE)
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
            torch.save(model.state_dict(), "./output/best_winrate_model.pt")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {ep} (patience {patience} exceeded)")
                break

        print(f"Epoch {ep}: train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}  best_val={best_val:.6f}")

    # plot + save
    plot_loss_curve(train_losses, val_losses, save_path="./output/loss_curve.png")
    print("Saved: best_winrate_model.pt")
    print("Saved: loss_curve.png")

    # ---- load best model
    model.load_state_dict(torch.load("./output/best_winrate_model.pt", map_location=DEVICE))

    # ---- print predictions on validation set
    plot_pred_vs_gt_annotated(model, val_loader, save_path="./output/pred_vs_gt_annotated.png")



if __name__ == "__main__":
    main(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)