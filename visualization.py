from matplotlib import pyplot as plt
import torch
import numpy as np
from config import DEVICE

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
    coverage=0.95,              # 95% accuracy interval
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