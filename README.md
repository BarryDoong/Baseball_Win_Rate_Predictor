# Baseball Win Rate Predictor ⚾📊

This project implements a neural network model to **predict MLB team win rate** using structured batting and pitching statistics.  
The model architecture is explicitly designed to reflect **baseball domain structure**, rather than treating players as an unordered set.

---

## 📌 Project Overview

- **Input**:  
  Team-level statistics for **9 batters** and **10 pitchers** per team-season
- **Output**:  
  Predicted **win rate** (continuous value in \[0, 1\])
- **Dataset size**: ~300 team-year samples

The model uses **window-based encoders** inspired by CNNs to capture interactions among batters and pitchers.

---

## 🧠 Model Architecture

### Batter Side (Offense)
- Exactly **9 batters**, ordered by **at-bats (AB)**
- A **circular sliding window of size 4** is applied:
  - Windows: (1–4), (2–5), …, (9–3)
- Each window is processed by a **shared MLP encoder**
- Window embeddings are **mean-aggregated** to form a batter representation

### Pitcher Side (Defense)
- Exactly **10 pitchers**, ordered by **innings pitched (IP)**
- Special windows are constructed:
  - `{1,6,7,8,9,10}`
  - `{2,6,7,8,9,10}`
  - …
  - `{5,6,7,8,9,10}`
- Each window is processed by a **shared MLP encoder**
- Window embeddings are **mean-aggregated** to form a pitcher representation

### Final Prediction
- Batter and pitcher embeddings are concatenated
- Fully connected prediction head
- **Sigmoid output** ensures predictions lie in \[0, 1\]

---

## 📁 Dataset Structure

⚠️ **Raw data is not included in this repository.**

Expected directory layout:
mlb_stats/
├── 2019/
│ ├── NYY/
│ │ ├── hitters.csv
│ │ ├── pitchers.csv
│ │ └── win_rate.csv
│ └── BOS/
│ ├── hitters.csv
│ ├── pitchers.csv
│ └── win_rate.csv
├── 2020/
│ └── ...


### Dataset Assumptions
- Each team-year has:
  - exactly **9 hitters**
  - exactly **10 pitchers**
- CSV files are **already sorted**:
  - hitters → by `atBats`
  - pitchers → by `IP`
- `win_rate.csv` contains a **single numeric value**

---

## 📊 Features Used

### Hitters
- `atBats`
- `AVG`
- `OBP`
- `SLG`
- `OPS`

### Pitchers
- `IP`
- `ERA`
- `WHIP`
- `SO`
- `W`

All features are standardized using statistics computed from the training set.

---

## 🚀 Training Details

- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Weight Initialization**: He (Kaiming) initialization
- **Stability Techniques**:
  - Gradient clipping
  - Sigmoid output layer
- **Typical Hyperparameters**:
batch_size = 16
epochs = 100
learning rate = 5e-4


---

## 📈 Evaluation & Visualization

The training script automatically generates:

- 📉 **Training / Validation loss curve**
- `loss_curve.png`
- 📍 **Prediction vs Ground Truth plot**
- x-axis: predicted win rate
- y-axis: ground truth
- includes reference line `y = x`
- each data point annotated with `year-team`

Example output:
pred_vs_gt_val_annotated.png


---

## ▶️ How to Run

1. Place your dataset under:
./mlb_stats_full/

2. Run training:
```bash
python model.py
```

🔒 Reproducibility Notes
- Training uses randomized initialization; results may vary slightly across runs
- Fixed seeds can be added for deterministic experiments if needed

🧩 Possible Extensions
- Attention-based window aggregation
- Cross-validation by season
- Feature importance analysis by batter/pitcher window
- Extension to game-level or lineup-level prediction

👤 Author
- Doong, Shao-Jyun

📜 License
- This project is intended for research and educational use.



