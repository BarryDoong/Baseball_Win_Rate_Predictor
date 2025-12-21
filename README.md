# Baseball Win Rate Predictor

This project implements a neural network model to **predict MLB team win rate** using structured batting and pitching statistic.  
This project is created for my own amusement, hence some place may be lack of details

## Implementation Steps
This project consists of two steps:   
- Fetching data from MLB website
  - `webwarm_4_player.py`
  - `webwarm_4_team.py`
- Implement the model
  - `main.py`

## Implementation Details

### Fetching Data ###
- `webwarm_4_player.py`
  - This file fetchs the top 9 batters (ordered by AB) and top 10 pitchers (ordered by IP) for each (team, year) combaination

- `webwarm_4_team.py`
  - This file fetchs the win rate for each (team, year) combaination


### Model Architecture ###
- Batter Side
  - A **circular sliding windows of size 4** is applied since the batter can be batted to score by as far as the third batter behind him
  - Each windows is passed through a shared **MLP encoder**
  - Window embeddings are **mean-aggregated** into a pitcher representation
 
- Pitcher Side
  - I applied a sepcial windows since I assume that the top 5 pithers are starting pitchers and others are relievers for a team:
    - `{1, 6, 7, 8, 9, 10}`
    - `{2, 6, 7, 8, 9, 10}`
    - ...
    - `{5, 6, 7, 8, 9, 10}`
  - Each window is passed through a shared **MLP encoder**
  - Window embeddings are **mean-aggregated** into a pitcher representation
 
- Loss Function
  - A customized loss function is applied.
  - The loss is increased when the ground-truth win rate lies within `[0.4, 0.7]`.
  - The loss is further penalized when the prediction error exceeds `0.05`.

- Final Prediction
  - Concatenate batter + pitcher embeddings
  - Fully connected head
  - **Sigmoid output** -> predicted win rate 

### Sample Port ###
- Input
  - At fetching data stage, webwarm will automatically save all the input data in ./mlb_stats_full. One can adjust input by changing the content in the folder.

- Output
  - Learning Curve:  
    Automatically saved as `loss_curve.png`.
    ![Learning Curve](loss_curve.png)
  - prediction vs ground truth:  
    Automatically saved as `pred_vs_gt_annotated.png`.  
    The figure visualizes prediction accuracy by showing interval bands around the ideal *y = x* line, making it easier to assess model performance.
    ![Pred vs Ground Truth](pred_vs_gt_annotated.png)

### Potential Extension ###
- I want to use this model to evaluate player's value at the market. By changing the player in the input data, One can observe the potential win_rate difference.
- This project is a practice for implementing **Neural Network**. My ultimate goal is to construct a baseball manager model to help the manager with on-court decision. 

### Author ###
- Doong, Shao-Jyun
    
