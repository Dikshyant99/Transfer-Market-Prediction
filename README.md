# Premier League Market Value Predictor

Linear regression model that predicts a player's market value from
goals, assists, matches played, age, position and club (big six).
Search a player by name and get a chart of predicted vs actual value.

## Setup
1. Get a free API token at football-data.org
2. Download `players.csv` and `player_valuations.csv` from
   github.com/dcaribou/transfermarkt-datasets and run `prepare_values.py`
3. `pip install requests pandas scikit-learn matplotlib seaborn`
4. Set `FOOTBALL_DATA_TOKEN` and run `python pl_value_predictor.py`

## Results
R² ≈ 0.43 on held-out players. The model underestimates the biggest stars.

Data: football-data.org (stats), Transfermarkt via dcaribou/transfermarkt-datasets (values).
