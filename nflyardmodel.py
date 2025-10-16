import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Importing Statistics from 1966-2021 seasons' Super Bowls
sb_stats_path_name = "/Users/atharvparashar/Documents/PycharmProjects/MLlearning/superbowloffensivestats.csv"

sb_stats = pd.read_csv(sb_stats_path_name)
sb_stats = sb_stats [sb_stats['pass_attempts'] > 1]
players = sb_stats["player"]

# Statistics that could make our prediction more accurate
sb_stats["completion_rate"] = sb_stats["completions"]/sb_stats["pass_attempts"]


features = ["times_sacked", "sack_yards", "completions", "pass_attempts", "longest_pass", "completion_rate", "passing_td"]
y = sb_stats['passing_yards'].copy()
X = sb_stats[features]

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state= 1)

sb_model = RandomForestRegressor(n_estimators = 300, random_state = 1)
sb_model.fit(train_x, train_y)
sb_model_values = sb_model.predict(test_x)

for i in range(0, len(sb_model_values)):
    sb_model_values[i] = int(sb_model_values[i])

# Display/Output

test_rows = sb_stats.loc[test_x.index, ["player", "team", "year", "passing_yards"]].copy() \
            if set(["team","year"]).issubset(sb_stats.columns) \
            else sb_stats.loc[test_x.index, ["player", "passing_yards"]].copy()
test_rows = test_rows.rename(columns={"passing_yards": "actual_yards"})
test_rows["predicted_yards"] = sb_model_values

print(test_rows)

print("\nTESTING ON DATA FROM SUPER BOWLS 57-59\n")
# Testing on data in 2022-2024 Super Bowls (PHI vs. KC, SF vs KC, PHI vs. KC)
new_sb_data_path = "/Users/atharvparashar/Documents/Computing/PycharmProjects/MLDSlearning/2022-2024 SB Quarterbacks.csv"
new_sb_data = pd.read_csv(new_sb_data_path)
print(new_sb_data['player'], new_sb_data['passing_yards'])
new_sb_data["completion_rate"] = new_sb_data["completions"]/new_sb_data["pass_attempts"]
new_sb_data = new_sb_data[features]
predictions = sb_model.predict(new_sb_data)

print(predictions)
