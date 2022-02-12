#---------------------------------- Imports ----------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lib.helpers import _read_dataframe, _create_test_train_split


#------------------------------- Model Testing -------------------------------#
def test_model(model: RandomForestRegressor):
  # Read in the reduced batter summary.
  summary = _read_dataframe("/Batter_Summary_Reduced.txt", False)
  df = summary.drop(columns=["Name", "Batter_ID"])

  # Split the dataset.
  _, X_test, _, y_test = _create_test_train_split(df)
  player_names = summary[-12:]["Name"]
  features = X_test.columns

  # Show the features selected as the most important.
  table_features(model, features)

  # Show table of error metrics.
  table_error_metrics(model, X_test, y_test)

  # Display scatter plot of true and predicted averages.
  plot_averages(y_test, model.predict(X_test))

  # Display table of true and predicted averages.
  table_averages(player_names, y_test, model.predict(X_test))


#----------------------------- Testing Functions -----------------------------#
# Create a table of the most important features.
def table_features(model, features):
  # Determine the importance of each feature.
  feats = {}
  for feature, importance in zip(features, model.feature_importances_):
      feats[feature] = importance

  # Create a table for the importances and sort the values.
  importances = pd.DataFrame.from_dict(
      feats, orient="index").rename(columns={0: "Gini-importance"})
  importances = importances.sort_values(by="Gini-importance", ascending=False)

  display(importances)


# Create a table of error metrics.
def table_error_metrics(model, X_test, y_test):
  # Coefficient of Determination.
  r2 = model.score(X_test, y_test)

  # Mean Average Error.
  mae = mean_absolute_error(y_test, model.predict(X_test))

  # Root Mean Squared Error.
  rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

  # Create a new table.
  d = {"Metric": ["Coefficient of Determination", "Mean Average Error",
                  "Root Mean Squared Error"], "Score": [r2, mae, rmse]}
  df = pd.DataFrame(d).set_index("Metric")

  display(df)


# Create a scatter plot of the true and predicted batting averages.
def plot_averages(true_averages, predicted_averages):
  # Plot each batter's predicted and true average.
  plt.scatter(true_averages, predicted_averages, color="#003b63", linewidth=5)

  # Plot the main diagonal.
  plt.axline([10, 10], [47.5, 47.5], color="#e56d54", linewidth=5)

  # Figure configurations.
  plt.xlim([5, 50])
  plt.ylim([5, 50])
  plt.rcParams.update({'font.size': 15})
  plt.xlabel("True Average")
  plt.ylabel("Predicted Average")
  plt.title("Comparison of True and Predicted Averages", pad=20)


# Create a table to compare true and predicted batting averages.
def table_averages(names, true_averages, predicted_averages):
  # Convert data to list.
  names = names.tolist()
  true_averages = true_averages.tolist()
  predicted_averages = predicted_averages.tolist()

  # Create a blank dataframe with the batter names.
  d = {"Averages": ["True Average", "Predicted Average"]}
  for num in range(0, len(names)):
      d[names[num]] = [true_averages[num], predicted_averages[num]]
  df = pd.DataFrame(d).set_index("Averages")

  display(df)
