#---------------------------------- Imports ----------------------------------#
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from lib.helpers import _write_dataframe_to_file, _read_dataframe, _create_test_train_split


#----------------------------- Summary Reduction -----------------------------#
def reduce_summary():
    # Read summary data and remove  unnecessary columns.
    summary = _read_summary_data()
    summary["Hand"] = np.where(summary["Hand"] == "Right", 0, 1)
    df = summary.drop(columns=["Name", "Batter_ID"])

    # Split the dataset.
    X_train, _, y_train, _ = _create_test_train_split(df)

    # Remove redundant features.
    X_train = remove_redundant_features(X_train)

    # Remove highly correlated features.
    X_train = remove_correlated_features(X_train)

    # Recursively reduce feature set.
    fields = recursive_feature_elimination(X_train, y_train)

    # Add the Name and Batter ID fields back into the set.
    fields = fields + ["Name", "Batter_ID",
                       "International_One_Day_Batting_Average"]

    # Reduce the summary features.
    summary_reduced = summary[fields]

    # Write cleaned data to file.
    _write_dataframe_to_file(summary_reduced, "/Batter_Summary_Reduced.txt")


#---------------------------- Test and Training  -----------------------------#
# Remove features of no importance.
def remove_redundant_features(X_train):
    # Remove columns that have no variance (all 0 in this case).
    X_train_reduced = X_train.loc[:, (X_train != 0).any(axis=0)]

    return X_train_reduced


# Remove features that are highly correlated.
def remove_correlated_features(X_train):
    # Remove predetermined set of highly correlated features.
    X_train_reduced = X_train.drop(columns=[
        "Domestic_One_Day_Innings_Count",
        "Domestic_Test_Innings_Count",
        "Domestic_T20_Innings_Count",
        "Domestic_One_Day_Innings_Percent",
        "Domestic_Test_Innings_Percent",
        "Domestic_T20_Innings_Percent",
        "Domestic_One_Day_Average_Entering_Ball",
        "Domestic_One_Day_Average_Entering_Score",
        "Domestic_Test_Average_Entering_Ball",
        "Domestic_Test_Average_Entering_Score",
        "Domestic_T20_Average_Entering_Ball",
        "Domestic_T20_Average_Entering_Score",
        "Domestic_One_Day_Runs_Per_Innings_Average",
        "Domestic_Test_Runs_Per_Innings_Average",
        "Domestic_T20_Runs_Per_Innings_Average",
    ])

    return X_train_reduced


# Recursively reduce features.
def recursive_feature_elimination(X_train, y_train):
    # Create a blank Random Forest Regressor.
    rfr = RandomForestRegressor(n_estimators=150, min_samples_split=5,
                                min_samples_leaf=1, max_features=2, max_depth=20, bootstrap=True)

    # Use recursive feature selection to choose the best remaining features.
    selector_RFE = RFE(rfr, n_features_to_select=20)
    selector_RFE = selector_RFE.fit(X_train, y_train)
    reduced_features = selector_RFE.get_feature_names_out(
        X_train.columns.tolist()).tolist()

    return reduced_features


#-------------------------- Data Reading Functions ---------------------------#
# Read summary data.
def _read_summary_data():
    return _read_dataframe("/Batter_Summary.txt")
