#---------------------------------- Imports ----------------------------------#
from sklearn.ensemble import RandomForestRegressor


#------------------------------ Model Creation -------------------------------#
def create_model():
    # Define the optimal hyperparameters.
    params = {"bootstrap": True,
              "max_depth": 16,
              "max_features": 2,
              "min_samples_leaf": 1,
              "min_samples_split": 3,
              "n_estimators": 142}

    # Create the Random Forest Regressor.
    rfr = RandomForestRegressor(bootstrap=params["bootstrap"],
                                max_depth=params["max_depth"],
                                max_features=params["max_features"],
                                min_samples_leaf=params["min_samples_leaf"],
                                min_samples_split=params["min_samples_split"],
                                n_estimators=params["n_estimators"])

    # Return the Random Forest Regressor.
    return rfr
