#---------------------------------- Imports ----------------------------------#
from sklearn.ensemble import RandomForestRegressor
from lib.helpers import _read_dataframe, _create_test_train_split


#------------------------------ Model Training -------------------------------#
def train_model(model: RandomForestRegressor):
    # Read in the reduced batter summary.
    df = _read_dataframe("/Batter_Summary_Reduced.txt", False);

    # Remove Name and Batter_ID from summary.
    df = df.drop(columns=["Name", "Batter_ID"])

    # Split the dataset.
    X_train, _, y_train, _ = _create_test_train_split(df)

    # Train the model.
    model.fit(X_train, y_train)    

    # Return the Random Forest Regressor.
    return model
