#---------------------------------- Imports ----------------------------------#
import pandas as pd
from lib.constants import DATA_PATH


#---------------------------- Test and Training  -----------------------------#
# Create test and training data split.
def _create_test_train_split(df: pd.DataFrame):
    # Define the size of the test dataset.
    test_size = 12

    # Split the features and target.
    X = df.drop(columns=["International_One_Day_Batting_Average"])
    y = df["International_One_Day_Batting_Average"]

    # Split the data.
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    # Separately fill missing data in training and test sets.
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    # Return the split data.
    return (X_train, X_test, y_train, y_test)


#------------------------- Data Reading and Writing  -------------------------#
# Write dataframe to file.
def _write_dataframe_to_file(dataframe: pd.DataFrame, filename: str):
    filename = DATA_PATH + filename
    dataframe.to_csv(filename, sep="\t", index=False)

# Read dataframe.
def _read_dataframe(filename: str, low_memory: bool = True):
    try:
        df = pd.read_csv(DATA_PATH + filename, delimiter="\t",
                      low_memory=low_memory)
    except FileNotFoundError:
        t = ("{} was not found in the directory {}. Please restore "
          "this file or update constants.py with the correct location.")
        raise FileNotFoundError(t.format(filename, DATA_PATH))

    return df
