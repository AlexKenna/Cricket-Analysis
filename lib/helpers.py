import pandas as pd
from lib.constants import DATA_PATH

#------------------------- Data Reading and Writing  -------------------------#
# Write dataframe to file.
def _write_dataframe_to_file(dataframe:pd.DataFrame, filename:str):
  filename = DATA_PATH + filename
  dataframe.to_csv(filename, sep="\t", index=False)

# Read dataframe.
def _read_dataframe(filename:str, low_memory:bool=True):
  try: 
    df = pd.read_csv(DATA_PATH + filename, delimiter="\t", low_memory=low_memory)
  except FileNotFoundError:
    t = ("{} was not found in the directory {}. Please restore "
         "this file or update constants.py with the correct location.")
    raise FileNotFoundError(t.format(filename, DATA_PATH))

  return df
