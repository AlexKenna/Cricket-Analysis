import pandas as pd
from constants import *

#------------------------- Data Reading and Writing  -------------------------#
# Write dataframe to file.
def _write_dataframe_to_file(dataframe:pd.DataFrame, filename:str):
  filename = DATA_PATH + filename
  dataframe.to_csv(filename, sep="\t", index=False)
