import pandas as pd
from constants import *

#------------------------------- Strip Matches ------------------------------#
# Read in matches data using tab delimiter (\t)
try: 
  match_data = pd.read_csv(DATA_PATH + "/Matches.txt", delimiter="\t")
except FileNotFoundError:
  raise FileNotFoundError("Match data file not found in the directory " 
                          + DATA_PATH + ". Please restore this file or " + 
                          "update constants.py with the correct location.")

# Filter only male formats
match_data = match_data.loc[match_data["Series Gender Id"] == 1]

# Filter only years from 2000-2021
years = [str(x) for x in range(CUT_OFF_YEAR_START, CUT_OFF_YEAR_END)]
match_data = match_data[match_data["Match YYMMDD"].astype(str).str[:4].isin(years)]

# Filter disability matches
match_data = match_data[~match_data.TeamA.str.contains("Disability") & 
                        ~match_data.TeamB.str.contains("Disability")]

# Only keep 1-day, 4-day, 5-day and T20 matches
match_data = match_data[match_data["Match Type Id"].isin([1,4,5,7])]


#----------------------------- Strip Deliveries ------------------------------#
# Initialise deliveries dataframe
deliveries_data = pd.DataFrame()

# Determine which matches are important to read from deliveries
match_ids = match_data["Match Id"]

# Read in deliveries data in chunks
chunksize = 10 ** 6

try:
  for chunk in pd.read_csv(DATA_PATH + "/Deliveries.txt", delimiter="\t", chunksize=chunksize):
    chunk = chunk[chunk["Match Id"].isin(match_ids)]
    deliveries_data = pd.concat([deliveries_data, chunk])
except FileNotFoundError:
  raise FileNotFoundError("Deliveries data file not found in the directory " 
                          + DATA_PATH + ". Please restore this file or " + 
                          "update constants.py with the correct location.")


#-------------------------------- Clean Data ---------------------------------#
# Find unique match IDs in deliveries and matches data 
deliveries_matches = deliveries_data["Match Id"].unique()
match_matches = match_data["Match Id"].unique()

# Determine which match IDs are not recorded in deliveries
anomalous_match_IDs = list(set(match_matches).difference(deliveries_matches))

# Remove the matches with no deliveries recorded
match_data = match_data[~match_data["Match Id"].isin(anomalous_match_IDs)]

# Remove columns from matches and deliveries that contain no data
match_data = match_data.dropna(axis=1, how="all")
deliveries_data = deliveries_data.dropna(axis=1, how="all")

# Remove duplicate data from deliveries file
match_columns = set(match_data.columns)
deliveries_columns = set(deliveries_data.columns)
deliveries_columns.remove("Match Id")
duplicate_columns = list(match_columns.intersection(deliveries_columns))

deliveries_data.drop(duplicate_columns, axis=1, inplace=True)


#---------------------------- Write Data to File -----------------------------#
# Write matches data
match_file_name = DATA_PATH + "/Matches_Clean.txt"
match_data.to_csv(match_file_name, sep="\t", index=False)

# Write deliveries data
deliveries_file_name = DATA_PATH + "/Deliveries_Clean.txt"
deliveries_data.to_csv(deliveries_file_name, sep="\t", index=False)

