from string import Template
import pandas as pd
from constants import DATA_PATH, MIN_INNINGS
from helpers import _write_dataframe_to_file

#------------------------------- Data Cleaning -------------------------------#
def clean_raw_data():
  # Read and basic clean match data.
  match_data = _read_match_data()
  match_data = _clean_match_data(match_data)

  # Read and basic clean delivery data.
  delivery_data = _read_delivery_data(match_data)
  delivery_data = _clean_delivery_data(delivery_data, match_data)

  # Perform a final clean of both datasets.
  #delivery_data, match_data = _clean_matches_and_deliveries(delivery_data, match_data)

  # Write cleaned data to file.
  _write_dataframe_to_file(match_data, "/Matches_Clean.txt")
  _write_dataframe_to_file(delivery_data, "/Deliveries_Clean.txt")


#---------------------------- Cleaning Functions -----------------------------#
# Clean match data.
def _clean_match_data(match_data:pd.DataFrame):
  match_data = _remove_unnecessary_match_columns(match_data)
  match_data = _remove_female_matches(match_data)
  match_data = _remove_disability_matches(match_data)
  match_data = _remove_non_australian_matches(match_data)
  match_data = _remove_uncommon_match_formats(match_data)
  match_data = _remove_non_OD_international_matches(match_data)

  return match_data

# Clean delivery data.
def _clean_delivery_data(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  delivery_data = _remove_foreign_deliveries(delivery_data, match_data)
  delivery_data = _remove_domestic_deliveries(delivery_data, match_data)
  delivery_data = _remove_too_few_australian_deliveries(delivery_data, match_data)
  delivery_data = _remove_too_few_domestic_deliveries(delivery_data, match_data)

  return delivery_data

# Clean both the match and delivery data.
def _clean_matches_and_deliveries(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  match_data = _remove_empty_matches(delivery_data, match_data)

  return delivery_data, match_data


#----------------------------- Filter Functions ------------------------------#
# Remove unnecessary match data columns.
def _remove_unnecessary_match_columns(df:pd.DataFrame):
  # Official's columns are unnecessary
  return df[df.columns.drop(list(df.filter(regex='Official+')))]

# Remove female formats from match data.
def _remove_female_matches(df:pd.DataFrame):
  return df.loc[df["Series Gender Id"] == 1]
  
# Remove match data for disability teams.
def _remove_disability_matches(df:pd.DataFrame):
  return df[~df.TeamA.str.contains("Disability") | 
            ~df.TeamB.str.contains("Disability")]

# Remove international games where Australia is not playing.
def _remove_non_australian_matches(df:pd.DataFrame):
  return df[~(df.Series.str.contains("International") & 
            ~df.TeamA.str.contains("Australia") & 
            ~df.TeamB.str.contains("Australia"))]

# Filter matches to only include 1-day, 4-day, 5-day, and T20.
def _remove_uncommon_match_formats(df:pd.DataFrame):
  return df[df["Match Type Id"].isin([1,4,5,7])]

# Remove international matches that are not 1 Day format.
def _remove_non_OD_international_matches(df:pd.DataFrame):
  return df[(df["Match Type Id"] == 1 & 
             df.Series.str.contains("International")) | 
             df.Series.str.contains("Domestic")]

# Remove deliveries to foreign batters in international one-days.
def _remove_foreign_deliveries(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  odis = match_data[match_data["Series"].str.contains("International")]["Match Id"].tolist()
  return delivery_data[~(delivery_data["Match Id"].isin(odis) & 
                       ~delivery_data["Team Batting"].str.contains("Australia"))]

# Remove deliveries to domestic batters that have not played international.
def _remove_domestic_deliveries(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  odis = match_data[match_data["Series"].str.contains("International")]["Match Id"].tolist()
  aus_odi_deliveries = delivery_data[delivery_data["Match Id"].isin(odis)]
  aus_odi_batters = aus_odi_deliveries["Striker Id"].unique()

  return delivery_data[delivery_data["Striker Id"].isin(aus_odi_batters)]

# Remove deliveries to batters that have batted in less than 10 ODI matches.
def _remove_too_few_australian_deliveries(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  # Extract international deliveries.
  int_matches = match_data[match_data["Series"].str.contains("International")]["Match Id"].tolist()
  int_deliveries = delivery_data[delivery_data["Match Id"].isin(int_matches)]

  # Count number of innings per batter.
  by_columns = ["Striker Id"]
  aggregates = {"Match Id": pd.Series.nunique}
  int_groupby_data = int_deliveries.groupby(by=by_columns).agg(aggregates)

  # Remove batters that have batted in less than 10 international One Day innings.
  valid_batters = int_groupby_data[int_groupby_data["Match Id"] >= 10].index
  return delivery_data[delivery_data["Striker Id"].isin(valid_batters)]

# Remove deliveries to batters that have batted in less than 10 domestic matches.
def _remove_too_few_domestic_deliveries(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  # Extract international deliveries.
  dom_matches = match_data[match_data["Series"].str.contains("Domestic")]["Match Id"].tolist()
  dom_deliveries = delivery_data[delivery_data["Match Id"].isin(dom_matches)]

  # Count number of innings per batter.
  by_columns = ["Striker Id"]
  aggregates = {"Match Id": pd.Series.nunique}
  dom_groupby_data = dom_deliveries.groupby(by=by_columns).agg(aggregates)

  # Remove batters that have batted in less than 10 international One Day innings.
  valid_batters = dom_groupby_data[dom_groupby_data["Match Id"] >= 10].index
  return delivery_data[delivery_data["Striker Id"].isin(valid_batters)]

# Remove matches that contain no delivery data.
def _remove_empty_matches(delivery_data:pd.DataFrame, match_data:pd.DataFrame):
  match_ids = match_data["Match Id"].unique().tolist()
  delivery_ids = delivery_data["Match Id"].unique().tolist()

  empty_matches = set(match_ids) - set(delivery_ids)

  return match_data[~(match_data["Match Id"].isin(empty_matches))]

#-------------------------- Data Reading Functions ---------------------------#
# Read match data.
def _read_match_data():
  try: 
    match_data = pd.read_csv(DATA_PATH + "/Matches.txt", delimiter="\t")
  except FileNotFoundError:
    t = Template("Match data file not found in the directory $path. Please "
        "restore this file or update constants.py with the correct location.")
    raise FileNotFoundError(t.substitute(DATA_PATH))

  return match_data
  

# Read delivery data.
def _read_delivery_data(match_data):
  # Initialise delivery data.
  delivery_data = pd.DataFrame()

  # Extract important match data.
  match_ids = match_data["Match Id"]
  match_columns = set(match_data.columns)
  match_columns.remove("Match Id")

  try:
    for chunk in pd.read_csv(DATA_PATH + "/Deliveries.txt", delimiter="\t", chunksize=10**6):
      chunk = chunk[chunk["Match Id"].isin(match_ids)]
      chunk.drop(
        [col for col in chunk.columns if col in match_columns], axis=1, inplace=True
      )

      # Combine filtered deliveries into single dataframe.
      delivery_data = pd.concat([delivery_data, chunk])

  except FileNotFoundError:
    t = Template("Delivery data file not found in the directory $path. Please "
        "restore this file or update constants.py with the correct location.")
    raise FileNotFoundError(t.substitute(DATA_PATH))

  return delivery_data
