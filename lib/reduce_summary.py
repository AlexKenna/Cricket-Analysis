from lib.helpers import _write_dataframe_to_file, _read_dataframe

#------------------------------- Data Cleaning -------------------------------#
def reduce_summary():
    # Read and basic clean match data.
    summary = _read_summary_data()

    # Define the best features (selected from the Feature Extraction notebook).
    fields = ["Batter_ID", "Name", "International_One_Day_Batting_Average",
      "Domestic_One_Day_Match_Count", "Domestic_Test_Match_Count",
      "Domestic_One_Day_Runs_Per_Out_Average",
      "Domestic_One_Day_High_Score", "Domestic_Test_Four_Rate",
      "Domestic_Test_Runs_Per_Out_Average", "Domestic_Test_High_Score",
      "Domestic_One_Day_Start_Rate", "Domestic_One_Day_50_Rate",
      "Domestic_Test_50_Rate",
      "Domestic_One_Day_Average_Entering_Wicket",
      "Domestic_Test_Average_Entering_Wicket",
      "Domestic_T20_Average_Entering_Wicket",
      "Domestic_One_Day_Average_Ball_Count",
      "Domestic_Test_Average_Ball_Count",
      "Domestic_T20_Average_Ball_Count",
      "Domestic_One_Day_Team_Run_Contribution_Percent",
      "Domestic_One_Day_Team_High_Score_Percent",
      "Domestic_Test_Team_Run_Contribution_Percent",
      "Domestic_Test_Team_High_Score_Percent"]

    # Reduce the summary features.
    summary_reduced = summary[fields]

    # Write cleaned data to file.
    _write_dataframe_to_file(summary_reduced, "/Batter_Summary_Reduced.txt")


#-------------------------- Data Reading Functions ---------------------------#
# Read summary data.
def _read_summary_data():
    return _read_dataframe("/Batter_Summary.txt")
