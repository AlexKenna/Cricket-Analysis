#---------------------------------- Imports ----------------------------------#
import numpy as np
import pandas as pd
from lib.helpers import _write_dataframe_to_file, _read_dataframe


#---------------------------------- Summary ----------------------------------#
def summarise_data():
    # Read in clean match and delivery data.
    match_data = _read_dataframe("/Matches_Clean.txt")
    delivery_data = _read_dataframe("/Deliveries_Clean.txt", low_memory=False)
    summary = _read_dataframe("/Batter_Summary.txt")

    # Rename dataframe columns.
    match_data.rename(columns={"Match Id": "Match_ID"}, inplace=True)
    delivery_data.rename(
        columns={"Striker Id": "Batter_ID", "Match Id": "Match_ID"}, inplace=True
    )

    # Add information to the summary.
    summary = _summarise_batter_attributes(summary, delivery_data)
    summary = _summarise_batter_odi_average(summary, delivery_data, match_data)
    summary = _summarise_batter_matches_played(
        summary, delivery_data, match_data)
    summary = _summarise_batter_outs(summary, delivery_data, match_data)
    summary = _summarise_batter_runs(summary, delivery_data, match_data)
    summary = _summarise_batter_milestones(summary, delivery_data, match_data)
    summary = _summarise_batter_batting_position(
        summary, delivery_data, match_data)
    summary = _summarise_batter_style(summary, delivery_data, match_data)
    summary = _summarise_batter_team_contribution(
        summary, delivery_data, match_data)

    # Write the summary to file.
    _write_dataframe_to_file(summary, "/Batter_Summary.txt")

#----------------------------- Summary Functions -----------------------------#
# Summarise each batters attributes.
def _summarise_batter_attributes(summary_data: pd.DataFrame, delivery_data: pd.DataFrame):
    # Extract important batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Get batter IDs and names.
    df = delivery_data[
        delivery_data["Batter_ID"].isin(batter_ids)
    ][["Batter_ID", "Striker", "Striker Hand"]]
    df.rename({"Striker": "Name", "Striker Hand": "Hand"},
              axis=1, inplace=True)

    # Drop duplicate data.
    df = df.drop_duplicates(["Batter_ID"])

    # Join data with summary table.
    summary_data = pd.merge(
        left=summary_data, right=df, on="Batter_ID", how="inner"
    )
    return summary_data

# Summarise each batters One-Day International average.
def _summarise_batter_odi_average(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for international games.
    match_ids = match_data[
        (match_data["Series"].str.contains("International")) &
        (match_data["Match Type Id"] == 1)
    ]["Match_ID"].tolist()

    # Extract deliveries to relevant batters.
    runs_df = delivery_data[
        (delivery_data["Batter_ID"].isin(batter_ids)) &
        (delivery_data["Match_ID"].isin(match_ids))
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]

    # Count the total number of runs for wach batter in ODI.
    runs_df = runs_df.groupby(
        "Batter_ID", as_index=False
    )["Bat Score"].sum().rename({"Bat Score": "Total_Runs"}, axis=1)

    # Extract the wickets of relevant batters.
    wickets_df = delivery_data[
        (delivery_data["Batter Out Id"].isin(batter_ids)) &
        (delivery_data["Match_ID"].isin(match_ids))
    ][["Batter_ID", "Match_ID", "Innings", "Batter Out Id"]]

    # Count the total number of outs for each batter.
    wickets_df = wickets_df.groupby(
        "Batter Out Id", as_index=False
    ).size().rename({"Batter Out Id": "Batter_ID", "size": "Out_Count"}, axis=1)

    # Determine the batting average of each batter.
    df = pd.merge(left=runs_df, right=wickets_df, on="Batter_ID", how="outer")
    df["Total_Runs"] = np.where(
        df["Out_Count"] < 1,
        df["Total_Runs"],
        df["Total_Runs"]/df["Out_Count"]
    )
    df.rename({
        "Total_Runs": "International_One_Day_Batting_Average"
    }, axis=1, inplace=True)
    df.drop(columns=["Out_Count"], inplace=True)

    # Join data into summary.
    summary_data = pd.merge(
        left=summary_data, right=df, on="Batter_ID", how="left"
    )
    return summary_data


# Summarise the matches played by each batter.
def _summarise_batter_matches_played(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df = delivery_data[
        delivery_data["Batter_ID"].isin(batter_ids)
    ][["Batter_ID", "Match_ID", "Team Batting ResultId", "Innings"]]
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day matches.
    games_df = df[df["Match Type Id"] == 1]
    summary_data = _summarise_matches(games_df, summary_data, "One_Day")

    # Summarise Test matches.
    games_df = df[df["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_matches(games_df, summary_data, "Test")

    # Summarise T20 matches.
    games_df = df[df["Match Type Id"] == 7]
    summary_data = _summarise_matches(games_df, summary_data, "T20")

    # Summarise the percentage of each format played by a batter.
    summary_data = _summarise_formats(summary_data)

    return summary_data


# Summarise the wickets of each batter.
def _summarise_batter_outs(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details
    df = delivery_data[
        (delivery_data["Batter Out Id"].isin(batter_ids)) &
        (~delivery_data["How Out"].isna())
    ][["Batter Out Id", "Match_ID", "Innings", "How Out"]]
    df = df.rename({"Batter Out Id": "Batter_ID"}, axis=1)
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day wickets.
    games_df = df[df["Match Type Id"] == 1]
    summary_data = _summarise_wickets(games_df, summary_data, "One_Day")

    # Summarise Test wickets.
    games_df = df[df["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_wickets(games_df, summary_data, "Test")

    # Summarise T20 wickets.
    games_df = df[df["Match Type Id"] == 7]
    summary_data = _summarise_wickets(games_df, summary_data, "T20")

    return summary_data


# Summarise the runs of each batter.
def _summarise_batter_runs(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df1 = delivery_data[
        delivery_data["Batter_ID"].isin(batter_ids)
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    df1 = pd.merge(left=match_df, right=df1, on="Match_ID")

    df2 = delivery_data[
        delivery_data["Batter Out Id"].isin(batter_ids)
    ][["Batter Out Id", "Match_ID", "Innings", "How Out"]]
    df2 = pd.merge(left=match_df, right=df2, on="Match_ID")

    # Summarise One-Day runs.
    games_df1 = df1[df1["Match Type Id"] == 1]
    games_df2 = df2[df2["Match Type Id"] == 1]
    summary_data = _summarise_run_spread(games_df1, summary_data, "One_Day")
    summary_data = _summarise_average(
        games_df1, games_df2, summary_data, "One_Day")
    summary_data = _summarise_high_score(games_df1, summary_data, "One_Day")

    # Summarise Test runs.
    games_df1 = df1[df1["Match Type Id"].isin([4, 5])]
    games_df2 = df2[df2["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_run_spread(games_df1, summary_data, "Test")
    summary_data = _summarise_average(
        games_df1, games_df2, summary_data, "Test")
    summary_data = _summarise_high_score(games_df1, summary_data, "Test")

    # Summarise T20 runs.
    games_df1 = df1[df1["Match Type Id"] == 7]
    games_df2 = df2[df2["Match Type Id"] == 7]
    summary_data = _summarise_run_spread(games_df1, summary_data, "T20")
    summary_data = _summarise_average(
        games_df1, games_df2, summary_data, "T20")
    summary_data = _summarise_high_score(games_df1, summary_data, "T20")
    return summary_data


# Summarise the milestones achieved by each batter (e.g., 50, 100, etc.).
def _summarise_batter_milestones(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df = delivery_data[
        (delivery_data["Batter_ID"].isin(batter_ids))
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day milestones.
    games_df = df[
        df["Match Type Id"] == 1
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    summary_data = _summarise_milestones(games_df, summary_data, "One_Day")

    # Summarise Test milestones.
    games_df = df[
        df["Match Type Id"].isin([4, 5])
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    summary_data = _summarise_milestones(games_df, summary_data, "Test")

    # Summarise T20 milestones.
    games_df = df[
        df["Match Type Id"] == 7
    ][["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    summary_data = _summarise_milestones(games_df, summary_data, "T20")
    return summary_data


# Summarise each batters batting position.
def _summarise_batter_batting_position(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df = delivery_data[(delivery_data["Batter_ID"].isin(batter_ids))][
        ["Batter_ID", "Match_ID", "Innings",
         "Cum Inning Balls", "Cum Inning Score", "Cum Inning Wickets"]
    ]
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day batting position.
    games_df = df[df["Match Type Id"] == 1]
    summary_data = _summarise_batting_position(
        games_df, summary_data, "One_Day")

    # Summarise Test batting position.
    games_df = df[df["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_batting_position(games_df, summary_data, "Test")

    # Summarise T20 batting position.
    games_df = df[df["Match Type Id"] == 7]
    summary_data = _summarise_batting_position(games_df, summary_data, "T20")
    return summary_data


# Summarise the batting style of each batter.
def _summarise_batter_style(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df = delivery_data[(delivery_data["Batter_ID"].isin(batter_ids))][[
        "Batter_ID", "Match_ID", "Innings", "Bat Score",
        "Inside Edge", "Outside Edge", "Play and Miss", "Hit on Pads",
        "Hit on Body", "Contact Error", "Opportunity"
    ]]
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day batting style.
    games_df = df[df["Match Type Id"] == 1]
    summary_data = _summarise_balls_per_innings(
        games_df, summary_data, "One_Day")
    summary_data = _summarise_false_shots(games_df, summary_data, "One_Day")
    summary_data = _summarise_strike_rate(games_df, summary_data, "One_Day")

    # Summarise Test batting style.
    games_df = df[df["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_balls_per_innings(games_df, summary_data, "Test")
    summary_data = _summarise_false_shots(games_df, summary_data, "Test")
    summary_data = _summarise_strike_rate(games_df, summary_data, "Test")

    # Summarise T20 batting style.
    games_df = df[df["Match Type Id"] == 7]
    summary_data = _summarise_balls_per_innings(games_df, summary_data, "T20")
    summary_data = _summarise_false_shots(games_df, summary_data, "T20")
    summary_data = _summarise_strike_rate(games_df, summary_data, "T20")
    return summary_data


# Summarise how each batter contributed to their teams.
def _summarise_batter_team_contribution(summary_data: pd.DataFrame, delivery_data: pd.DataFrame, match_data: pd.DataFrame):
    # Extract batter IDs.
    batter_ids = summary_data["Batter_ID"].tolist()

    # Extract match IDs and match types for domestic games.
    match_df = match_data[
        match_data["Series"].str.contains("Domestic")
    ][["Match_ID", "Match Type Id"]]

    # Extract deliveries to relevant batters and add match details.
    df = delivery_data[
        ["Batter_ID", "Match_ID", "Innings",
         "Bat Score", "Team Batting Id", "Cum Inning Score"]
    ]
    df = pd.merge(left=match_df, right=df, on="Match_ID")

    # Summarise One-Day team contribution.
    games_df = df[df["Match Type Id"] == 1]
    summary_data = _summarise_team_run_contribution(
        games_df, summary_data, "One_Day", batter_ids
    )
    summary_data = _summarise_team_highest_scorer(
        games_df, summary_data, "One_Day", batter_ids
    )

    # Summarise Test team contribution.
    games_df = df[df["Match Type Id"].isin([4, 5])]
    summary_data = _summarise_team_run_contribution(
        games_df, summary_data, "Test", batter_ids
    )
    summary_data = _summarise_team_highest_scorer(
        games_df, summary_data, "Test", batter_ids
    )

    # Summarise T20 team contribution.
    games_df = df[df["Match Type Id"] == 7]
    summary_data = _summarise_team_run_contribution(
        games_df, summary_data, "T20", batter_ids
    )
    summary_data = _summarise_team_highest_scorer(
        games_df, summary_data, "T20", batter_ids
    )
    return summary_data


#----------------------------- Helper Functions ------------------------------#
# Function to summarise a players matches.
def _summarise_matches(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels
    gamesLabel = "Domestic_{}_Match_Count".format(format_label)
    inningsLabel = "Domestic_{}_Innings_Count".format(format_label)
    winLabel = "Domestic_{}_Win_Rate".format(format_label)

    # Count the number of games and innings played.
    games_df = df.copy()
    games_df["Innings"] = (
        df["Match_ID"].astype(str) + df["Innings"].astype(str)
    ).astype(int)
    games_df = games_df.groupby(
        ["Batter_ID"], as_index=False
    ).agg({"Innings": "nunique", "Match_ID": "nunique"})
    games_df.rename(
        columns={"Match_ID": gamesLabel, "Innings": inningsLabel}, inplace=True
    )

    # Count the number of wins.
    wins_df = df[df["Team Batting ResultId"].isin([1, 2, 3, 4, 5, 15])]
    wins_df = wins_df.groupby(
        ["Batter_ID"], as_index=False
    ).agg({"Match_ID": "nunique"})
    wins_df.rename(columns={"Match_ID": winLabel}, inplace=True)

    # Add wins to game dataframe and determine win-rate.
    games_df = pd.merge(left=games_df, right=wins_df,
                        on="Batter_ID", how="left")
    games_df[winLabel].fillna(0, inplace=True)
    games_df[winLabel] = np.where(
        games_df[gamesLabel] < 1,
        games_df[gamesLabel],
        games_df[winLabel]/games_df[gamesLabel]
    )

    # Insert data into summary dataframe.
    summary = pd.merge(left=summary, right=games_df,
                       on="Batter_ID", how="left")
    summary.fillna({gamesLabel: 0, inningsLabel: 0, winLabel: 0}, inplace=True)
    summary = summary.astype(
        {gamesLabel: int, inningsLabel: int, winLabel: float}
    )

    return summary


# Function to summarise the percentage of each format played by a batter.
def _summarise_formats(summary: pd.DataFrame):
    # Format labels.
    one_day_match_count_label = "Domestic_One_Day_Match_Count"
    one_day_match_percent_label = "Domestic_One_Day_Match_Percent"
    one_day_innings_count_label = "Domestic_One_Day_Innings_Count"
    one_day_innings_percent_label = "Domestic_One_Day_Innings_Percent"
    test_match_count_label = "Domestic_Test_Match_Count"
    test_match_percent_label = "Domestic_Test_Match_Percent"
    test_innings_count_label = "Domestic_Test_Innings_Count"
    test_innings_percent_label = "Domestic_Test_Innings_Percent"
    t20_match_count_label = "Domestic_T20_Match_Count"
    t20_match_percent_label = "Domestic_T20_Match_Percent"
    t20_innings_count_label = "Domestic_T20_Innings_Count"
    t20_innings_percent_label = "Domestic_T20_Innings_Percent"

    # Extract necessary information.
    matches = summary[[
        "Batter_ID",
        one_day_match_count_label,
        test_match_count_label,
        t20_match_count_label
    ]].rename({
        one_day_match_count_label: one_day_match_percent_label,
        test_match_count_label: test_match_percent_label,
        t20_match_count_label: t20_match_percent_label
    }, axis=1)
    innings = summary[[
        "Batter_ID",
        one_day_innings_count_label,
        test_innings_count_label,
        t20_innings_count_label
    ]].rename({
        one_day_innings_count_label: one_day_innings_percent_label,
        test_innings_count_label: test_innings_percent_label,
        t20_innings_count_label: t20_innings_percent_label
    }, axis=1)

    # Determine the total number of games and innings for each batter.
    matches["Total_Games"] = (
        matches[one_day_match_percent_label] +
        matches[test_match_percent_label] +
        matches[t20_match_percent_label]
    )
    innings["Total_Innings"] = (
        innings[one_day_innings_percent_label] +
        innings[test_innings_percent_label] +
        innings[t20_innings_percent_label]
    )

    # Determine percentages.
    matches[one_day_match_percent_label] = np.where(
        matches["Total_Games"] < 1,
        matches["Total_Games"],
        matches[one_day_match_percent_label]/matches["Total_Games"]
    )
    matches[test_match_percent_label] = np.where(
        matches["Total_Games"] < 1,
        matches["Total_Games"],
        matches[test_match_percent_label]/matches["Total_Games"]
    )
    matches[t20_match_percent_label] = np.where(
        matches["Total_Games"] < 1,
        matches["Total_Games"],
        matches[t20_match_percent_label]/matches["Total_Games"]
    )
    matches.drop(columns=["Total_Games"], inplace=True)

    innings[one_day_innings_percent_label] = np.where(
        innings["Total_Innings"] < 1,
        innings["Total_Innings"],
        innings[one_day_innings_percent_label]/innings["Total_Innings"]
    )
    innings[test_innings_percent_label] = np.where(
        innings["Total_Innings"] < 1,
        innings["Total_Innings"],
        innings[test_innings_percent_label]/innings["Total_Innings"]
    )
    innings[t20_innings_percent_label] = np.where(
        innings["Total_Innings"] < 1,
        innings["Total_Innings"],
        innings[t20_innings_percent_label]/innings["Total_Innings"]
    )
    innings.drop(columns=["Total_Innings"], inplace=True)

    # Merge data into summary.
    summary = pd.merge(left=summary, right=matches, on="Batter_ID", how="left")
    summary = pd.merge(left=summary, right=innings, on="Batter_ID", how="left")
    return summary


# Function to summarise each batters wickets.
def _summarise_wickets(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels.
    not_out_label = "Domestic_{}_Not_Out_Percent".format(format_label)
    bowled_label = "Domestic_{}_Bowled_Percent".format(format_label)
    caught_label = "Domestic_{}_Caught_Percent".format(format_label)
    lbw_label = "Domestic_{}_LBW_Percent".format(format_label)
    hit_wicket_label = "Domestic_{}_Hit_Wicket_Percent".format(format_label)
    stumped_label = "Domestic_{}_Stumped_Percent".format(format_label)
    run_out_label = "Domestic_{}_Run_Out_Percent".format(format_label)
    handled_ball_label = "Domestic_{}_Handled_Ball_Percent".format(
        format_label)
    hit_ball_twice_label = "Domestic_{}_Hit_Ball_Twice_Percent".format(
        format_label)
    obstructed_field_label = "Domestic_{}_Obstructed_Field_Percent".format(
        format_label)
    timed_out_label = "Domestic_{}_Timed_Out_Percent".format(format_label)

    innings_label = "Domestic_{}_Innings_Count".format(format_label)

    # Determine how often a batter gets out in each way.
    how_out = df.groupby(
        ["Batter_ID", "How Out"]
    ).size().reset_index().rename({0: "Out_Percent"}, axis=1)
    how_out = pd.merge(
        left=summary[["Batter_ID", innings_label]],
        right=how_out,
        on="Batter_ID",
        how="left"
    )

    # Determine how often a batter is not out.
    not_out = how_out.groupby(
        "Batter_ID", as_index=False
    ).agg({"Out_Percent": "sum"})
    not_out = pd.merge(
        left=summary[["Batter_ID", innings_label]],
        right=not_out,
        on="Batter_ID",
        how="left"
    )
    not_out["Out_Percent"] = np.where(
        not_out[innings_label] < 1,
        not_out[innings_label],
        not_out[innings_label] - not_out["Out_Percent"]
    )
    not_out = not_out[not_out["Out_Percent"] > 0]
    not_out["How Out"] = "NO"
    how_out = how_out.append(not_out, ignore_index=True)

    # Determine percentage of each wicket occurring.
    how_out["Out_Percent"] = np.where(
        how_out[innings_label] < 1,
        how_out[innings_label],
        how_out["Out_Percent"]/how_out[innings_label]
    )
    how_out = how_out[["Batter_ID", "How Out", "Out_Percent"]]

    # Extract the percentage of each wicket occurrence.
    not_out = how_out.loc[
        how_out["How Out"] == "NO", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": not_out_label}, axis=1)
    bowled = how_out.loc[
        how_out["How Out"] == "B", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": bowled_label}, axis=1)
    caught = how_out.loc[
        how_out["How Out"] == "C", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": caught_label}, axis=1)
    lbw = how_out.loc[
        how_out["How Out"] == "LB", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": lbw_label}, axis=1)
    hit_wicket = how_out.loc[
        how_out["How Out"] == "HW", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": hit_wicket_label}, axis=1)
    stumped = how_out.loc[
        how_out["How Out"] == "S", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": stumped_label}, axis=1)
    run_out = how_out.loc[
        how_out["How Out"] == "RO", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": run_out_label}, axis=1)
    handled_ball = how_out.loc[
        how_out["How Out"] == "HB", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": handled_ball_label}, axis=1)
    hit_ball_twice = how_out.loc[
        how_out["How Out"] == "HT", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": hit_ball_twice_label}, axis=1)
    obstructed_field = how_out.loc[
        how_out["How Out"] == "OF", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": obstructed_field_label}, axis=1)
    timed_out = how_out.loc[
        how_out["How Out"] == "TO", ["Batter_ID", "Out_Percent"]
    ].rename({"Out_Percent": timed_out_label}, axis=1)

    # Merge all occurrences into a single dataframe.
    s = not_out
    s = pd.merge(left=s, right=bowled, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=caught, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=lbw, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=hit_wicket, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=stumped, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=run_out, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=handled_ball, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=hit_ball_twice, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=obstructed_field, on="Batter_ID", how="outer")
    s = pd.merge(left=s, right=timed_out, on="Batter_ID", how="outer")

    # Remove all NaNs.
    s.fillna({
        not_out_label: 0, bowled_label: 0, caught_label: 0,
        lbw_label: 0, hit_wicket_label: 0, stumped_label: 0,
        run_out_label: 0, handled_ball_label: 0, hit_ball_twice_label: 0,
        obstructed_field_label: 0, timed_out_label: 0
    }, inplace=True
    )

    # Merge wicket percentages into summary data.
    summary = pd.merge(left=summary, right=s, on="Batter_ID", how="left")
    return summary


# Function to summarise the spread of runs for each batter (e.g., % of dots, 1s, 2s,...).
def _summarise_run_spread(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels.
    dot_label = "Domestic_{}_Dot_Rate".format(format_label)
    one_label = "Domestic_{}_One_Rate".format(format_label)
    two_label = "Domestic_{}_Two_Rate".format(format_label)
    three_label = "Domestic_{}_Three_Rate".format(format_label)
    four_label = "Domestic_{}_Four_Rate".format(format_label)
    five_label = "Domestic_{}_Five_Rate".format(format_label)
    six_label = "Domestic_{}_Six_Rate".format(format_label)

    # Extract run and ball count for each batter.
    runs = df[
        ["Batter_ID", "Bat Score"]
    ].groupby(["Batter_ID", "Bat Score"], as_index=False).size()
    runs = runs.rename({"size": "Count"}, axis=1)
    balls = df[
        ["Batter_ID", "Bat Score"]
    ].groupby("Batter_ID", as_index=False).size()
    balls = balls.rename({"size": "Balls"}, axis=1)

    # Determine rate of occurrence of each run type.
    runs = pd.merge(left=runs, right=balls, on="Batter_ID", how="left")
    runs["Count"] = np.where(
        runs["Balls"] < 1,
        runs["Balls"],
        runs["Count"]/runs["Balls"]
    )

    # Extract run rates into individual dataframes.
    dots = runs[
        runs["Bat Score"] == 0][["Batter_ID", "Count"]
                                ].rename({"Count": dot_label}, axis=1)
    ones = runs[
        runs["Bat Score"] == 1][["Batter_ID", "Count"]
                                ].rename({"Count": one_label}, axis=1)
    twos = runs[
        runs["Bat Score"] == 2][["Batter_ID", "Count"]
                                ].rename({"Count": two_label}, axis=1)
    threes = runs[
        runs["Bat Score"] == 3][["Batter_ID", "Count"]
                                ].rename({"Count": three_label}, axis=1)
    fours = runs[
        runs["Bat Score"] == 4][["Batter_ID", "Count"]
                                ].rename({"Count": four_label}, axis=1)
    fives = runs[
        runs["Bat Score"] == 5][["Batter_ID", "Count"]
                                ].rename({"Count": five_label}, axis=1)
    sixes = runs[
        runs["Bat Score"] == 6][["Batter_ID", "Count"]
                                ].rename({"Count": six_label}, axis=1)

    # Combine run rates into single dataframe.
    s = dots
    s = pd.merge(left=s, right=ones, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=twos, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=threes, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=fours, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=fives, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=sixes, on="Batter_ID", how="left")

    # Fill empty values (e.q., when someone has not scored a particular run).
    s.fillna({
        dot_label: 0, one_label: 0, two_label: 0, three_label: 0,
        four_label: 0, five_label: 0, six_label: 0
    }, inplace=True
    )

    # Merge rates into the summary.
    summary = pd.merge(left=summary, right=s, on="Batter_ID", how="left")
    return summary


# Function to summarise the averages of each batter.
def _summarise_average(df1: pd.DataFrame, df2: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels
    runs_per_innings_label = "Domestic_{}_Runs_Per_Innings_Average".format(
        format_label)
    runs_per_out_label = "Domestic_{}_Runs_Per_Out_Average".format(
        format_label)
    innings_label = "Domestic_{}_Innings_Count".format(format_label)
    outs_label = "Domestic_{}_Out_Count".format(format_label)

    # Extract number of runs for each batter.
    runs = df1[["Batter_ID", "Bat Score"]]
    runs = runs.groupby(["Batter_ID"], as_index=False).sum()

    # Extract number of outs for each batter.
    outs = df2[
        ~df2["How Out"].isna()
    ]["Batter Out Id"].to_frame().rename({"Batter Out Id": "Batter_ID"}, axis=1)
    outs = outs.groupby(
        "Batter_ID", as_index=False
    ).size().rename({"size": outs_label}, axis=1)

    # Extract number of innings for each batter.
    innings = summary[["Batter_ID", innings_label]]

    # Merge innings and outs into runs dataframe.
    runs = pd.merge(left=runs, right=innings, on="Batter_ID", how="left")
    runs = pd.merge(left=runs, right=outs, on="Batter_ID", how="left")

    # Determine batting averages.
    runs[runs_per_innings_label] = np.where(
        runs[innings_label] < 1,
        runs["Bat Score"],
        runs["Bat Score"]/runs[innings_label]
    )
    runs[runs_per_out_label] = np.where(
        runs[outs_label] < 1,
        runs["Bat Score"],
        runs["Bat Score"]/runs[outs_label]
    )
    runs = runs[["Batter_ID", runs_per_innings_label, runs_per_out_label]]

    # Merge averages into summary.
    summary = pd.merge(left=summary, right=runs, on="Batter_ID", how="left")
    return summary


# Function to summarise the high score for each batter.
def _summarise_high_score(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format label
    high_score_label = "Domestic_{}_High_Score".format(format_label)

    # Determine high score for each batter.
    df = df[["Batter_ID", "Match_ID", "Innings", "Bat Score"]]
    df = df.groupby(
        ["Batter_ID", "Match_ID", "Innings"], as_index=False
    ).sum()
    df = df.groupby(
        ["Batter_ID"], as_index=False
    )["Bat Score"].max().rename({"Bat Score": high_score_label}, axis=1)

    # Add high score to summary.
    summary = pd.merge(left=summary, right=df, on="Batter_ID", how="left")
    return summary


# Function to summarise milestones.
def _summarise_milestones(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels.
    duck_label = "Domestic_{}_Duck_Rate".format(format_label)
    start_label = "Domestic_{}_Start_Rate".format(format_label)
    fifty_label = "Domestic_{}_50_Rate".format(format_label)
    hundred_label = "Domestic_{}_100_Rate".format(format_label)
    onefifty_label = "Domestic_{}_150_Rate".format(format_label)
    twohundred_label = "Domestic_{}_200_Rate".format(format_label)
    twofifty_label = "Domestic_{}_250_Rate".format(format_label)
    threehundred_label = "Domestic_{}_300_Rate".format(format_label)
    innings_label = "Domestic_{}_Innings_Count".format(format_label)

    # Determine runs per game.
    df = df.groupby(["Batter_ID", "Match_ID", "Innings"], as_index=False).sum()

    # Extract different milestones.
    ducks = df[
        df["Bat Score"] == 0
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": duck_label}, axis=1
    )
    starts = df[
        (df["Bat Score"] >= 1) & (df["Bat Score"] < 50)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": start_label}, axis=1
    )
    fifties = df[
        (df["Bat Score"] >= 50) & (df["Bat Score"] < 100)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": fifty_label}, axis=1
    )
    hundreds = df[
        (df["Bat Score"] >= 100) & (df["Bat Score"] < 150)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": hundred_label}, axis=1
    )
    onefifties = df[
        (df["Bat Score"] >= 150) & (df["Bat Score"] < 200)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": onefifty_label}, axis=1
    )
    twohundreds = df[
        (df["Bat Score"] >= 200) & (df["Bat Score"] < 250)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": twohundred_label}, axis=1
    )
    twofifties = df[
        (df["Bat Score"] >= 250) & (df["Bat Score"] < 300)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": twofifty_label}, axis=1
    )
    threehundreds = df[
        (df["Bat Score"] >= 300) & (df["Bat Score"] < 350)
    ].groupby("Batter_ID", as_index=False).size().rename(
        {"size": threehundred_label}, axis=1
    )

    # Merge milestones into a single dataframe.
    s = summary["Batter_ID"].to_frame()
    s = pd.merge(left=s, right=ducks, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=starts, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=fifties, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=hundreds, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=onefifties, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=twohundreds, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=twofifties, on="Batter_ID", how="left")
    s = pd.merge(left=s, right=threehundreds, on="Batter_ID", how="left")

    # Fill NaN fields.
    s.fillna({
        duck_label: 0, start_label: 0, fifty_label: 0,
        hundred_label: 0, onefifty_label: 0, twohundred_label: 0,
        twofifty_label: 0, threehundred_label: 0
    }, inplace=True
    )

    # Convert each milestone to a percentage.
    innings = summary[summary[innings_label]
                      != 0][["Batter_ID", innings_label]]
    s = pd.merge(left=s, right=innings, on="Batter_ID", how="left")
    s[duck_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[duck_label]/s[innings_label]
    )
    s[start_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[start_label]/s[innings_label]
    )
    s[fifty_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[fifty_label]/s[innings_label]
    )
    s[hundred_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[hundred_label]/s[innings_label]
    )
    s[onefifty_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[onefifty_label]/s[innings_label]
    )
    s[twohundred_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[twohundred_label]/s[innings_label]
    )
    s[twofifty_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[twofifty_label]/s[innings_label]
    )
    s[threehundred_label] = np.where(
        s[innings_label] < 1,
        s[innings_label],
        s[threehundred_label]/s[innings_label]
    )

    # Drop unnecessary columns.
    s.drop(columns=[innings_label], inplace=True)

    # Merge milestones into summary dataframe.
    summary = pd.merge(left=summary, right=s, on="Batter_ID", how="left")

    return summary

# Function to summarise the batting position of each batter.


def _summarise_batting_position(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels
    first_ball_label = "Domestic_{}_Average_Entering_Ball".format(format_label)
    entering_score_label = "Domestic_{}_Average_Entering_Score".format(
        format_label)
    entering_wickets_label = "Domestic_{}_Average_Entering_Wicket".format(
        format_label)

    # Extract first ball of each batter.
    df = df.groupby(
        ["Batter_ID", "Match_ID", "Innings"], as_index=False
    )[["Cum Inning Balls", "Cum Inning Score", "Cum Inning Wickets"]].first()

    # Summarise batting positions.
    df = df.groupby(
        ["Batter_ID"], as_index=False
    ).agg({
        "Cum Inning Balls": "mean",
        "Cum Inning Score": "mean",
        "Cum Inning Wickets": "median"
    })
    df.rename({
        "Cum Inning Balls": first_ball_label,
        "Cum Inning Score": entering_score_label,
        "Cum Inning Wickets": entering_wickets_label
    }, axis=1, inplace=True)

    # Merge summarised data into summary.
    summary = pd.merge(left=summary, right=df, on="Batter_ID", how="left")

    return summary


# Function to summarise the number of balls a batter has faced.
def _summarise_balls_per_innings(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels.
    ball_label = "Domestic_{}_Average_Ball_Count".format(format_label)
    innings_label = "Domestic_{}_Innings_Count".format(format_label)

    # Extract the total number of balls faced and innings played.
    balls = df[["Batter_ID", "Bat Score"]]
    balls = balls.groupby(
        ["Batter_ID"], as_index=False
    )["Bat Score"].size().rename({"size": ball_label}, axis=1)
    innings = summary[["Batter_ID", innings_label]]

    # Determine the average number of balls faced per innings.
    balls = pd.merge(left=balls, right=innings, on="Batter_ID", how="left")
    balls[ball_label] = np.where(
        balls[innings_label] < 1,
        balls[innings_label],
        balls[ball_label]/balls[innings_label]
    )
    balls = balls[["Batter_ID", ball_label]]

    # Merge average number of balls faced into summary data.
    summary = pd.merge(left=summary, right=balls, on="Batter_ID", how="left")

    return summary


# Function to summarise the false shots played by each batter.
def _summarise_false_shots(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels
    false_shot_label = "Domestic_{}_False_Shot_Rate".format(format_label)

    # Extract balls that were false shots.
    false_shots = df[
        (df["Inside Edge"] == "Y") |
        (df["Outside Edge"] == "Y") |
        (df["Play and Miss"] == "Y") |
        (df["Hit on Pads"] == "Y") |
        (df["Hit on Body"] == "Y") |
        (df["Contact Error"] == "Y") |
        (df["Opportunity"] == "Y")
    ][["Batter_ID", "Bat Score"]]
    false_shots = false_shots.groupby(
        ["Batter_ID"], as_index=False
    ).size().rename({"size": false_shot_label}, axis=1)

    # Determine the number of balls faced by each batter.
    balls = df[["Batter_ID", "Bat Score"]]
    balls = balls.groupby(
        ["Batter_ID"], as_index=False
    )["Bat Score"].size().rename({"size": "Balls"}, axis=1)

    # Determine the percentage of false shots per batter.
    false_shots = pd.merge(
        left=balls,
        right=false_shots,
        on="Batter_ID",
        how="left"
    )
    false_shots[false_shot_label] = np.where(
        false_shots["Balls"] < 1,
        false_shots["Balls"],
        false_shots[false_shot_label]/false_shots["Balls"]
    )
    false_shots = false_shots[["Batter_ID", false_shot_label]]

    # Add the false shot data to the summary.
    summary = pd.merge(
        left=summary,
        right=false_shots,
        on="Batter_ID",
        how="left"
    )
    return summary


# Function to summarise the strike rate of each batter.
def _summarise_strike_rate(df: pd.DataFrame, summary: pd.DataFrame, format_label: str):
    # Format labels
    strike_rate_label = "Domestic_{}_Strike_Rate".format(format_label)

    # Extract runs and balls.
    df = df[["Batter_ID", "Bat Score"]]
    df = df.groupby("Batter_ID", as_index=False).agg(
        {"Bat Score": ["sum", "count"]})
    df.columns = df.columns.to_flat_index()
    df.rename({df.columns[0]: "Batter_ID", df.columns[1]: strike_rate_label, df.columns[2]: "Balls"}, axis=1, inplace=True)

    # Summarise strike rate.
    df[strike_rate_label] = np.where(
        df["Balls"] < 1,
        df["Balls"],
        df[strike_rate_label]/df["Balls"]
    )
    df = df[["Batter_ID", strike_rate_label]]

    # Add strike rate to dataframe.
    summary = pd.merge(left=summary, right=df, on="Batter_ID", how="left")
    return summary


# Function to summarise the run contribution of each batter to their team.
def _summarise_team_run_contribution(df: pd.DataFrame, summary: pd.DataFrame, format_label: str, batter_ids: list):
    # Format labels.
    high_score_label = "Domestic_{}_Team_Run_Contribution_Percent".format(
        format_label)

    # Extract the final score for each team in each match/innings.
    team_total = df.groupby(
        ["Match_ID", "Innings", "Team Batting Id"], as_index=False).last()
    team_total = team_total[["Match_ID", "Innings", "Team Batting Id", "Cum Inning Score"]].rename(
        {"Cum Inning Score": "Team_Total"}, axis=1)

    # Align the total number of runs scored in each game with each batter.
    team_total = pd.merge(left=df, right=team_total, on=[
                          "Match_ID", "Innings", "Team Batting Id"], how="left")
    team_total = team_total[team_total["Batter_ID"].isin(
        batter_ids)][["Batter_ID", "Match_ID", "Innings", "Team_Total"]]
    team_total.drop_duplicates(
        ["Batter_ID", "Match_ID", "Innings"], inplace=True)

    # Determine the total number of runs each batters teams scored.
    team_total = team_total[["Batter_ID", "Team_Total"]]
    team_total = team_total.groupby("Batter_ID", as_index=False).sum()

    # Extract the number of runs each batter has scored.
    total_runs = df[df["Batter_ID"].isin(batter_ids)]
    total_runs = total_runs.groupby("Batter_ID", as_index=False)[
        "Bat Score"].sum()
    total_runs.rename({"Bat Score": high_score_label}, axis=1, inplace=True)

    # Determine each batters contribution to their teams as a percentage.
    total_runs = pd.merge(left=total_runs, right=team_total,
                          on="Batter_ID", how="left")
    total_runs[high_score_label] = np.where(
        total_runs["Team_Total"] < 1,
        total_runs["Team_Total"],
        total_runs[high_score_label]/total_runs["Team_Total"]
    )
    total_runs = total_runs[["Batter_ID", high_score_label]]

    # Merge team run contribution into summary.
    summary = pd.merge(left=summary, right=total_runs,
                       on="Batter_ID", how="left")
    return summary


# Function to summarise how often a batter is the high scorer of their team.
def _summarise_team_highest_scorer(df: pd.DataFrame, summary: pd.DataFrame, format_label: str, batter_ids: list):
    # Format labels
    high_score_label = "Domestic_{}_Team_High_Score_Percent".format(
        format_label)
    innings_label = "Domestic_{}_Innings_Count".format(format_label)

    # Extract the number of times each batter was the highest scorer.
    highscore = df.groupby(["Match_ID", "Innings", "Team Batting Id",
                           "Batter_ID"], as_index=False)["Bat Score"].sum()
    idx = highscore.groupby(["Match_ID", "Innings", "Team Batting Id"])[
        "Bat Score"].transform(max) == highscore["Bat Score"]
    highscore = highscore[idx]
    highscore = highscore[highscore["Batter_ID"].isin(batter_ids)]
    highscore = highscore.groupby("Batter_ID", as_index=False).size().rename({
        "size": high_score_label}, axis=1)

    # Convert number of highest scores to percentage.
    innings = summary[["Batter_ID", innings_label]]
    innings = innings[innings[innings_label] > 0]
    highscore = pd.merge(left=innings, right=highscore,
                         on="Batter_ID", how="left")
    highscore[high_score_label] = np.where(
        highscore[innings_label] < 1,
        highscore[innings_label],
        highscore[high_score_label]/highscore[innings_label]
    )
    highscore = highscore[["Batter_ID", high_score_label]]
    highscore.fillna({high_score_label: 0}, inplace=True)

    # Add to summary.
    summary = pd.merge(left=summary, right=highscore,
                       on="Batter_ID", how="left")
    return summary
