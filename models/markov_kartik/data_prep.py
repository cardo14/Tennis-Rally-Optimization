import pandas as pd

points_data = pd.read_csv("data/raw/charting-m-points-2020s.csv")

match_data = pd.read_csv("data/raw/charting-m-matches.csv")

matches = match_data[['match_id', 'Surface', 'Date']]

data = points_data.merge(matches, on='match_id', how='left')

data = data[data["Date"].astype(str).str.fullmatch(r"\d{8}")]

data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

hard_data = data[
    (data['Surface'] == 'Hard') &
    (data['Date'].between('2022-01-01', '2024-12-31'))
]

hard_data = hard_data.sort_values(by=['match_id', 'Pt']).reset_index(drop=True)

cols = [
    "match_id", "Pt", "Svr",
    "1st", "2nd", "Notes",
    "PtWinner", "Surface", "Date"
]

model_df = hard_data[cols].copy()

model_df.to_csv(
    "data/processed/points_hard_2022_2024.csv",
    index=False
)