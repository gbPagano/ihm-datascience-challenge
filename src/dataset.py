import pandas as pd

RAW_DATA_PATH = "data/raw/MiningProcess_Flotation_Plant_Database.csv"


def process_raw_data():
    data = pd.read_csv(RAW_DATA_PATH, decimal=",", parse_dates=["date"])
    # filtering the initial data gap
    data = data.set_index("date")["2017-03-29 11:00:00":]

    # there is a missing record on 2017-04-10 00:00:00
    # considering that the missing record is the last sample of the day
    # we will fill this gap by placing the average of the day
    missing_sample = data.loc["2017-04-10 00:00:00"]

    new_row = pd.DataFrame(
        {pd.to_datetime("2017-04-10 00:00:00"): missing_sample.mean()}
    ).T

    part_1 = data[:"2017-04-09 23:00:00"]
    part_2 = data["2017-04-10 00:00:00":]

    data = pd.concat([part_1, new_row, part_2])

    # correcting the indexes
    new_index = pd.Series(
        pd.date_range(
            start="2017-03-29 12:00:00", end="2017-09-09 23:59:40", freq="20s"
        )
    )
    data.set_index(new_index, inplace=True)

    # grouping air flow and level columns
    df_airflow = data[data.columns[7:14]]
    df_level = data[data.columns[14:21]]

    data.drop(columns=df_airflow.columns, inplace=True)
    data.drop(columns=df_level.columns, inplace=True)
    data["Flotation Air Flow"] = df_airflow.mean(axis=1)
    data["Flotation Level"] = df_level.mean(axis=1)

    # ignoring the "% Iron Concentrate" column, as requested on Kaggle
    data.drop(columns="% Iron Concentrate", inplace=True)

    return data


def main():
    data = process_raw_data()

    # save 20s dataset
    every_20s_df = data
    # create every minute dataset
    every_minute_df = data.resample("min").mean()
    # create every hour dataset
    every_hour_df = data.resample("h").mean()

    every_20s_df.to_csv("data/processed/every_20s_dataset.csv")
    every_minute_df.to_csv("data/processed/every_minute_dataset.csv")
    every_hour_df.to_csv("data/processed/every_hour_dataset.csv")


if __name__ == "__main__":
    main()
