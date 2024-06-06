import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.utils.eval import evaluate_regressor
from src.utils.plotting import plot_forecast


def v1():
    data = pd.read_csv("data/processed/every_hour_dataset.csv", index_col=0)

    # based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    # getting previous timeseries data
    df = pd.DataFrame()
    for col in data.columns:
        df[f"{col} t-2"] = data[col].shift(+2)
        df[f"{col} t-1"] = data[col].shift(+1)
        if col == "% Silica Concentrate":
            df[f"{col} t0"] = data[col]
    df.dropna(inplace=True)

    x = df.drop(columns="% Silica Concentrate t0")
    y = df["% Silica Concentrate t0"]

    return x, y


def v2():
    data = pd.read_csv("data/processed/every_hour_dataset.csv", index_col=0)

    # based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    # getting previous timeseries data
    df = pd.DataFrame()
    for col in data.columns:
        if col == "% Silica Concentrate":
            df[f"{col} t-3"] = data[col].shift(+3)
            df[f"{col} t-2"] = data[col].shift(+2)
            df[f"{col} t-1"] = data[col].shift(+1)
            df[f"{col} t0"] = data[col]
    df.dropna(inplace=True)

    x = df.drop(columns="% Silica Concentrate t0")
    y = df["% Silica Concentrate t0"]

    return x, y


def base(data, name="XGB Regressor"):
    x, y = data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.80, test_size=0.20, shuffle=False
    )

    model = XGBRegressor(random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    evaluate_regressor(y_test, y_pred, name=name)

    indexes = pd.to_datetime(x.index[int(len(x) * 0.8) :])
    plot_forecast(indexes, y_test, y_pred, name)

    return model


def main():
    model_v1 = base(v1(), "XGB Regressor - V1")
    model_v2 = base(v2(), "XGB Regressor - V2")


if __name__ == "__main__":
    main()
