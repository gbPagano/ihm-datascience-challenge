import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential

from src.utils.eval import evaluate_regressor
from src.utils.plotting import plot_forecast


def v1():
    data = pd.read_csv("data/processed/every_hour_dataset.csv", index_col=0)

    # based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    # getting previous timeseries data
    timesteps = 3
    df = pd.DataFrame()
    for col in data.columns:
        for i in range(1, timesteps + 1):
            df[f"{col} t-{i}"] = data[col].shift(+i)
        if col == "% Silica Concentrate":
            df[f"{col} t0"] = data[col]
    df.dropna(inplace=True)

    return df


def v2():
    data = pd.read_csv("data/processed/every_hour_dataset.csv", index_col=0)

    # based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    # getting previous timeseries data
    timesteps = 3
    df = pd.DataFrame()
    for col in data.columns:
        if col == "% Silica Concentrate":
            for i in range(1, timesteps + 1):
                df[f"{col} t-{i}"] = data[col].shift(+i)
                df[f"{col} t0"] = data[col]
    df.dropna(inplace=True)

    return df



def base(df, name="LSTM", interval=5, timesteps=3, epochs=4):
    x = df.drop(columns="% Silica Concentrate t0")
    y = df["% Silica Concentrate t0"].values.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler_x.fit_transform(x)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y = scaler_y.fit_transform(y)

    n_train = int(np.round(len(x) * 0.70))
    n_valid = int(np.round(len(x) * 0.15))

    x_train = scaled_x[:n_train]
    y_train = scaled_y[:n_train]
    x_val = scaled_x[n_train : (n_train + n_valid)]
    y_val = scaled_y[n_train : (n_train + n_valid)]
    x_test = scaled_x[(n_train + n_valid) :]

    assert (len(x_test) + len(x_train) + len(x_val)) == len(x)

    x_train = x_train.reshape(
        (x_train.shape[0], timesteps, x_train.shape[1] // timesteps)
    )
    x_val = x_val.reshape((x_val.shape[0], timesteps, x_val.shape[1] // timesteps))
    x_test = x_test.reshape((x_test.shape[0], timesteps, x_test.shape[1] // timesteps))

    model = Sequential(
        [
            Input((x_train.shape[1], x_train.shape[2])),
            LSTM(units=64, return_sequences=True),
            LSTM(units=64),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mae", metrics=[RootMeanSquaredError()])

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(x_val, y_val),
        shuffle=False,
    )

    y_test_original = y[(n_train + n_valid) :]
    y_pred_scaled = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    evaluate_regressor(y_test_original, y_pred, name=name)

    indexes = pd.to_datetime(df.index[(n_train + n_valid) :])
    plot_forecast(indexes, y_test_original, y_pred, name=name)

    return model


def main():
    model_v1 = base(v1(), "LSTM - V1", epochs=8)
    model_v2 = base(v2(), "LSTM - V2", epochs=8)


if __name__ == "__main__":
    main()
