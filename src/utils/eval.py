from rich import print
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regressor(y, pred, name="Prediction Evaluation"):
    mae = mean_absolute_error(y, pred)
    rmse = root_mean_squared_error(y, pred)
    r2 = r2_score(y, pred)

    print(f"\n:: [cyan]{name}[/] ::")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
