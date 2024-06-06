import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator


def plot_forecast(indexes, y_original, y_pred, name):
    plt.style.use("ggplot")
    plt.figure(figsize=(16, 8))
    plt.plot(indexes, y_original, label="Original", color="#1F6DAC")
    plt.plot(indexes, y_pred, label=f"Predição com {name}", color="#EE781E")
    plt.title(f"Dados de Teste Original X Predição com {name}", loc="left", size=14)
    plt.ylabel("Concentrado Silica (%)")
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator(interval=5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    plt.show()
