import pandas as pd
import matplotlib.pyplot as plt

# загружаем csv с предсказаниями модели по валидации( 5 последних процентов от трейна)
df = pd.read_csv("val_predictions.csv")

for col in ["y_pred_real", "y_true_real"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

df["timestamp"] = pd.to_datetime(df["timestamp"])
routes = [10,30,50,143,325]
# графики предсказаний и настоящих значений
plt.figure(figsize=(14, 6))

for route_id in routes:
    sample = df[df["route_id"] == route_id].sort_values("timestamp")

    line = plt.plot(
        sample["timestamp"],
        sample["y_true_real"],
        linestyle="-",
        label=f"Реальность {route_id}"
    )

    color = line[0].get_color()

    plt.plot(
        sample["timestamp"],
        sample["y_pred_real"],
        linestyle="--",
        marker="o",
        color=color,
        label=f"Предсказание модели  {route_id}"
    )

plt.title("Сравнение действительности и модели")
plt.xlabel("Время")
plt.ylabel("Таргет")
plt.legend()
plt.grid()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()