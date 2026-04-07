import pandas as pd
import numpy as np

# среднее по каждому офису из обучающей выборки используем для примерного заполнения автопарка в каждом офисе для примера
reas=pd.read_csv("reasoning.csv")



CAPACITY = {
    "small": 5,
    "medium": 10,
    "large": 20
}
np.random.seed(42)
fleet = []
for _, row in reas.iterrows():
    office = int(row["office_from_id"])
    avg_load = row["target_2h"]
    total_capacity_needed = avg_load * 3
    small = int(total_capacity_needed * np.random.uniform(0.2, 0.4) / CAPACITY["small"])
    medium = int(total_capacity_needed * np.random.uniform(0.3, 0.5) / CAPACITY["medium"])
    large = int(total_capacity_needed * np.random.uniform(0.2, 0.4) / CAPACITY["large"])

    fleet.append({
        "office_from_id": office,
        "small_trucks": small,
        "medium_trucks": medium,
        "large_trucks": large
    })

fleet_df = pd.DataFrame(fleet)
fleet_df.to_csv("Example_autopark.csv", index=False)

print(fleet_df)