import pandas as pd
import numpy as np
# загрузка предсказаний, пример автопарка, и трейна для вычисления средних
df = pd.read_csv("test_preds.csv")
fleet_df = pd.read_csv("Example_autopark.csv")
train_df = pd.read_parquet("train_team_track.parquet")





df["timestamp"] = pd.to_datetime(df["timestamp"])
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
train_df["hour"] = train_df["timestamp"].dt.hour

pattern = train_df.groupby(
    ["route_id", "hour"]
)["target_2h"].mean().reset_index()

pattern = pattern.rename(columns={"target_2h": "mean_volume"})



CAPACITY = {
    "small": 5,
    "medium": 10,
    "large": 20
}

def allocate_smart(volume, fleet_row, low=0.08, high=0.12):
    target_min = volume * (1 + low)
    target_max = volume * (1 + high)
    total_cap = (
            fleet_row["small_trucks"] * 5 +
            fleet_row["medium_trucks"] * 10 +
            fleet_row["large_trucks"] * 20
    )
    if total_cap == 0:
        return {"small": 0, "medium": 0, "large": 0}
    share_small = fleet_row["small_trucks"] * 5 / total_cap
    share_medium = fleet_row["medium_trucks"] * 10 / total_cap
    share_large = fleet_row["large_trucks"] * 20 / total_cap
    vol_small = volume * share_small
    vol_medium = volume * share_medium
    vol_large = volume * share_large

    s = int(np.ceil(vol_small / 5))
    m = int(np.ceil(vol_medium / 10))
    l = int(np.ceil(vol_large / 20))
    s=min(s,fleet_row["small_trucks"])
    m = min(m, fleet_row["small_trucks"])
    l = min(l, fleet_row["small_trucks"])


    def capacity(s, m, l):
        return s * 5 + m * 10 + l * 20
    cap = capacity(s, m, l)
    if target_min <= cap <= target_max:
        return {"small": s, "medium": m, "large": l}

    best = (s, m, l)
    best_score = float("inf")

    for ds in range(-3, 4):
        for dm in range(-3, 4):
            for dl in range(-3, 4):

                ns = max(0, min(fleet_row["small_trucks"], s + ds))
                nm = max(0, min(fleet_row["medium_trucks"], m + dm))
                nl = max(0, min(fleet_row["large_trucks"], l + dl))

                cap = capacity(ns, nm, nl)

                if cap < target_min:
                    continue

                over = cap - target_min

                prop_penalty = (
                        abs(ns - s) +
                        abs(nm - m) +
                        abs(nl - l)
                )

                score = over + 2 * prop_penalty

                if score < best_score:
                    best_score = score
                    best = (ns, nm, nl)

    return {
        "small": best[0],
        "medium": best[1],
        "large": best[2],
    }


fleet_state = fleet_df.set_index("office_from_id").copy()

plans = []

for (office, route), group in df.groupby(["office_from_id", "route_id"]):

    group = group.sort_values("timestamp").reset_index(drop=True)
    fleet_row = fleet_state.loc[office]

    values = group["y_pred"].values
    timestamps = group["timestamp"]

    vol_2h = values[3]
    hour_2h = timestamps[3].hour
    trucks_2h = allocate_smart(vol_2h, fleet_row)

    mean_row = pattern[
        (pattern["route_id"] == route) &
        (pattern["hour"] == hour_2h)
        ]

    manual_vol_2h = mean_row["mean_volume"].values[0] if not mean_row.empty else vol_2h
    manual_trucks_2h = allocate_smart(manual_vol_2h, fleet_row)

    fleet_state.loc[office, "small_trucks"] -= trucks_2h["small"]
    fleet_state.loc[office, "medium_trucks"] -= trucks_2h["medium"]
    fleet_state.loc[office, "large_trucks"] -= trucks_2h["large"]

    plans.append({
        "timestamp": timestamps[3],
        "office_from_id": office,
        "route_id": route,
        "horizon": "2h",
        "volume": vol_2h,
        "manual_volume": manual_vol_2h,
        "small_trucks": trucks_2h["small"],
        "medium_trucks": trucks_2h["medium"],
        "large_trucks": trucks_2h["large"],
        "manual_small": manual_trucks_2h["small"],
        "manual_medium": manual_trucks_2h["medium"],
        "manual_large": manual_trucks_2h["large"],
    })

    vol_4h = values[7]
    hour_4h = timestamps[7].hour
    trucks_4h = allocate_smart(vol_4h, fleet_row)

    mean_row = pattern[
        (pattern["route_id"] == route) &
        (pattern["hour"] == hour_4h)
        ]
    manual_vol_4h = mean_row["mean_volume"].values[0] if not mean_row.empty else vol_4h
    manual_trucks_4h = allocate_smart(manual_vol_4h, fleet_row)



    fleet_row["small_trucks"] -= trucks_4h["small"]
    fleet_row["medium_trucks"] -= trucks_4h["medium"]
    fleet_row["large_trucks"] -= trucks_4h["large"]

    plans.append({
        "timestamp": timestamps[7],
        "office_from_id": office,
        "route_id": route,
        "horizon": "4h",
        "volume": vol_4h,
        "manual_volume": manual_vol_4h,
        "small_trucks": trucks_4h["small"],
        "medium_trucks": trucks_4h["medium"],
        "large_trucks": trucks_4h["large"],
        "manual_small": manual_trucks_4h["small"],
        "manual_medium": manual_trucks_4h["medium"],
        "manual_large": manual_trucks_4h["large"],

    })

plan_df = pd.DataFrame(plans)
plan_df.to_csv("optimized_plan.csv", index=False)
final_fleet = fleet_state.reset_index()
final_fleet.to_csv("Final_autopark.csv", index=False)
print(final_fleet.head())
print(plan_df.head())