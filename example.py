import streamlit as st
import pandas as pd

# загрузка плана отпарвления тс на каждый маршрут, финального и исходного состояния автопарка
# Здесь создание интерактивного дэшборда
plan = pd.read_csv("optimized_plan.csv")
final_fleet = pd.read_csv("Final_autopark.csv")
init_fleet = pd.read_csv("Example_autopark.csv")

plan["timestamp"] = pd.to_datetime(plan["timestamp"])

st.set_page_config(layout="wide")
st.title("Логистика")

col1, col2 = st.columns(2)

with col1:
    office = st.selectbox(
        "Office",
        sorted(plan["office_from_id"].unique())
    )

with col2:
    routes = sorted(
        plan[plan["office_from_id"] == office]["route_id"].unique()
    )
    route = st.selectbox(" Route", routes)

filtered = plan[
    (plan["office_from_id"] == office) &
    (plan["route_id"] == route)
].sort_values("timestamp")

st.subheader("Предсказаный объем, средний, их разница")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Предсказанный объем", int(filtered["volume"].sum()))

with col2:
    st.metric("Средний объем в такой же момент времени", int(filtered["manual_volume"].sum()))

with col3:
    diff = filtered["volume"].sum() - filtered["manual_volume"].sum()
    st.metric("Δ модель против среднего", int(diff))

st.subheader("модель против среднего")

if not filtered.empty:
    chart_df = filtered.set_index("timestamp")[["volume", "manual_volume"]]
    st.line_chart(chart_df)
else:
    st.warning("Нет данных для выбранного фильтра")

st.subheader("Использование ТС")

if not filtered.empty:
    trucks_df = filtered.set_index("timestamp")[[
        "small_trucks", "medium_trucks", "large_trucks"
    ]]
    st.line_chart(trucks_df)

st.subheader("План")
st.dataframe(filtered, use_container_width=True)

st.subheader("Сравнение состояний парка ТС")

col1, col2 = st.columns(2)

with col1:
    st.write("Иходное состояние ТС по офису")
    st.dataframe(
        init_fleet[init_fleet["office_from_id"] == office],
        use_container_width=True
    )

with col2:
    st.write("Конечное состояние ТС по офиу")
    st.dataframe(
        final_fleet[final_fleet["office_from_id"] == office],
        use_container_width=True
    )

