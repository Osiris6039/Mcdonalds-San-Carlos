
import streamlit as st
from components import input_data, event_history
from utils.model_utils import load_model, retrain_model, predict_next_10_days
from utils.data_utils import load_data, get_last_7_days, display_accuracy_graph

st.set_page_config(page_title="AI Sales & Customer Forecaster", layout="wide")

st.title("📈 AI Sales & Customer Volume Forecaster")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1e/McDonald%27s_logo.svg", width=120)
page = st.sidebar.radio("Navigate", ["Input Daily Data", "Event History", "10-Day Forecast"])

data = load_data()

if page == "Input Daily Data":
    input_data.render(data)

elif page == "Event History":
    event_history.render()

elif page == "10-Day Forecast":
    model = load_model()
    retrain_model(data)
    forecast_df = predict_next_10_days(data, model)
    st.subheader("📊 10-Day Forecast")
    st.dataframe(forecast_df)
    st.download_button("📥 Download Forecast", forecast_df.to_csv(index=False), file_name="10_day_forecast.csv")
    st.subheader("📉 Accuracy Graphs")
    display_accuracy_graph(data)
    st.markdown("Note: AI considers weather, patterns, and past events to project future trends.")
