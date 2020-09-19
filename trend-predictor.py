import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fbprophet import Prophet

from pytrends.request import TrendReq

# get google trends data from keyword list
@st.cache
def get_data(keyword):
    keyword = [keyword]
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keyword)
    df = pytrend.interest_over_time()
    df.drop(columns=['isPartial'], inplace=True)
    df.reset_index(inplace=True)
    df.columns = ["ds", "y"]
    return df

# make forecasts for a new period
def make_pred(df, periods):
    prophet_basic = Prophet()
    prophet_basic.fit(df)
    future = prophet_basic.make_future_dataframe(periods=periods)
    forecast = prophet_basic.predict(future)
    fig1 = prophet_basic.plot(forecast, xlabel="date", ylabel="trend", figsize=(10, 6))
    fig2 = prophet_basic.plot_components(forecast)
    forecast = forecast[["ds", "yhat"]]

    return forecast, fig1, fig2

# set streamlit page configuration
st.beta_set_page_config(page_title="Trend Predictor",
                               page_icon=":crystal_ball",
                               layout='centered',
                               initial_sidebar_state='auto')

# sidebar
st.sidebar.write("""
## Choose a keyword and a prediction period :dizzy:
""")
keyword = st.sidebar.text_input("Keyword", "Sunglasses")
periods = st.sidebar.slider('Prediction time in days:', 7, 365, 90)
details = st.sidebar.checkbox("Show details")

# main section
st.write("""
# Trend Predictor App :crystal_ball:
### This app predicts the **Google Trend** you want!
""")
st.write("Evolution of interest:", keyword)

df = get_data(keyword)
forecast, fig1, fig2 = make_pred(df, periods)

st.pyplot(fig1)
    
if details:
    st.write("### Details :mag_right:")
    st.pyplot(fig2)