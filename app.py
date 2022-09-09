import numpy as np
import streamlit as st
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def prepare(df):
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    df = df.dropna()
    data = df['Close'].values.reshape((-1, 1))
    return df, data


def prediction(num_prediction, data, model, lookback):
    data = data.reshape((-1))
    prediction_list = data[-lookback:]
    for _ in range(num_prediction):
        x = prediction_list[-lookback:].reshape((1, lookback, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    return prediction_list[lookback-1:]


def predict_dates(num_prediction, df):
    last_date = df['Date'].values[-1]
    return pd.date_range(last_date, periods=num_prediction+1).tolist()


def lstm(firm, data, regressor, generator, day):
    model = regressor.predict(generator)
    forecast = prediction(day, data, regressor, 1)
    dates = predict_dates(day, firm)
    plt.plot(firm['Date'], firm['Close'])
    plt.plot(firm['Date'][1:], model)
    plt.plot(dates, forecast)


tcs, tcs_data = prepare(pd.read_csv('TCS.csv'))
nestle, nestle_data = prepare(pd.read_csv('NESTLE.csv'))
ultra, ultra_data = prepare(pd.read_csv('ULTRA.csv'))
tcs_lstm, tcs_gen = tf.keras.models.load_model('tcs_model'), pk.load(open('tcs_gen.pkl', 'rb'))
nestle_lstm, nestle_gen = tf.keras.models.load_model('nestle_model'), pk.load(open('nestle_gen.pkl', 'rb'))
ultra_lstm, ultra_gen = tf.keras.models.load_model('ultra_model'), pk.load(open('ultra_gen.pkl', 'rb'))

st.write('Indraneel Dey')
st.write('Indian Institute of Technology, Madras')
st.title('Stock Price Forecasting')
st.write('Select the company whose closing stock price you wish to forecast')
company = st.selectbox('Select the company', ['TCS', 'Nestle', 'Ultratech'])
st.write('Enter the number of days after 12 August 2022 for which you wish to forecast')
days = st.number_input('Days', min_value=10)
if st.button('Forecast'):
    if company == 'TCS':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(lstm(tcs, tcs_data, tcs_lstm, tcs_gen, days))
    if company == 'Nestle':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(lstm(nestle, nestle_data, nestle_lstm, nestle_gen, days))
    if company == 'Ultratech':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(lstm(ultra, ultra_data, ultra_lstm, ultra_gen, days))
