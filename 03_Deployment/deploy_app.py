import functions
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model 

import time
from datetime import datetime
from sklearn.metrics import r2_score


st.set_page_config(page_title='FEX Forecasting',
                layout="wide")

path = '~'
df = functions.import_main_df(f'{path}/02_Model/final_dataset.csv')
path2020 = f'{path}/02_Model/all_countries2020.csv'

first_column = ['Canada', 'Australia', 'Brazil', 'China', 'Denmark', 'Japan', 'Korea', 'Mexico', 'New Zealand', 
                'Norway', 'Sweden', 'Switzerland', 'South Africa', 'UK', 'US']

pageop = st.sidebar.radio(
    "Go to",
    ('Best Model', 'Beta'))

if pageop == 'Best Model':
    option = st.sidebar.selectbox(
        'Choose A Country',
        first_column)

    if option == 'Australia':
        country_code = 'AUD'
    if option == 'Brazil':
        country_code = 'BRL'
    if option == 'Switzerland':
        country_code = 'CHF'
    if option == 'Canada':
        country_code = 'CND'
    if option == 'China':
        country_code = 'CNY'
    if option == 'Denmark':
        country_code = 'DKK'
    if option == 'UK':
        country_code = 'GBP'
    if option == 'US':
        country_code = 'USD'
    if option == 'Japan':
        country_code = 'JPY'
    if option == 'Korea':
        country_code = 'KRW'
    if option == 'Mexico':
        country_code = 'MXN'
    if option == 'Norway':
        country_code = 'NOK'
    if option == 'New Zealand':
        country_code = 'NZD'
    if option == 'Sweden':
        country_code = 'SEK'
    if option == 'South Africa':
        country_code = 'ZAR'

    fitted_future_predictions = functions.import_main_df(f'data_files/fitted_future_predictions_{country_code}.csv')
    fitted_future_predictions['index'] = pd.to_datetime(fitted_future_predictions['index'])
    fitted_future_predictions = fitted_future_predictions.set_index('index')

    fitted_training_predictions = functions.import_main_df(f'data_files/fitted_training_predictions_{country_code}.csv')
    fitted_training_predictions['index'] = pd.to_datetime(fitted_training_predictions['index'])
    fitted_training_predictions = fitted_training_predictions.set_index('index')

    fitted_dataset_train = functions.import_main_df(f'data_files/fitted_dataset_train_{country_code}.csv')
    fitted_dataset_train['index'] = pd.to_datetime(fitted_dataset_train['index'])
    fitted_dataset_train = fitted_dataset_train.set_index('index')

    r2 = r2_score(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

    rmse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                    fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"], squared=False)

    mse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                    fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

    df2020 = functions.get_2020(fitted_future_predictions, country_code, path2020)

    start_time2 = st.slider("Select Start Date", min_value=datetime(2000, 1, 3), max_value=datetime(2020, 11, 30),
                        value=datetime(2001, 6, 14), format="MM/DD/YY", key=f'only_{country_code}')
    with st.beta_container():
        st.bokeh_chart(functions.bokeh_plotting(fitted_future_predictions, country_code, fitted_training_predictions, 
                        fitted_dataset_train, df2020, start_time2), use_container_width=False)
        
    expander = st.beta_expander("Evaluation Metrics")
    expander.write(f'R-squared: {r2}')
    expander.write(f'Root Mean Square Error: {rmse}')
    expander.write(f'Mean Square Error: {mse}')

###################

if pageop == 'Beta':

    fitted_modelCND = load_model('data_files/trained_model_on_CND.h5', compile=True)
    fitted_modelUSD = load_model('data_files/trained_model_on_USD.h5', compile=True)

    st.text('Choose a Base Model')

    left_column, right_column = st.beta_columns(2)

    pressed = left_column.checkbox('Canada')
    if pressed:

        fitted_model = fitted_modelCND
        country_code = 'CND'
        
        fitted_future_predictions = functions.import_main_df(f'data_files/fitted_future_predictions_{country_code}.csv')
        fitted_future_predictions['index'] = pd.to_datetime(fitted_future_predictions['index'])
        fitted_future_predictions = fitted_future_predictions.set_index('index')

        fitted_training_predictions = functions.import_main_df(f'data_files/fitted_training_predictions_{country_code}.csv')
        fitted_training_predictions['index'] = pd.to_datetime(fitted_training_predictions['index'])
        fitted_training_predictions = fitted_training_predictions.set_index('index')

        fitted_dataset_train = functions.import_main_df(f'data_files/fitted_dataset_train_{country_code}.csv')
        fitted_dataset_train['index'] = pd.to_datetime(fitted_dataset_train['index'])
        fitted_dataset_train = fitted_dataset_train.set_index('index')

        r2 = r2_score(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                    fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

        rmse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                        fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"], squared=False)

        mse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                        fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

        df2020 = functions.get_2020(fitted_future_predictions, country_code, path2020)

        start_time2 = st.slider("Select Start Date", min_value=datetime(2000, 1, 3), max_value=datetime(2020, 11, 30),
                            value=datetime(2001, 6, 14), format="MM/DD/YY", key=f'only_{country_code}')
        with st.beta_container():
            st.bokeh_chart(functions.bokeh_plotting(fitted_future_predictions, country_code, fitted_training_predictions, 
                            fitted_dataset_train, df2020, start_time2), use_container_width=False)
        
    pressed2 = right_column.checkbox('United States')
    if pressed2:

        fitted_model = fitted_modelUSD
        country_code = 'USD'

        fitted_future_predictions = functions.import_main_df(f'data_files/fitted_future_predictions_{country_code}.csv')
        fitted_future_predictions['index'] = pd.to_datetime(fitted_future_predictions['index'])
        fitted_future_predictions = fitted_future_predictions.set_index('index')

        fitted_training_predictions = functions.import_main_df(f'data_files/fitted_training_predictions_{country_code}.csv')
        fitted_training_predictions['index'] = pd.to_datetime(fitted_training_predictions['index'])
        fitted_training_predictions = fitted_training_predictions.set_index('index')

        fitted_dataset_train = functions.import_main_df(f'data_files/fitted_dataset_train_{country_code}.csv')
        fitted_dataset_train['index'] = pd.to_datetime(fitted_dataset_train['index'])
        fitted_dataset_train = fitted_dataset_train.set_index('index')

        r2 = r2_score(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                    fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

        rmse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                        fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"], squared=False)

        mse = functions.mean_squared_error(fitted_dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                        fitted_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])

        df2020 = functions.get_2020(fitted_future_predictions, country_code, path2020)

        start_time2 = st.slider("Select Start Date", min_value=datetime(2000, 1, 3), max_value=datetime(2020, 11, 30),
                            value=datetime(2001, 6, 14), format="MM/DD/YY", key=f'only_{country_code}')
        with st.beta_container():
            st.bokeh_chart(functions.bokeh_plotting(fitted_future_predictions, country_code, fitted_training_predictions, 
                            fitted_dataset_train, df2020, start_time2), use_container_width=False)
        

    if pressed:
        st.write('Select Another Country')
        option = st.selectbox('Please Select A Country', (' ', 'Australia', 'Japan', 'Korea', 'Mexico', 'New Zealand', 'Norway', 
                                                        'Sweden', 'Switzerland', 'UK', 'US'))
        
        if option != ' ':
            st.write('You have selected:', option)

        if option == 'Australia':
            country_var = 'AUD'
        if option == 'Japan':
            country_var = 'JPY'
        if option == 'Korea':
            country_var = 'KRW'
        if option == 'Mexico':
            country_var = 'MXN'
        if option == 'New Zealand':
            country_var = 'NZD'
        if option == 'Norway':
            country_var = 'NOK'
        if option == 'Sweden':
            country_var = 'SEK'
        if option == 'Switzerland':
            country_var = 'CHF'
        if option == 'UK':
            country_var = 'GBP'
        if option == 'US':
            country_var = 'USD'
        
        if option != ' ':
            start_time = st.slider("Select Start Date", min_value=datetime(2000, 1, 3), max_value=datetime(2020, 11, 30),
                        value=datetime(2001, 6, 14), format="MM/DD/YY", key=country_var)

            df_train, dates_list, var_list = functions.other_country_vars(df, country_var)

            df_matrixtrain, df_matrixtrain_scaled, scaler, scaler_y = functions.matrix_and_scale(df_train, var_list)

            n_future = 365
            n_past = 3
            X_train, y_train = functions.data_for_model(n_future, n_past, df_matrixtrain_scaled)

            future_datelist = functions.get_future_dates(dates_list, n_future)

            pred_future, pred_train = functions.get_predictions(fitted_model, X_train, n_future, n_past)

            y_pred_future, y_pred_train = functions.inverse_predictions(pred_future, pred_train, scaler_y)

            df_future_predictions, df_training_predictions = functions.pred_to_dataframe(y_pred_future, country_var, future_datelist, 
            y_pred_train, dates_list, n_past, n_future)

            # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
            df_training_predictions.index = df_training_predictions.index.to_series().apply(functions.datetime_to_timestamp)

            dataset_train = functions.dataset_for_visual(df_train, var_list, dates_list)

            df2020 = functions.get_2020(df_future_predictions, country_var, path2020)

            st.bokeh_chart(functions.bokeh_plotting(df_future_predictions, country_var, df_training_predictions, 
                dataset_train, df2020, start_time), use_container_width=False)

    elif pressed:
        st.write('Select Another Country')
        option = st.selectbox('Please Select A Country', (' ', 'Australia', 'Canada' 'Japan', 'Korea', 'Mexico', 'New Zealand', 'Norway', 
                                                        'Sweden', 'Switzerland', 'UK'))

        if option != ' ':
            st.write('You have selected:', option)

        if option == 'Australia':
            country_var = 'AUD'
        if option == 'Japan':
            country_var = 'JPY'
        if option == 'Korea':
            country_var = 'KRW'
        if option == 'Mexico':
            country_var = 'MXN'
        if option == 'New Zealand':
            country_var = 'NZD'
        if option == 'Norway':
            country_var = 'NOK'
        if option == 'Sweden':
            country_var = 'SEK'
        if option == 'Switzerland':
            country_var = 'CHF'
        if option == 'UK':
            country_var = 'GBP'
        if option == 'US':
            country_var = 'USD'
        
        if option != ' ':
            start_time = st.slider("Select Start Date", min_value=datetime(2000, 1, 3), max_value=datetime(2020, 11, 30),
                        value=datetime(2001, 6, 14), format="MM/DD/YY", key=country_var)

            df_train, dates_list, var_list = functions.other_country_vars(df, country_var)

            df_matrixtrain, df_matrixtrain_scaled, scaler, scaler_y = functions.matrix_and_scale(df_train, var_list)

            n_future = 365
            n_past = 3
            X_train, y_train = functions.data_for_model(n_future, n_past, df_matrixtrain_scaled)

            future_datelist = functions.get_future_dates(dates_list, n_future)

            pred_future, pred_train = functions.get_predictions(fitted_model, X_train, n_future, n_past)

            y_pred_future, y_pred_train = functions.inverse_predictions(pred_future, pred_train, scaler_y)

            df_future_predictions, df_training_predictions = functions.pred_to_dataframe(y_pred_future, country_var, future_datelist, 
            y_pred_train, dates_list, n_past, n_future)

            # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
            df_training_predictions.index = df_training_predictions.index.to_series().apply(functions.datetime_to_timestamp)

            dataset_train = functions.dataset_for_visual(df_train, var_list, dates_list)

            df2020 = functions.get_2020(df_future_predictions, country_var, path2020)

            st.bokeh_chart(functions.bokeh_plotting(df_future_predictions, country_var, df_training_predictions, 
                dataset_train, df2020, start_time), use_container_width=False)
