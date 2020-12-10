# Import modules and packages
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Import from Sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import from Keras
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from tensorflow.keras.models import load_model 

# Import from Bokeh
from bokeh.layouts import layout
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, output_notebook, show
from bokeh.models import Span, Legend, BoxAnnotation, Toggle, HoverTool


## In order of appearance ##

def import_main_df(path): #path as a string 
    df = pd.read_csv(path)
    try:
        df = df.drop(columns='Unnamed: 0')
    except:
        pass

    return df

def dataset_to_train(df, var_list): 
    """returns a dataframe with only the variables specified in var_list"""

    df_train = df[var_list]
    print(f'Shape of training dataset = {df_train.shape}')
    
    #make a list of the dates (useful for visualization purposes)
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in df['Time Series']]
    print(f'Number of dates = {len(dates_list)}')
    print(f'Selected features: {var_list}')

    return df_train, dates_list

def other_country_vars(df_train, country_code):
    """creates var_list for the specified country_code and passes it through dataset_to_train"""

    var_list = ['Time Series']
    for i in list(df_train.columns):
        if i[0:3] == country_code:
            var_list.append(i)

    always_add_list = ['fc_year', 'price_gold', 'h1n1']
    for i in always_add_list:
        var_list.append(i)

    df_train, dates_list = dataset_to_train(df_train, var_list)

    return df_train, dates_list, var_list

def matrix_and_scale(df_train, var_list):
    """transforms dataframe to matrix and scales values"""

    df_matrixtrain = df_train[var_list[1:]].to_numpy()
    print(f'Shape of training matrix = {df_matrixtrain.shape}')

    scaler = StandardScaler()
    df_matrixtrain_scaled = scaler.fit_transform(df_matrixtrain)
    print(f'Shape of scaled training matrix = {df_matrixtrain_scaled.shape}')

    #separate scaler for y for when scaling is inversed 
    scaler_y = StandardScaler()
    scaler_y.fit_transform(df_matrixtrain[:, 3:4])

    return df_matrixtrain, df_matrixtrain_scaled, scaler, scaler_y

def data_for_model(n_future, n_past, df_matrixtrain_scaled):
    """reshapes the matrix to fit the X_train, y_train size required for LSTM"""

    X_train = []
    y_train = []

    for i in range(n_past, len(df_matrixtrain_scaled) - n_future + 1):
        X_train.append(df_matrixtrain_scaled[i - n_past:i, 0:df_matrixtrain_scaled.shape[1]])
        y_train.append(df_matrixtrain_scaled[i + n_future - 1:i + n_future, 3])

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(f'Shape of X_train = {X_train.shape}')
    print(f'Shape of y_train = {y_train.shape}')

    return X_train, y_train

def make_model(n_past, df_matrixtrain_scaled, n_units, n_dropout):
    """building the model with LSTM layers, dropout layers, and a final dense layer"""

    model = Sequential()

    model.add(LSTM(units = n_units, return_sequences = True, input_shape=(n_past, df_matrixtrain_scaled.shape[1])))
    model.add(Dropout(n_dropout))

    model.add(LSTM(units = n_units, return_sequences = True))
    model.add(Dropout(n_dropout))

    model.add(LSTM(units = n_units, return_sequences = True))
    model.add(Dropout(n_dropout))

    model.add(LSTM(units = n_units))
    model.add(Dropout(n_dropout))

    model.add(Dense(units = 1))

    model.compile(optimizer = "adam", loss = 'mse')

    return model

def fit_training(model, X_train, y_train, n_epoch, n_batch):
    """fitting the model"""

    history = model.fit(X_train, y_train, shuffle=False, epochs=n_epoch, batch_size=n_batch)

    return history, model

def get_future_dates(dates_list, n_future):
    """get dates for the forecasted range"""

    future_dates = pd.date_range(dates_list[-1], periods=n_future, freq='1d').tolist()

    #convert to datetime
    future_datelist = []
    for i in future_dates:
        future_datelist.append(i.date())
    
    return future_datelist

def get_predictions(model, X_train, n_future, n_past):
    """predictions and forecast values"""

    pred_train = model.predict(X_train[n_past:])
    pred_future = model.predict(X_train[-n_future:])

    return pred_future, pred_train

def inverse_predictions(pred_future, pred_train, scaler_y):
    """inversing the scaler operation for the y value"""
    
    y_pred_train = scaler_y.inverse_transform(pred_train)
    y_pred_future = scaler_y.inverse_transform(pred_future)


    return y_pred_future, y_pred_train

def pred_to_dataframe(y_pred_future, target_var, future_datelist, y_pred_train, dates_list, n_past, n_future):
    """converting the predicted & forecasted y values into a dataframe format"""

    df_future_predictions = pd.DataFrame(y_pred_future, columns=[f'{target_var}_USD']).set_index(pd.Series(future_datelist))
    df_training_predictions = pd.DataFrame(y_pred_train, columns=[f'{target_var}_USD']).set_index(pd.Series(dates_list[2 * n_past + n_future -1:]))
    
    return df_future_predictions, df_training_predictions

def datetime_to_timestamp(x):
    #where x is a datetime value
    return dt.datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

def dataset_for_visual(df_train, var_list, dates_list):
    """create dataframe useful for clean visualization"""

    dataset_train = pd.DataFrame(df_train, columns=var_list)
    dataset_train.index = dates_list
    dataset_train.index = pd.to_datetime(dataset_train.index)

    return dataset_train

def get_2020(path, df_future_predictions, country_code):
    """create dataframe of 2020 values for clean visualization"""

    df_2020 = pd.read_csv(path)
    df_2020 = df_2020[['Date', f'{country_code}_USD']]
    df_2020['Date'] = pd.to_datetime(df_2020['Date'])
    df_2020.rename(columns={f'{country_code}_USD':f'{country_code}_actual'}, inplace=True)
    pred_to_merge = df_future_predictions.reset_index()
    pred_to_merge['index'] = pd.to_datetime(pred_to_merge['index'])
    full_2020 = pd.merge(df_2020, pred_to_merge, left_on='Date', right_on='index')

    full_2020 = full_2020.set_index('Date')

    return full_2020

def plotting(start_plotting_date, df_future_predictions, df_training_predictions, dataset_train, target_var, full_2020):
    plot_2020 = '2019-12-31' #starting plot date for the 2020 values (both actual & forecasted)

    plt.plot(df_future_predictions.index, df_future_predictions[f'{target_var}_USD'], color='r', label='Predicted 2020 Forex')
    plt.plot(df_training_predictions.loc[start_plotting_date:].index, df_training_predictions.loc[start_plotting_date:][f'{target_var}_USD'], color='orange', label='Predicted Forex')
    
    plt.plot(full_2020.loc[plot_2020:].index, full_2020.loc[plot_2020:][f'{target_var}_actual'], color='green', label='Actual 2020 Forex')
    plt.plot(dataset_train.loc[start_plotting_date:].index, dataset_train.loc[start_plotting_date:][f'{target_var}_USD'], color='b', label='Actual Forex')
    
    plt.axvline(x = min(df_future_predictions.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predictions & Actual Forex Values')
    plt.xlabel('Dates')
    plt.ylabel('Exchange Rate Value')
    plt.xticks(rotation=45, fontsize=8)
    plt.show() 


def bokeh_plotting(df_future_predictions, country_code, df_training_predictions, dataset_train, full_2020):
    actual_data = dataset_train.loc['2001-06-14':].index
    actual = dataset_train.loc['2001-06-14':][f'{country_code}_USD']

    actual2020_dates = full_2020.loc['2020-01-02':].index
    actual2020 = full_2020.loc['2020-01-02':][f'{country_code}_actual']

    pred_date = df_future_predictions.index
    predictions = df_future_predictions[f'{country_code}_USD']

    training_data = df_training_predictions.loc['2001-06-14':].index
    training = df_training_predictions.loc['2001-06-14':][f'{country_code}_USD']

    # create a new plot with a datetime axis type
    p = figure(plot_width=1200, plot_height=500, x_axis_type="datetime", toolbar_location="above")

    # add renderers
    actual = p.line(actual_data, actual, color='cornflowerblue')
    actual_2020 = p.line(actual2020_dates, actual2020, color='mediumblue')

    predictions_2020 = p.line(pred_date, predictions, color='deeppink')
    predictions = p.line(training_data, training, color='firebrick')

    #add a line 
    line2020 = Span(location=min(pred_date), dimension='height', line_color='#3e3a3a',
                                            line_dash='dashed', line_width=3)
    p.add_layout(line2020)

    #legend outside the graph 
    legend = Legend(items=[("2020 Predictions", [predictions_2020]), ("Predictions" , [predictions]),
                                    ("Actual" , [actual]), ("2020 Actual" , [actual_2020])], location="center")
    p.add_layout(legend, 'right')
    p.legend.click_policy="hide"

    #add the background colour
    green_box = BoxAnnotation(left=df_future_predictions.index[2], fill_color='thistle', fill_alpha=0.2)
    p.add_layout(green_box)

    #change the information for the hovering 
    hover = HoverTool(tooltips=[('Date', '$x{%F}'), ('FEX','$y')],
                    formatters={'$index':'datetime', '$x': 'datetime'})
    p.add_tools(hover)

    p.grid.grid_line_alpha = 0

    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Exchange Rate'
    p.title.text = f'{country_code} Exchange Rate Predictions'

    p.ygrid.band_fill_alpha = 0.2
    p.ygrid.band_fill_color = "ivory"

    show(p)

def eval_metrics(dataset_train, df_training_predictions, country_code):
    scaler2 = StandardScaler()

    reg_rmse = mean_squared_error(dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                                    df_training_predictions.loc["2001-06-20":][f"{country_code}_USD"], squared=False)
    print(f'RMSE: {reg_rmse}')

    scale_rmse = mean_squared_error(scaler2.fit_transform(np.array(dataset_train.loc["2001-06-20":][f"{country_code}_USD"]).reshape(-1,1)), 
                    scaler2.fit_transform(np.array(df_training_predictions.loc["2001-06-20":][f"{country_code}_USD"]).reshape(-1,1)), squared=False)
    print(f'RMSE of Scaled Values: {scale_rmse}')

    reg_r2 = r2_score(dataset_train.loc["2001-06-20":][f"{country_code}_USD"], 
                        df_training_predictions.loc["2001-06-20":][f"{country_code}_USD"])
    print(f'R2_Score: {reg_r2}')

    scale_r2 = r2_score(scaler2.fit_transform(np.array(dataset_train.loc["2001-06-20":][f"{country_code}_USD"]).reshape(-1,1)), 
                        scaler2.fit_transform(np.array(df_training_predictions.loc["2001-06-20":][f"{country_code}_USD"]).reshape(-1,1)))
    print(f'R2_Score of Scaled Values: {scale_r2}')
