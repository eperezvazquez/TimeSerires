
import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from prophet import Prophet
from prophet.plot import plot_plotly

#Load data train and test A
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_a.index = pd.to_datetime(data_train_a.index)
data_train_b.index = pd.to_datetime(data_train_b.index)
epoch_t = 1529272655
real_t = pd.to_datetime(epoch_t, unit='s')
real_t
real_t.tz_localize('UTC').tz_convert('US/Pacific')

data_train_a.columns = ['CPU']
data_train_b.columns = ['CPU']
data_train_a.plot(title="CPU - DATE", figsize=(15,6))
data_train_b.plot(title="CPU - DATE", figsize=(15,6),color='red')
#cpu A
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
stepwise_model.fit(data_train_a)
stepwise_model.fit(data_train_a).plot_diagnostics(figsize=(15, 12))
plt.show()
future_forecast = stepwise_model.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_a.index,columns=['Prediction'])
pd.concat([data_test_a,future_forecast],axis=1).plot()

#cpu b
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
stepwise_model.fit(data_train_b).plot_diagnostics(figsize=(15, 12))
plt.show()
future_forecast = stepwise_model.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_b.index,columns=['Prediction'])
pd.concat([data_test_b,future_forecast],axis=1).plot()
