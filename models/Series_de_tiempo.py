
#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#!pip install seaborn
#!pip install sklearn
#!pip install shutil
#!pip install sqlalchemy
#!pip install plotly
#!pip install pmdarima

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

#advertencias
import warnings
warnings.filterwarnings('ignore')

#Visualizacion de las columnas
pd.set_option('display.max_columns', None)

#Dataset links:

cpu_train_a =  pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv')

cpu_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv')

cpu_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv')

cpu_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv')

df_test = pd.concat([cpu_test_a,cpu_test_b], axis=0)

df_train = pd.concat([cpu_train_a,cpu_train_b], axis=0)

df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_train =  df_train.set_index('datetime')
df_train.head()

plt.figure(figsize=(18,9))
plt.plot(df_train.index, df_train['cpu'], linestyle = "-")
plt.xlabel = ('Dates')
plt.ylabel = ('registros_cpu')
plt.show()

df_train['rolling_sum'] = df_train.rolling(3).sum()
df_train.head(10)

df_train['rolling_sum_backfilled'] = df_train['rolling_sum'].fillna(method = 'backfill')
df_train.head()

df_train.index = pd.to_datetime(df_train.index)

import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt

df_train.plot(title = 'Registros de cpu del 27/01/2017 - 28/01/2017', figsize=(15,6))
plt.show()

import statsmodels.api as sm
from pylab import rcParams
from pmdarima.arima import auto_arima


data = pd.concat([cpu_train_a,cpu_train_b, cpu_test_a, cpu_test_b], axis=0)
data['datetime'] = pd.to_datetime(data['datetime'])
data =  data.set_index('datetime')
data.head()

data.plot(title="Registros de cpu ", figsize=(15,6))

exogenous_feature = ['rolling_sum_backfilled']

model = auto_arima(df_train_1.cpu, exogenous = df_train_1[exogenous_feature],
trace = True, error_action = 'ignore', suppress_warnings=True)
model.fit(df_train_1.cpu, exogenous_feature = df_train_1[exogenous_feature])

train = data.loc['2017-01-27 00:00:00':'2017-01-27 23:59:00']
test = data.loc['2017-01-28 00:00:00':'2017-01-28 23:59:00']
model.fit(train)

import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt

df_train.plot(title = 'Registros de cpu del 27/01/2017 - 28/01/2017', figsize=(15,6))
plt.show()

model.fit(train).plot_diagnostics(figsize=(15, 12))
plt.show()




