#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import sys; sys.path.append('/home/evgenlazarev/.local/lib/python3.7/site-packages')


# In[3]:


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
# from business_models import hahn
from business_models.hahn import HahnDataLoader
hahn = HahnDataLoader(token='y1_AQAD-qJSK5nTAAADvgAAAAAAOG0BoSiLXJtKSCqBS4ApUeQSJklepHI')
import holidays

from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode()
sns.set_theme()

from tqdm.notebook import tqdm as ntqdm
ntqdm.pandas()


# ### FUNCTIONS

# In[4]:


import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


# In[5]:


def lineplot_errors(df, x, y_fact, y_pred, upper, lower):
    x, y_fact, y_pred, upper, lower = df[x], df[y_fact], df[y_pred], df[upper], df[lower]
    
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x, y_fact, label=x, color='orange')
    ax.plot(x, y_pred, label=x)
    ax.plot(x, lower, color='tab:blue', alpha=0.1)
    ax.plot(x, upper, color='tab:blue', alpha=0.1)
    ax.fill_between(x, lower, upper, alpha=0.2)

    plt.show()


# ### UPDATE DATA

# In[6]:


with open('/home/evgenlazarev/logdata8131-supply-forecast/deliveries_fact_agg_yt.sql') as _file:
    _query = _file.read()

hahn(_query)
print('>> SOURCE DATA HAS BEEN UPDATED.')


# ### GET DATA

# In[7]:


YT_PATH = '//home/taxi-delivery/analytics/dev/evgenlazarev/supply_forecast/{table_name}'

TABLE_NAMES = ['deliveries_fact_agg']


# In[8]:


SRC_DFS = {}
for _table_name in ntqdm(TABLE_NAMES):
    _tmp_df = hahn.read(YT_PATH.format(table_name=_table_name))
    print(f'> {_table_name} has been loaded.')
    _tmp_df.to_csv(f'{_table_name}.csv', index=False)
    print(f'> {_table_name} has been written to csv.')
    SRC_DFS[_table_name] = _tmp_df


# In[9]:


# Get data from .csv
SRC_DFS = {}
for _table_name in ntqdm(TABLE_NAMES):
    SRC_DFS[_table_name] = pd.read_csv(f'{_table_name}.csv')
    print(f'> {_table_name} has been loaded from csv.')

print('>> SOURCE DATA HAS BEEN LOADED.')

# In[10]:


for _df_name, _df in SRC_DFS.items():
    print(f'{_df_name}: {_df.shape}')


# ### DATA PREPARATION

# ##### Deliveries

# In[11]:


_dlvrs = SRC_DFS['deliveries_fact_agg'].copy()
_dlvrs.head()


# In[12]:


# utc_claim_created_dt: Int to Date
_dlvrs['utc_claim_created_dt'] = pd.to_datetime(_dlvrs['utc_claim_created_dt_str'], format="%Y-%m-%d")

# Fill empties in cargo_size_type
_dlvrs['cargo_size_type'] = _dlvrs['cargo_size_type'].replace('', 'unknown')
_dlvrs.groupby(by=['cargo_size_type']).agg({'claims_cnt':'sum'}).sort_values(by=['claims_cnt'], ascending=False)

# Remove strange types like 'M'
_dlvrs = _dlvrs[_dlvrs['cargo_size_type'].isin(['van', 'lcv_m', 'lcv_l', 'unknown'])]

# Keep only Russia
_dlvrs = _dlvrs[_dlvrs['source_country_name_en'] == 'Russia'].reset_index().drop(columns=['index'])

# Remove current month
# _dlvrs = _dlvrs[_dlvrs['utc_claim_created_dt'] < datetime.today().replace(day=1)]

_dlvrs.head()


# In[13]:


# VARIABLES
COUNTRY_NAME = 'Russia'
CITY_NAME = 'Moscow'
CARGO_SIZE_TYPE = 'van'

DATE_FIELD = 'utc_claim_created_dt'
Y_FIELD = 'cdr_confirmed_deliveries_cnt'
#

_data_main = _dlvrs[
    (_dlvrs['source_country_name_en'] == COUNTRY_NAME)
    & (_dlvrs['source_city_name_en'] == CITY_NAME)
    & (_dlvrs['cargo_size_type'] == CARGO_SIZE_TYPE)
][[DATE_FIELD, Y_FIELD]].rename(columns={DATE_FIELD:'ds', Y_FIELD:'y'}).reset_index().drop(columns=['index'])

_data_main.head()
sns.set(font_scale=1)
plt.figure(figsize=(20, 8), dpi=80)

_data_viz = _data_main.copy()
_data_viz['year'] = _data_viz['ds'].dt.year
_data_viz['pseudo_ds'] = _data_viz['ds'].apply(lambda x: x.replace(year = 2022))

ax = sns.lineplot(data=_data_viz, x='pseudo_ds', y='y', hue='year', palette=sns.color_palette())


# ### FORECAST

# #### Deliveries Forecast

# In[14]:


def prepare_data(_data, _country_name, _city_name, _cargo_size_type, _date_field='utc_claim_created_dt', _y_field='cdr_confirmed_deliveries_cnt'):
    # Fetch data
    _data_main = _data[
        (_data['source_country_name_en'] == _country_name)
        & (_data['source_city_name_en'] == _city_name)
        & (_data['cargo_size_type'] == _cargo_size_type)
    ][[_date_field, _y_field]].rename(columns={_date_field:'ds', _y_field:'y'}).reset_index().drop(columns=['index'])

    # Interpolate mobilization
    _model = Prophet(
        seasonality_mode='additive',
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=.0075,
        interval_width=.95,
    )
    _model.fit(_data_main[(_data_main['ds'] >= '2022-06-01') & (_data_main['ds'] < '2022-08-30')])

    _predicted_data = _model.make_future_dataframe(periods=90, freq='D')
    _predicted_data = _model.predict(_predicted_data)

    _data_main = _data_main.merge(_predicted_data[['ds', 'yhat']], on='ds', how='left')
    _data_main['y_wo_mob'] = [_row['y'] if (_row['ds'] < date(2022, 9, 5)) or (_row['ds'] > date(2022, 11, 20)) else _row['yhat'] for _, _row in _data_main.iterrows()]
    _data_main = _data_main.drop(columns='yhat')

    return _data_main


# ##### Predicting

# In[15]:


from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot

def model_fit(_data, x, y, _params):
    _data = _data[[x, y]].rename(columns={y:'y'})

    _model = Prophet(
        seasonality_mode=_params['seasonality_mode'],
        yearly_seasonality=_params['yearly_seasonality'], 
        weekly_seasonality=_params['weekly_seasonality'], 
        daily_seasonality=_params['daily_seasonality'],
        holidays=_params['holidays'],
        changepoints=_params['changepoints'],
        changepoint_prior_scale=_params['changepoint_prior_scale'],
        interval_width=.95,
    )
    _model.add_country_holidays(country_name='RU')
    _model.fit(_data)

    return _model

def model_predict(_model, periods, freq='D'):
    _predicted_data = _model.make_future_dataframe(periods=periods, freq=freq)
    _predicted_data = _model.predict(_predicted_data)

    return _predicted_data

def perform_crossvalidation(_model, _params_list):
    def cv_params_to_str(_params):
        return ', '.join([f'{_param}: {_value}' for _param, _value in _params.items()])

    _cv_results = {}
    for _params in _params_list:
#         print(f'> {_params}')
        _data_cv = cross_validation(_model, initial=_params['initial'], period=_params['period'], horizon=_params['horizon'])
        _data_pm = performance_metrics(_data_cv)
        _cv_results[cv_params_to_str(_params)] = _data_pm

    return _cv_results


# In[16]:


HOLIDAYS = pd.DataFrame(holidays.Russia(years=[2022, 2023]).items(), columns=['ds', 'holiday'])
HOLIDAYS['lower_window'], HOLIDAYS['upper_window'] = 0, 0


# In[17]:


HOLIDAYS = pd.concat([
  HOLIDAYS[['holiday', 'ds', 'lower_window', 'upper_window']],
  pd.DataFrame({
    'holiday': 'SMO',
    'ds': pd.date_range(start='20220222', end='20230501', freq='D'),
    'lower_window': 0,
    'upper_window': 0,
  }),
  # pd.DataFrame({
  #   'holiday': 'Mobilization',
  #   'ds': pd.date_range(start='20220920', end='20221010', freq='D'),
  #   'lower_window': 0,
  #   'upper_window': 0,
  # })
])


# In[18]:


# Tune Params
PROPHET_PARAMS_LIST = {
    'seasonality_mode': ['additive', 'multiplicative'],
    'yearly_seasonality': [True],
    'monthly_seasonality': [True],
    'weekly_seasonality': [True],
    'daily_seasonality': [False],
    'holidays': [HOLIDAYS],
    # 'changepoints': [['2022-12-15', '2022-12-31', '2023-01-01', '2023-09-01']],
    # 'changepoints': [['2022-12-15', '2022-12-31', '2023-01-01', '2023-09-01'], None],
    'changepoints': [None],
    'changepoint_range': [.8, .9, 1.0],
    'changepoint_prior_scale': [.01, .05, .1, .25, .5],
    # 'changepoint_prior_scale': [.1, .25, .5, .75, 1.0],
    # 'changepoint_prior_scale': list(np.arange(.01, .06, .01)) + [round(x, 3) for x in list(np.arange(.05, .3, .05))],
    # 'changepoint_prior_scale': [.01, .05, .1, .2, .25],
    'interval_width': [.95],
}

from sklearn.model_selection import ParameterGrid
_pgrid = ParameterGrid(PROPHET_PARAMS_LIST)
N_MODELS = sum([1 for _ in _pgrid])
N_MODELS


# In[19]:


# --- Disable annoying logs ---------------- #
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
# ------------------------------------------ #

START_DATE = '2022-01-01'
# END_DATE = '2023-09-18'
END_DATE = date.today().strftime('%Y-%m-%d')
COUNTRY_NAME = 'Russia'
CITIES_LIST = [
    'Moscow', 'Saint-Petersburg', 'Krasnodar', 'Novosibirsk', 'Rostov-on-Don',
    'Ekaterinburg', 'Chelyabinsk','Nizhny Novgorod', 'Voronezh', 'Kazan',
    'Samara', 'Ufa', 'Tyumen' #, 'Krasnogorsk', 'Lyuberci' -> Их нет в source_city_name_en. По всей видимости, они входят в Мск
]

_dlvrs = _dlvrs[_dlvrs['source_country_name_en'] == COUNTRY_NAME]
CITY_CARGO_SIZE_DF = _dlvrs[_dlvrs['source_city_name_en'].isin(CITIES_LIST)][['source_city_name_en', 'cargo_size_type']]\
    .drop_duplicates().reset_index().drop(columns='index')\
    .sort_values(by=['source_city_name_en', 'cargo_size_type'])
    
CV_PARAM_LIST = [
    # {'initial': '516 days', 'period': '14 days', 'horizon': '14 days'},
    {'initial': '516 days', 'period': '30 days', 'horizon': '30 days'},
]

print('>> MODEL HAS BEEN STARTED.')
_MODELS_RES = {}
for _, _row in CITY_CARGO_SIZE_DF.iterrows():
    _data = _dlvrs.copy()
    _data = prepare_data(
        _data=_data[(_data['utc_claim_created_dt'] >= START_DATE) & (_data['utc_claim_created_dt'] < END_DATE)],
        _country_name=COUNTRY_NAME,
        _city_name=_row['source_city_name_en'],
        _cargo_size_type=_row['cargo_size_type'],
        _date_field='utc_claim_created_dt',
        _y_field='cdr_confirmed_deliveries_cnt'
    )

    _tuning_res = {}
    for _idx, _params in enumerate(_pgrid):
        _model = model_fit(_data=_data, x='ds', y='y_wo_mob', _params=_params)
        _cv_results = perform_crossvalidation(_model=_model, _params_list=CV_PARAM_LIST)

        _tuning_res[_idx] = {
            '_params': _params,
            '_model': _model,
            '_cv_results': _cv_results,
        }
    
    _MODELS_RES[f"{_row['source_city_name_en']}-{_row['cargo_size_type']}"] = _tuning_res
print('>> MODEL TRAINING HAS BEEN COMPLETED.')

# ##### Finding best model

# In[20]:


def get_avg_mape(_cv_results):
    # _tmp = [_res['mape'].mean() for _res in _cv_results.values()]
    # return sum(_tmp) / len(_tmp)
    return list(_cv_results.values())[0].iloc[:-1]['mape'].mean()

def find_best_model(_models):
    return min(_models.items(), key=lambda _model: get_avg_mape(_model[1]['_cv_results']))


# In[21]:


_MODELS_RES_FINAL = {}
for _model_type_id, _models in _MODELS_RES.items():
    # for _idx, _info in _models.items():
        # for _params, _cv_results in _info['_cv_results'].items():
        #     print(f"> {_params}\n>> MAPE = {_cv_results['mape'].mean()}\n>> MAPE, % = {round(100*_cv_results['mape'].mean(), 2)}%")

    _best_model = find_best_model(_models)
    _MODELS_RES_FINAL[_model_type_id] = _best_model


# In[22]:


for _model_type_id, _model in _MODELS_RES_FINAL.items():
    print(f"> {_model_type_id} - MAPE = {round(100.0*get_avg_mape(_model[1]['_cv_results']), 2)}%")


# In[27]:


# Save models to file
import json
from prophet.serialize import model_to_json, model_from_json

for _model_type_id, _model in _MODELS_RES_FINAL.items():
    _model_params = _model[1]['_params']
    try:
        _model_params.pop('holidays')
    except:
        pass
    with open(f'/home/evgenlazarev/logdata8131-supply-forecast/models/deliveries-{_model_type_id}-params.txt', 'w') as f:
        f.write(json.dumps(_model_params))
    with open(f'/home/evgenlazarev/logdata8131-supply-forecast/models/deliveries-{_model_type_id}-model.json', 'w') as f:
        f.write(model_to_json(_model[1]['_model']))


# ##### Predict till the end of 2023

# In[25]:


_DELIVERIES_FORECAST_DF = pd.DataFrame()
for _key, _model_info in ntqdm(_MODELS_RES_FINAL.items()):
    _model = _model_info[1]['_model']
    _tmp = _model.make_future_dataframe(periods=180)
    _tmp = _tmp[
        # (_tmp['ds'] >= datetime.today().strftime('%Y-%m-%d'))
        (_tmp['ds'] >= END_DATE)
    ]
    _tmp = _model.predict(_tmp)
    _tmp = _tmp[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    _tmp['group_name'] = _key
    _tmp['_mape_cv_mean'] = get_avg_mape(_model_info[1]['_cv_results'])

    _DELIVERIES_FORECAST_DF = pd.concat([_DELIVERIES_FORECAST_DF, _tmp])

_DELIVERIES_FORECAST_DF = _DELIVERIES_FORECAST_DF[['group_name', '_mape_cv_mean', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# _DELIVERIES_FORECAST_DF['forecast_dt'] = datetime.today().strftime('%Y-%m-%d')
_DELIVERIES_FORECAST_DF['forecast_dt'] = END_DATE


# In[28]:


_DELIVERIES_FORECAST_DF.to_csv('/home/evgenlazarev/logdata8131-supply-forecast/models/DELIVERY_FORECAST.csv', index=False)

hahn.write(
    dataframe=_DELIVERIES_FORECAST_DF,
    full_path=YT_PATH.format(table_name='deliveries_forecast'),
    table_name='deliveries_forecast',
    types={
        'group_name': 'String',
        '_mape_cv_mean': 'Double',
        'ds': 'String',
        'yhat': 'Double',
        'yhat_lower': 'Double',
        'yhat_upper': 'Double',
        'forecast_dt': 'String',
    },
    if_exists='replace',
    force_drop=True
)

hahn.write(
    dataframe=_DELIVERIES_FORECAST_DF,
    full_path=YT_PATH.format(table_name='deliveries_forecast_hist'),
    table_name='deliveries_forecast_hist',
    types={
        'group_name': 'String',
        '_mape_cv_mean': 'Double',
        'ds': 'String',
        'yhat': 'Double',
        'yhat_lower': 'Double',
        'yhat_upper': 'Double',
        'forecast_dt': 'String',
    },
    if_exists='append'
)
print('>> FORECAST HAS BEEN SAVED TO YT.')

# In[ ]:


### Create datasource for the dashboard
with open('/home/evgenlazarev/logdata8131-supply-forecast/supply_forecast_dashboard_yt.sql') as _file:
    _dashboard_ds_query = _file.read()

hahn(_dashboard_ds_query)
print('>> DATALENS TABLE HAS BEEN CREATED TO YT.')