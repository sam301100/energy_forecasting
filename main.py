import pandas as pd
import os
import itertools
import utils as ut
import warnings
warnings.filterwarnings("ignore")


input_path = 'input_data/'
result_path = 'results/'

input_excel_file = [(name[0], filename) for name in os.walk(input_path) for filename in name[2] if filename.endswith('.xlsx')]

if len(input_excel_file) != 1:
    raise ValueError('Please check input folder. Might be there are not any excel file or more than one excel file.')

active_generation_all = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]), sheet_name='Generation')
wind_speed_all = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]), sheet_name='Wind_Speed')
breakdown_all = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]), sheet_name='Breakdown')

for loc_key in set(active_generation_all.reset_index()['Loc. No.']):

    active_generation = active_generation_all[active_generation_all['Loc. No.'] == loc_key]
    wind_speed = wind_speed_all[wind_speed_all['Turbine'] == loc_key]
    breakdown_data = breakdown_all[breakdown_all['Loc. No.'] == loc_key]

    if (active_generation.shape[0]) & (wind_speed.shape[0]) <= 12:
        raise ValueError('Not Sufficient dataset to forecast.')

    if breakdown_data.shape[0] <= 365:
        raise ValueError('Not Sufficient dataset to forecast.')

    if not os.path.exists(result_path+loc_key):
        os.mkdir(os.path.join(result_path, loc_key))

    ###############################################

    active_generation = active_generation.reset_index().rename(columns={'Daily Gen.(kWh)': 'Generation'})
    active_generation.set_index('Date', inplace=True)
    active_generation.set_index(pd.to_datetime(active_generation.index), inplace=True)

    active_generation = active_generation['Generation'].resample('MS').sum()
    active_generation = active_generation.fillna(' ')
    active_generation.sort_index(inplace=True)

    ###############################################

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    aic_values = ut.aic_sarima_model(active_generation, seasonal_pdq, pdq)
    aic_values.sort(key=lambda x: x[2])
    param = aic_values[0][0]
    param_seasonal = aic_values[0][1]

    results_gen = ut.sarima_model(active_generation, param, param_seasonal)

    updated_results_gen = ut.forecast_plot(results_gen, 12, active_generation, loc_key, result_path+loc_key+'/')

    ut.save_excel(updated_results_gen, 'Generation', 12, loc_key, result_path+loc_key+'/')

    ###############################################

    wind_speed = wind_speed.reset_index().rename(columns={'period': 'Date', 'Wind\nSpeed\n(m/s)': 'Wind_Speed'})
    wind_speed.set_index('Date', inplace=True)
    wind_speed.set_index(pd.to_datetime(wind_speed.index), inplace=True)

    wind_speed = wind_speed['Wind_Speed']
    wind_speed = wind_speed.fillna(' ')
    wind_speed.sort_index(inplace=True)

    ################################################

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    aic_values = ut.aic_sarima_model(wind_speed, seasonal_pdq, pdq)
    aic_values.sort(key=lambda x: x[2])
    param = aic_values[0][0]
    param_seasonal = aic_values[0][1]

    results_wind = ut.sarima_model(wind_speed, param, param_seasonal)

    updated_results_wind = ut.forecast_plot(results_wind, 12, wind_speed, loc_key, result_path+loc_key+'/')

    ut.save_excel(updated_results_wind, 'Wind_Speed', 12, loc_key, result_path+loc_key+'/')

    ################################################

    prediction_period = 365

    breakdown_data = breakdown_data.rename(columns={'Gen. Date': 'Date'})
    breakdown_data['Date'] = pd.to_datetime(breakdown_data['Date'])

    breakdown_GF = ut.extract_data(breakdown_data, 'GF')

    breakdown_FM = ut.extract_data(breakdown_data, 'FM')

    breakdown_S = ut.extract_data(breakdown_data, 'S')

    breakdown_U = ut.extract_data(breakdown_data, 'U')

    ################################################

    forecast_GF = ut.randomForest(breakdown_GF, 'GF', prediction_period)
    ut.save_excel(forecast_GF, 'GF', prediction_period, loc_key, result_path + loc_key + '/')
    ut.save_plots(forecast_GF, 'GF', loc_key, result_path + loc_key + '/')

    forecast_FM = ut.randomForest(breakdown_FM, 'FM', prediction_period)
    ut.save_excel(forecast_FM, 'FM', prediction_period, loc_key, result_path + loc_key + '/')
    ut.save_plots(forecast_FM, 'FM', loc_key, result_path + loc_key + '/')

    forecast_S = ut.randomForest(breakdown_S, 'S', prediction_period)
    ut.save_excel(forecast_S, 'S', prediction_period, loc_key, result_path + loc_key + '/')
    ut.save_plots(forecast_S, 'S', loc_key, result_path + loc_key + '/')

    forecast_U = ut.randomForest(breakdown_U, 'U', prediction_period)
    ut.save_excel(forecast_U, 'U', prediction_period, loc_key, result_path + loc_key + '/')
    ut.save_plots(forecast_U, 'U', loc_key, result_path + loc_key + '/')
