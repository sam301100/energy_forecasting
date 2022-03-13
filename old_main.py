import pandas as pd
import os
import itertools
import old_utils as ut
import warnings
warnings.filterwarnings("ignore")


input_path = r"input_data/"
result_path = 'results/'

input_excel_file = [(name[0], filename) for name in os.walk(input_path)
                               for filename in name[2] if filename.endswith('.xlsx')]
#print(input_excel_file)

if len(input_excel_file) != 1:
    raise ValueError('Plese check input folder. Might be there are not any excel file or more than one excel file.')

#read sheets in dataframe
active_generation_all = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]),sheet_name = 'Generation')
#print(active_generation_all)
wind_speed_all = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]),sheet_name = 'Wind_Speed')
#print(wind_speed_all)

for loc_key in set(active_generation_all.reset_index()['Loc. No.']):
    active_generation = active_generation_all[active_generation_all['Loc. No.'] == loc_key]
    wind_speed = wind_speed_all[wind_speed_all['Turbine'] == loc_key]
    #print(active_generation.shape,wind_speed.shape)
    #used for row and column
    if (active_generation.shape[0]) & (wind_speed.shape[0]) <= 10:
        raise ValueError('Not Sufficient dataset to forecast.')

    active_generation.set_index('Date',inplace=True)
    active_generation.set_index(pd.to_datetime(active_generation.index),inplace=True)
    #for using datetime format for 'Date' column

    y = active_generation['Daily Gen.(kWh)'].resample('MS').sum()
    y = y.fillna(' ')
    y.sort_index(inplace=True)
    #print(y)
    #'Date' and 'Daily Gen.(kWh)' column sum of each month generation and sort
    loc = input_excel_file[0][1].split(".")[0]
    #Combined_data_K497

    ##############################################
    p = d = q = range(0, 2)
    #print(p,d,q)
    #range(0, 2) range(0, 2) range(0, 2)
    pdq = list(itertools.product(p, d, q))
    #print(pdq)
    #[(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    #print(seasonal_pdq)
    #[(0, 0, 0, 12), (0, 0, 1, 12), (0, 1, 0, 12), (0, 1, 1, 12), (1, 0, 0, 12), (1, 0, 1, 12), (1, 1, 0, 12), (1, 1, 1, 12)]

    aic_values = ut.sarima_model(y,seasonal_pdq,pdq)
    # to determine p, d, q, m values by aic_values
    aic_values.sort(key=lambda x: x[2])
    param = aic_values[0][0]
    param_seasonal = aic_values[0][1]

    results_gen = ut.sarima_model_v1(y, param, param_seasonal)

    updated_results_gen = ut.forecast_plot(results_gen, 12, y, loc)
    updated_results_gen = updated_results_gen.fillna(' ')

    ###############################################
    wind_speed = wind_speed.reset_index().rename(columns={'period': 'Date'})
    wind_speed.set_index('Date',inplace=True)
    wind_speed.set_index(pd.to_datetime(wind_speed.index), inplace=True)
    z = wind_speed['Wind\nSpeed\n(m/s)']
    z = z.fillna(' ')
    z.sort_index(inplace=True)
    #print(z)
    # column period renamed as 'Date' and Wind Speed (m/s)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    aic_values = ut.sarima_model(z,seasonal_pdq,pdq)
    # to determine p, d, q, m values by aic_values
    aic_values.sort(key = lambda x: x[2])
    param = aic_values[0][0]
    param_seasonal = aic_values[0][1]

    results_wind = ut.sarima_model_v1(z, param, param_seasonal)

    updated_results_wind = ut.forecast_plot(results_wind,12, z, loc)
    updated_results_wind = updated_results_wind.fillna(' ')

    #print(updated_results_gen)
    #print(updated_results_wind)

    final_results = pd.merge(updated_results_gen,updated_results_wind)
    #print(final_results)
    ut.save_plots(loc,final_results)

#ut.clean_all(input_excel_file)

'''
Actual Process
1. Take excel data from both sheets
2. Get month and its generation or month and wind speed in 2 seperate data frames
3. Use SARIMAX model on data frames and forecast generation and wind speed for next months (as specified)
4. Save graphs and data in results file
'''