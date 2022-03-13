import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xlsxwriter
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


def extract_data(dataframe, factor):
    dataframe = dataframe[['Date', factor]]
    dataframe['Probability'] = [1.0 if x > 0 else 0.0 for x in dataframe[factor]]
    dataframe = dataframe.drop([factor], axis=1)
    dataframe = dataframe.rename(columns={'Probability': factor})
    return dataframe


def add_dates(dataframe, factor, forecast_length):
    end_point = len(dataframe)
    df = pd.DataFrame(index=range(forecast_length), columns=range(2))
    df.columns = ['Date', factor]
    dataframe = dataframe.append(df)
    dataframe = dataframe.reset_index(drop=True)
    x = dataframe.at[end_point - 1, 'Date']
    x = pd.to_datetime(x, format='%Y-%m-%d')
    for i in range(forecast_length):
        dataframe.at[dataframe.index[end_point + i], 'Date'] = x + timedelta(days=i+1)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%Y-%m-%d')
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Day'] = dataframe['Date'].dt.day
    return dataframe


def randomForest(dataframe, factor, forecast_length):
    new_dataframe = add_dates(dataframe, factor, forecast_length)
    new_dataframe = new_dataframe.reset_index(drop=True)

    end_point = len(dataframe)
    train = new_dataframe.loc[:end_point - 1, :]
    train_x = train[['Month', 'Day']]
    train_y = train[factor]

    rfr = RandomForestRegressor(n_estimators=100, random_state=1)
    fit = rfr.fit(train_x, train_y)

    # df = new_dataframe[['Month', 'Day']]
    # pred = fit.predict(df)
    # plt.bar(new_dataframe.Date, new_dataframe[factor])
    # plt.bar(new_dataframe.Date, pred)
    # plt.show()

    # print(confusion_matrix(dataframe[factor], pred[:-365].round()))
    # print(classification_report(dataframe[factor], pred[:-365].round()))
    # print('Accuracy:', (accuracy_score(dataframe[factor], pred[:-365].round()) * 100).__round__(2))

    forecast_values = []
    input_data = new_dataframe.loc[end_point:, ~new_dataframe.columns.isin(['Date', factor])]
    prediction = fit.predict(input_data)

    for i in range(end_point):
        forecast_values.append(np.NAN)
    for i in range(forecast_length):
        forecast_values.append(prediction[i])
    new_dataframe['forecast_'+factor] = forecast_values
    new_dataframe = new_dataframe.drop(columns=['Day', 'Month'])
    return new_dataframe


def save_plots(excel_data, factor, loc, folder):
    excel_data = excel_data.fillna(0.0)
    plt.figure(figsize=(14, 4))
    plt.bar(excel_data['Date'], excel_data[factor])
    plt.bar(excel_data['Date'], excel_data['forecast_'+factor])
    plt.xlabel('Date')
    plt.ylabel(factor+'_Probability')
    plt.legend(['Actual', 'Forecast'])
    plt.suptitle(loc + '_' + factor)
    plt.savefig(folder+'{}_{}'.format(loc, factor) + '.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close("all")


def save_excel(excel_data, sheet_name, forecast_length, loc, folder):
    excel_data = excel_data.fillna(' ')
    workbook = xlsxwriter.Workbook(folder+loc+'_'+sheet_name+'.xlsx')
    worksheet = workbook.add_worksheet(loc+'_'+sheet_name)
    bold = workbook.add_format({'bold': 1})
    headings = ['Date', sheet_name, 'forecast_'+sheet_name]
    worksheet.write_row('A1', headings, bold)
    date_format = workbook.add_format({'num_format': 'dd-mm-yyyy'})
    worksheet.write_column('A2', list(excel_data['Date']), date_format)
    worksheet.write_column('B2', list(excel_data[sheet_name][:len(excel_data)-forecast_length]))
    worksheet.write_column('C2', list(excel_data['forecast_'+sheet_name]))
    workbook.close()


input_path = 'input_data/'
result_path = 'results/'
input_excel_file = [(name[0], filename) for name in os.walk(input_path) for filename in name[2] if filename.endswith('.xlsx')]

if len(input_excel_file) != 1:
    raise ValueError('Please check input folder. Might be there are not any excel file or more than one excel file.')

breakdown_data = pd.read_excel(os.path.join(input_excel_file[0][0], input_excel_file[0][1]), sheet_name='Breakdown')
breakdown_data = breakdown_data.rename(columns={'Gen. Date': 'Date'})
breakdown_data['Date'] = pd.to_datetime(breakdown_data['Date'])

loc_key = breakdown_data['Loc. No.'][0]

breakdown_GF = extract_data(breakdown_data, 'GF')

breakdown_FM = extract_data(breakdown_data, 'FM')

breakdown_S = extract_data(breakdown_data, 'S')

breakdown_U = extract_data(breakdown_data, 'U')

length = len(breakdown_data)
prediction_period = 365

forecast_GF = randomForest(breakdown_GF, 'GF', prediction_period)
save_excel(forecast_GF, 'GF', prediction_period, loc_key, result_path+loc_key+'/')
save_plots(forecast_GF, 'GF', loc_key, result_path+loc_key+'/')

forecast_FM = randomForest(breakdown_FM, 'FM', prediction_period)
save_excel(forecast_FM, 'FM', prediction_period, loc_key, result_path+loc_key+'/')
save_plots(forecast_FM, 'FM', loc_key, result_path+loc_key+'/')

forecast_S = randomForest(breakdown_S, 'S', prediction_period)
save_excel(forecast_S, 'S', prediction_period, loc_key, result_path+loc_key+'/')
save_plots(forecast_S, 'S', loc_key, result_path+loc_key+'/')

forecast_U = randomForest(breakdown_U, 'U', prediction_period)
save_excel(forecast_U, 'U', prediction_period, loc_key, result_path+loc_key+'/')
save_plots(forecast_U, 'U', loc_key, result_path+loc_key+'/')
