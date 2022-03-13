from datetime import timedelta
import numpy as np
import xlsxwriter
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def aic_sarima_model(dataframe, seasonal_pdq, pdq):
    aic = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(dataframe,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
# Enforee_stationary: Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model
# Enforce_invertibility: Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model.
                results = mod.fit()
                aic.append((param, param_seasonal, results.aic))
# AIC value is calculated on basis of the number of variable in model and the likelihood of model to fit better
            except:
                continue
    return aic


def sarima_model(dataframe, param, param_seasonal):
    mod = sm.tsa.statespace.SARIMAX(dataframe,
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    return results


def rearrange_results(data, forecast_data):
    excel_data = pd.DataFrame(data).reset_index()
    clm_name = pd.DataFrame(data).reset_index().columns[-1]
    # gives the column name of last column
    forecast = forecast_data.reset_index().rename(columns={'index': 'Date', 'predicted_mean': 'forecast_' + clm_name})
    # reset index and rename ('Date' is no longer the index)
    excel_data = pd.concat([excel_data, forecast])
    # excel_data = excel_data[['Date','Generation','forecast_Generation']]

    return excel_data


def forecast_plot(results, prediction_period, data, loc, folder):
    pred = results.predict(start=0, end=len(data)-1)

    # plt.plot(data)
    # plt.plot(pred)
    # plt.show()

    mape = []
    for x in range(len(data)):
        temp = abs(data.iloc[x]-pred[x])/abs(data.iloc[x])
        if temp < float('inf'):
            mape.append(temp)
    mape = np.mean(mape)
    print('Accuracy:', (100 - (mape * 100)).__round__(2))

    predict_uc = results.get_forecast(steps=int(prediction_period))
    # gives lower and upper bound of predicted values
    # pred = results.forecast(steps=int(prediction_period))
    # gives the actual mean of lower and upper bound of predicted values
    predict_ci = predict_uc.conf_int()
    # Returns the confidence interval of the fitted parameters

    clm_name = pd.DataFrame(data).reset_index().columns[-1]

    plt.figure(figsize=(14, 4))
    plt.plot(data)
    plt.plot(predict_uc.predicted_mean)
    plt.fill_between(predict_ci.index, predict_ci.iloc[:, 0], predict_ci.iloc[:, 1], color='dimgrey', alpha=0.25)
    plt.xlabel('Date')
    plt.ylabel(clm_name)
    plt.legend(['Actual', 'Forecast'])
    plt.suptitle(loc + '_' + clm_name)
    plt.savefig(folder + '{}_{}'.format(loc, clm_name) + '.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    results = rearrange_results(data, predict_uc.predicted_mean)

    return results


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

    df = new_dataframe[['Month', 'Day']]
    pred = fit.predict(df)

    # plt.bar(new_dataframe.Date, new_dataframe[factor])
    # plt.bar(new_dataframe.Date, pred)
    # plt.show()

    print()
    print(confusion_matrix(dataframe[factor], pred[:-forecast_length].round()))
    print(classification_report(dataframe[factor], pred[:-forecast_length].round()))
    print('Accuracy:', (accuracy_score(dataframe[factor], pred[:-forecast_length].round()) * 100).__round__(2))

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
