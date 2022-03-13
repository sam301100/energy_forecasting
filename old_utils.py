import numpy as np
import xlsxwriter
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


def aic_sarima_model(dataframe, seasonal_pdq, pdq):
    AIC = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(dataframe,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                #Enforee_stationary: Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model
                #Enforce_invertibility: Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model.
                results = mod.fit()
                AIC.append((param, param_seasonal, results.aic))
            except:
                continue
    return AIC


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
    #print(clm_name)
    forecast = forecast_data.reset_index().rename(columns={'index': 'Date', 'predicted_mean': 'forecast_'+clm_name})
    #print(forecast)
    excel_data = pd.concat([excel_data, forecast])
    #excel_data = excel_data[['Date','Generation','forecast_Generation']]
    
    return excel_data


def forecast_plot(results, prediction_period, y, loc, folder):
    #yhat=results.predict(start=0, end=len(y)-1)
    #plt.plot(y)
    #plt.plot(yhat)
    #plt.show()

    #mape = []
    #for x in range(len(y)):
    #    t = abs(y.iloc[x]-yhat[x])/abs(y.iloc[x])
    #    if t <1000000000000000000 :
    #        mape.append(t)
    #mape = np.mean(mape)
    #print(mape)

    pred_uc = results.get_forecast(steps=int(prediction_period))
    # gives lower and upper bound of predicted values
    #pred = results.forecast(steps=int(prediction_period))
    # gives the actual mean of lower and upper bound of predicted values
    pred_ci = pred_uc.conf_int()
    #Returns the confidence interval of the fitted parameters

    clm_name = pd.DataFrame(y).reset_index().columns[-1]

    plt.figure(figsize=(14, 4))
    plt.plot(y)
    plt.plot(pred_uc.predicted_mean)
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='dimgrey', alpha=0.25)
    plt.xlabel('Date')
    plt.ylabel(clm_name)
    plt.legend(['Actual', 'Forecast'])
    plt.suptitle(loc + '_' + clm_name)
    plt.savefig(folder + '{}_{}'.format(loc, clm_name) + '.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    results = rearrange_results(y, pred_uc.predicted_mean)
    
    return results


def save_excel(excel_data, sheet_name, forecast_length, loc, folder):
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

