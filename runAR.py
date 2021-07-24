import glob
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import AR_Adam
import warnings
warnings.filterwarnings('ignore')


def getPICI_W(observed, upper, lower):
    p_upper = upper
    p_lower = lower
    ci = []
    for i in range(len(p_upper)):
        c = p_lower[i] <= observed[i] and observed[i] <= p_upper[i]
        if c == True:
            ci.append(1)
        else:
            ci.append(0)
    return sum(ci)/len(p_upper), sum(p_upper-p_lower)/len(p_upper)

def runAR_Adam(trainDf,county,horizon,target):

    vtest_metrics = {'County':[], 'Horizon':[],
                            'R2':[], 'RMSE':[],'MAE':[], 'MAPE':[], 'PICP':[], 'MPIW':[]}


    vout_data = {'Date':[],'County':[], 'Horizon':[],
                            'Predicted':[], 'Observed':[],'pred_lower':[], 'pred_upper':[],'Season':[]}

    vars_ = ['LST', 'Rainfall','SoilMoist',f'{target}','Season','Date']

    if target == 'VCI':
        data_df0 = AR_Adam.addSeason(trainDf)
        scale_df1, means = AR_Adam.scaleValuesV(data_df0.loc[:,vars_])
    if target == 'NDVI_Anom':
        data_df0 = AR_Adam.addSeason(trainDf)
        scale_df1 = AR_Adam.scaleValuesA(data_df0.loc[:,vars_])

    X_train, y_train = AR_Adam.fcast_train_testDF_AR(scale_df1, p_order0=0 ,
                                                        p_order1=0, p_order2=0,
                                                        q_order=3, s_lags=1, date=1, target_var=target,
                                                        f_horizon=horizon)



    print(X_train.columns)
    strt = np.arange(0, len(X_train), 200)

    for t, h in enumerate(strt):
        sub_xdata = X_train[h:h+500]
        sub_ydata = y_train[h:h+500]
        print(f'{county}: Step {t+1} horizons {horizon} for weeks ahead.....')
        print(sub_xdata.iloc[:,0].shape)

        train_len = len(sub_xdata[:-100])
        respredict, ypred, nopredict, forecast = AR_Adam.astro_predict_one(Y=sub_ydata.values, X=sub_xdata.iloc[:,0].values, nlags=3, trainlength=train_len)


        if target=='VCI':
            new_preds = ypred+means
            new_ytest = sub_ydata.values[-100:]+means

        elif target=='NDVI_Anom':
            new_preds = ypred
            new_ytest = sub_ydata.values[-100:]


        vout_data['County'].extend(list(np.repeat(county,len(new_preds))))
        vout_data['Horizon'].extend(list(np.repeat(horizon,len(new_preds))))
        vout_data['Predicted'].extend(new_preds)
        vout_data['Observed'].extend(new_ytest)

        aic, bic, r2, mae, mape,rmse = AR_Adam.calcIC(new_ytest, new_preds, 4)
        vtest_metrics['County'].append(county)
        vtest_metrics['Horizon'].append(horizon)
        vtest_metrics['R2'].append(r2)
        vtest_metrics['MAE'].append(mae)
        vtest_metrics['RMSE'].append(rmse)
        vtest_metrics['MAPE'].append(mape)

        interval = 1.96 * rmse

        lower, upper = new_preds - interval, new_preds + interval

        vout_data['pred_lower'].extend(lower)
        vout_data['pred_upper'].extend(upper)
        vout_data['Date'].extend(sub_xdata['Date_lag_0'][-100:])
        vout_data['Season'].extend(sub_xdata['Season_lag_0'][-100:])

        picp, mpiw = getPICI_W(new_ytest, upper, lower)

        vtest_metrics['PICP'].append(picp)
        vtest_metrics['MPIW'].append(mpiw)


    return pd.DataFrame(vtest_metrics), pd.DataFrame(vout_data)


county_grp = ['Baringo','Embu','Garissa','Isiolo','Kajiado',
                'Kilifi','Kitui','Laikipia','Makueni','Mandera',
                'Marsabit','Meru','Narok','Nyeri','Samburu',
                'Taita-Taveta','Tana-River','Tharaka-Nithi',
                'Turkana','Wajir','West-Pokot']

metrics_ls = []
values_ls = []
for h in [4,6,8,10,12]:
    for c in county_grp:
        print(c)
        data_df = pd.read_csv(f'New3M/s2_{c}_data_SM_3M.csv')
        data_df['Date']=pd.to_datetime(data_df.Date)
        # print(data_df.head())
        metrics, values = runAR_Adam(data_df, c, h, 'VCI')
        metrics_ls.append(metrics)
        values_ls.append(values)
pd.concat(values_ls).to_csv(f'ARDL_CV/Values_VCI_ForecastDF_VCI_AR_New.csv', index=False)
pd.concat(metrics_ls).to_csv(f'ARDL_CV/Metrics_VCI_ForecastDF_VCI_AR_New.csv', index=False)
