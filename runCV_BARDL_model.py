import pandas as pd
import numpy as np
import PreProc_BARDL
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import warnings

# Function to train and predict via time series rolling cross validation
def trainForcast(path, fh, trainDf=None, target=None,county=None, anom=None):

    forecast_df_ls = []
    pred_arr = []

    X_train, y_train, cty_gp, means = PreProc_BARDL.PrepData(trainDf, df_type='singles', lst_p0=6,precip_p1=6,soil_p2=6,targ_q=6, target=target, f_horizon=fh, anom=anom)
    X_sub = X_train
    print(f'{county}-{target}--{y_train.name}-at {fh} week ahead')

    X_data = X_sub.drop(['County_Code_lag_0', y_train.name], axis=1)
    y_data = X_sub[y_train.name]
    # print(X_data.columns)
    print(X_data.shape)
    print(means)

    strt = np.arange(0, len(X_data), 200)
    print(strt)

    for t, h in enumerate(strt):
        sub_xdata = X_data[h:h+500]
        sub_ydata = y_data[h:h+500]
        # sub_cty_idx = cty_idx[h:h+500]
        print(f'{county}: Step {t+1} horizons {fh} for weeks ahead.....')
        print(len(sub_xdata[-100:]))

        #Call and run the model for parameter inferrence
        with PreProc_BARDL.BARDL_factoryB(X_data=sub_xdata[:400], y_data=sub_ydata[:400], sampler='MCMC') as HB_Model:
            trace_h = pm.sample(2000, tune=2000, target_accept=0.95)
            # trace_h = pm.sampling_jax.sample_numpyro_nuts(2000, tune=2000, target_accept=0.95)

            # Save model parameters HMC sampling trace infomation
            az.summary(trace_h).to_csv(path+f'.../{county}_traceSummary_{y_train.name}_Set{h}_H{fh}_Anomaly.csv')
            _ = az.plot_trace(trace_h, compact=True)
            plt.savefig(path+f'.../{county}tracePlot_{y_train.name}_H{fh}_Set{h}_H{fh}_Anomaly.png')


            forcasts, new_pred = PreProc_BARDL.testModelCV(x_test=sub_xdata[-100:], y_test=sub_ydata[-100:], county=county,
                                                    model=HB_Model, trace=trace_h, horizon=fh,
                                                    test_means=means, sampler='MCMC')
        forecast_df_ls.append(forcasts)
        pred_arr.append(new_pred)

    return pd.concat(forecast_df_ls), np.concatenate((pred_arr), axis=1)
    return means

# list of counties
county_grp = ['Embu','Baringo','Garissa','Isiolo','Kajiado',
                'Kilifi','Kitui','Laikipia','Makueni','Mandera',
                'Marsabit','Meru','Narok','Nyeri','Samburu',
                'Taita-Taveta','Tana-River','Tharaka-Nithi',
                'Turkana','Wajir','West-Pokot']


#Run Cross validation Function and save
path = '..path/to/data'
for i, h in enumerate([4,6,8,10,12]):
    forecast_ls = []
    pred_arr = []

    for c in county_grp:
        trainData = pd.read_csv(path+f'DataSets/Anom_DataPool/Weekly_{c}_SIM_Smooth_Anom_3M.csv')
        out_forecast = trainForcast(path, fh=h, trainDf=trainData, target='VCI', county=c, anom=True)
        forecast_ls.append(out_forecast[0])
        pred_arr.append(out_forecast[1])
    #
    pd.concat(forecast_ls).to_csv(path+f'.../VCI_ForecastDF_H{h}_CV_HMC_Anomaly.csv', index=False)
    np.save(path+f'.../VCI_ForecastArr_H{h}_CV_HMC_Anomaly.npy', np.concatenate((pred_arr), axis=1))
