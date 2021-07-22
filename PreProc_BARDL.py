import glob
import pandas as pd
import numpy as np
import pymc3 as pm

# Function to create direct forcast data structure
def makeDirFcastDf2(df, p_lags0,p_lags1,p_lags2, q_lags, c_lags,date, target):
    new_df = pd.DataFrame()
    col = df.columns
    for h in range(1,17):
        new_df[f'{target}_0']=df[target]
        new_df[f'{target}_{h}']=df[target].shift(periods=-h)

        if q_lags == 0:
            pass
        elif q_lags == 1:
            new_df[f'{target}_lag_0']=df[target]
        elif q_lags >= 2:
            for q in range(1,q_lags):
                new_df[f'{target}_lag_0']=df[target]
                new_df[f'{target}_lag_{q}']=df[target].shift(periods=q)

        if p_lags0 == 0:
            pass
        elif p_lags0 == 1:
            new_df[f'{col[0]}_lag_0']=df[col[0]]
        elif p_lags0 >= 2:
            for p0 in range(1,p_lags0):
                new_df[f'{col[0]}_lag_0']=df[col[0]]
                new_df[f'{col[0]}_lag_{p0}']=df[col[0]].shift(periods=p0)

        if p_lags1 == 0:
            pass
        elif p_lags1 == 1:
            new_df[f'{col[1]}_lag_0']=df[col[1]]
        elif p_lags1 >= 2:
            for p1 in range(1,p_lags1):
                new_df[f'{col[1]}_lag_0']=df[col[1]]
                new_df[f'{col[1]}_lag_{p1}']=df[col[1]].shift(periods=p1)

        if p_lags2 == 0:
            pass
        elif p_lags2 == 1:
            new_df[f'{col[2]}_lag_0']=df[col[2]]
        elif p_lags2 >= 2:
            for p2 in range(1,p_lags2):
                new_df[f'{col[2]}_lag_0']=df[col[2]]
                new_df[f'{col[2]}_lag_{p2}']=df[col[2]].shift(periods=p2)
#
        if c_lags == 1:
            new_df['County_lag_0']=df['County']
        if date == 1:
            new_df['Date_lag_0']=df['Date']

    return new_df

#Function for spliting data in to training and test set DataFrame
def fcast_train_testDF1(df,p_order0,p_order1,p_order2,q_order, target_var, c_lags,date, f_horizon=None):
    newdf_ls = []
    X_train2 = pd.DataFrame()
    cty_ = df.County.unique().tolist()

    for c in cty_:
        df2 = df[df.County == c]
        newdf1 = makeDirFcastDf2(df2,p_order0,p_order1, p_order2, q_order, c_lags, date, target_var)
        useDF1 = newdf1.loc[:,[v for v in newdf1.columns if 'lag' in v]]
        useDF1[f'{target_var}_{f_horizon}'] = newdf1[f'{target_var}_{f_horizon}']

        X_train = useDF1.dropna()
        newdf_ls.append(X_train)

    X_train2 = pd.concat(newdf_ls, axis=0)
    y_train = X_train2.loc[:,f'{target_var}_{f_horizon}']
    return X_train2, y_train, cty_

# Function to scale values
def scaleValues(df, target):
    sdf = df.iloc[:,:-3]
#     print(sdf.columns)
    sdf1 = (sdf-sdf.mean())/sdf.std()
    if target == 'VCI':
        sdf1[target] = df[target]/100
    elif target == 'NDVI':
        sdf1[target] = df[target]
    sdf1['County'] = df.County
    sdf1['Date'] = df.Date
#     print(sdf1.head())
    return sdf1

# Function to Detred target variable prior to fittig
def detrend(df, target, df_type=None):
    df_ls = []
    pooled_means = {'county':[], 'means':[]}
    if df_type=='singles':
        df0 = df
        tgmeans = df0[[target]].mean()
        df0[target] = df0[[target]] - tgmeans
        if target == 'VCI':
            single_mean = tgmeans.values/100
        elif target =='NDVI':
            single_mean = tgmeans.values
    elif df_type=='pooled':
        for c in df.County.unique().tolist():
            df2 = df[df.County == c]
            tgmeans = df2[[target]].mean()
            df2[target] = df2[[target]] - tgmeans
            pooled_means['county'].append(c)
            if target == 'VCI':
                pooled_means['means'].append(tgmeans.values[0]/100)
            elif target =='NDVI':
                pooled_means['means'].append(tgmeans.values[0])
            df_ls.append(df2)
    if df_type=='singles':
        return df0, single_mean
    elif df_type=='pooled':
        return pd.concat(df_ls), pooled_means

# Preprocess data for direct forecasting
def PrepData(tr_df, df_type, lst_p0, precip_p1,soil_p2,targ_q, target, f_horizon, anom=None, growing_ssn=None):
    if anom == True:
        select_vars = ['LST_Anom','Rainfall_Anom', 'SoilMoist_Anom',f'{target}','County','Date']
    elif anom ==False:
        select_vars = ['LST','Rainfall', 'SoilMoist',f'{target}','County','Date']

    pq_order = [lst_p0,precip_p1,soil_p2,targ_q]

    if anom == False:
        tr_df['Rainfall'] = tr_df['Rainfall'].ewm(com=5).mean()

    if growing_ssn == 'MAM':
        tr_df = tr_df.loc[tr_df['Season'].isin(['mam'])]
    elif growing_ssn == 'OND':
        tr_df = tr_df.loc[tr_df['Season'].isin(['ond'])]
    else:
        pass

    tr_df2, target_means = detrend(tr_df, target, df_type)
    scale_df = scaleValues(tr_df2.loc[:,select_vars], target)
    X_trainX, y_trainX, cty_grp = fcast_train_testDF1(scale_df, p_order0=pq_order[0] ,
                                                                    p_order1=pq_order[1], p_order2=pq_order[2],
                                                                    q_order=pq_order[3], c_lags=1,date=1,
                                                                    target_var=target,
                                                                    f_horizon=f_horizon)
#     print(X_trainX.columns)
    return X_trainX, y_trainX, cty_grp, target_means


def NewcatProbs(farr0):
    cats = {'FNo-Drought':[],'FDrought':[]}
    for i in np.arange(farr0.shape[1]):
        cats['FNo-Drought'].append(len(farr0[:,i][(farr0[:,i] > 0.35)])/1000)
        cats['FDrought'].append(len(farr0[:,i][(farr0[:,i] < 0.35)])/1000)
    return pd.DataFrame(cats)

def BARDL_factoryB(X_data=None, y_data=None, sampler=None):
    #Function for infering HBVAR parameters with PyMC3
    coords = {'var':X_data.columns, 'Obs':np.arange(X_data.shape[0])}

    # Model for County Level Grouping
    with pm.Model(coords=coords) as hadl_model_h1:
        # Get Data
        if sampler == 'MCMC':
            x_input = pm.Data('x_input', X_data)
            y_input = pm.Data('y_input', y_data)

        elif sampler == 'JAX':
            x_input = X_data.values
            y_input = y_data

        # prior for average intercept:
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1.0)
        mu_beta = pm.Normal('mu_beta', mu=0, sigma=0.5, dims='var')

        # Model
        mean = mu_alpha + (mu_beta*x_input).sum(axis=1)

        #Model Error
        sigma_z = pm.HalfNormal('sigma_z', 1)

        # Define likelihood
        y_pred = pm.Normal('y_pred', mu=mean, sigma=sigma_z, observed=y_input, testval=1, dims="Obs")

        return hadl_model_h1


# Function for posterior prediction on held out data
def testModelCV(x_test, y_test, county, trace, horizon, test_means,date=None, model=None, sampler=None, model_factory=None):
    lcmeans = test_means
    hor = np.repeat(horizon, len(y_test))
    county = np.repeat(county, len(y_test))
    y_empty = np.empty_like(y_test.values)
    Date = date.values

    if sampler == 'MCMC':
        with model:
            pm.set_data({"x_input":x_test, 'y_input':y_empty})
            pred2_ = pm.sample_posterior_predictive(trace, samples=1000)
            new_pred = pred2_['y_pred']+lcmeans
            probs= NewcatProbs(new_pred)
            print(new_pred.shape)
            print(y_test.values.shape)
            if horizon >=10:
                v = y_test.name[:-3]
            else:
                v = y_test.name[:-2]
            forecastDf = pd.DataFrame({'County':county,
                                'Horizon':hor,
                                'Date':Date,
                                f'{v}_Forecast':new_pred.mean(axis=0),
                                f'{v}_Upper1':np.percentile(new_pred, 97.5, axis=0),
                                f'{v}_Upper0':np.percentile(new_pred, 75, axis=0),
                                f'{v}_Lower1':np.percentile(new_pred, 25, axis=0),
                                f'{v}_Lower0':np.percentile(new_pred, 2.5, axis=0),
                                f'{v}_Observed':y_test.values+lcmeans})

    if sampler == 'JAX':
        with BARDL_factoryB(X_data=x_test, y_data=y_empty, sampler='JAX') as HB_Model:
            pred2_ = pm.sample_posterior_predictive(trace, samples=1000)
            new_pred = pred2_['y_pred']+lcmeans
            probs= NewcatProbs(new_pred)
            print(new_pred.shape)
            print(y_test.values.shape)
            if horizon >=10:
                v = y_test.name[:-3]
            else:
                v = y_test.name[:-2]
            forecastDf = pd.DataFrame({'County':county,
                                'Horizon':hor,
                                'Date':Date,
                                f'{v}_Forecast':new_pred.mean(axis=0),
                                f'{v}_Upper1':np.percentile(new_pred, 97.5, axis=0),
                                f'{v}_Upper0':np.percentile(new_pred, 75, axis=0),
                                f'{v}_Lower1':np.percentile(new_pred, 25, axis=0),
                                f'{v}_Lower0':np.percentile(new_pred, 2.5, axis=0),
                                f'{v}_Observed':y_test.values+lcmeans})

            forecastDf['Obs_No-Drought'] = np.where((forecastDf[f'{v}_Observed'] > 0.35), 1, 0)
            forecastDf['Obs_Drought'] = np.where((forecastDf[f'{v}_Observed'] < 0.35), 1, 0)

            forecastDf[['FNo-Drought','FDrought']] = probs

    return forecastDf, new_pred
