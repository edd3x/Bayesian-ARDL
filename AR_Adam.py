#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:10:57 2019

@author: abb22
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob



def astro_regress_one(Y,X,nlags):

    nobs=len(Y)

    Xsegs=[]
    Ysegs=[]
    segstart=0
    nsegs=0

    for t in range(nobs-1):
        if not np.isnan(X[t]) and not np.isnan(Y[t]):
            if np.isnan(X[t+1]) or np.isnan(Y[t+1]):
                if t+1-segstart>nlags:
                    Xsegs.append(X[segstart:t+1])
                    Ysegs.append(Y[segstart:t+1])
                    nsegs=nsegs+1
        if np.isnan(X[t]) or np.isnan(Y[t]):
            if not np.isnan(X[t+1]) and not np.isnan(Y[t+1]):
                segstart=t+1

    if not np.isnan(X[nobs-1]) and not np.isnan(Y[nobs-1]):
        if nobs-segstart>nlags:
            Xsegs.append(X[segstart:nobs])
            Ysegs.append(Y[segstart:nobs])
            nsegs=nsegs+1

    nobs=0
    for i in range(nsegs):
        nobs=nobs+len(Xsegs[i])

    regressors = np.zeros((nobs-nsegs*nlags,nlags))
    ydep=np.zeros(nobs-nsegs*nlags)

    segstart=0

    for i in range(nsegs):
        XX=Xsegs[i]
        YY=Ysegs[i]
        nobsseg=len(XX)
        ydep[segstart:segstart+nobsseg-nlags] = YY[nlags:]
        for tau in range(nlags):
            regressors[segstart:segstart+nobsseg-nlags,tau] = XX[nlags-tau-1:nobsseg-tau-1]
        segstart=segstart+nobsseg-nlags

    beta=np.zeros(nlags)
    ypred=np.zeros(nobs-nsegs*nlags)
    u=np.zeros(nobs-nsegs*nlags)

    regrees = np.linalg.lstsq(regressors,ydep)
    beta=regrees[0]
    ypred = np.dot(regressors,beta)  # keep hold of predicted values
    u = ydep-ypred
    res=np.cov(u)

    return beta, u, res, ypred

def astro_predict_one(Y,X,nlags,trainlength):

    nobs=len(Y)
    ntests=nobs-trainlength

    ypred=np.zeros(ntests)
    u=np.zeros(ntests)

    nopredict=0

    for k in range(ntests):
        ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
        beta=ret[0]

        predictors = np.zeros(nlags)
        for tau in range(nlags):
            predictors[tau] = X[k+trainlength-tau-1]

        ypred[k]=np.dot(predictors,beta)
        u[k]=Y[k+trainlength]-ypred[k]
        if np.isnan(u[k]):
            nopredict=nopredict+1

    respredict=np.sqrt(np.nanvar(u))

    k=ntests
    ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
    beta=ret[0]

    predictors = np.zeros(nlags)
    for tau in range(nlags):
        predictors[tau] = X[k+trainlength-tau-1]

    forecast=np.dot(predictors,beta)

    return respredict, ypred, nopredict, forecast

def forecast(VCI):
    VCImean=np.nanmean(VCI)
    VCIz=VCI-VCImean
    nlags0=3
    trainlength=200
    l=len(VCI)

    VCIpred=np.zeros((9,l))
    Forecast=np.zeros(9)
    Sigma=np.zeros(9)
    Forecast[0]=VCI[l-1]

    for i in range(0,8):

        Y=VCIz[i:]
        X=VCIz[0:l-i]
        ret=astro_predict_one(Y,X,nlags0,trainlength)
        ypred=ret[1]
        VCIpred[i,trainlength+i:]=ypred
        VCIpred[i,:]=VCIpred[i,:]+VCImean
        Forecast[i+1]=ret[3]+VCImean
        Sigma[i+1]=ret[0]

    return Forecast, Sigma



#Function for creating direct forecast DataFrame
def makeDirFcastDfAR(df, p_lags0,p_lags1,p_lags2, q_lags, s_lags, date, target):
    new_df = pd.DataFrame()
    col = df.columns
    # print(col)
    for h in range(1,13):
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

        if s_lags == 1:
            new_df[f'{col[-2]}_lag_0']=df[col[-2]]
        elif s_lags >= 2:
            for s in range(1,s_lags):
                new_df[f'{col[-2]}_lag_0']=df[col[-2]]
                new_df[f'{col[-2]}_lag_{s}']=df[col[-2]].shift(periods=s)

        if date == 1:
            new_df['Date_lag_0']=df['Date']

    return new_df.dropna()


#Function for spliting data in to training and test set DataFrame

def fcast_train_testDF_AR(df, p_order0,p_order1,p_order2,q_order, target_var, s_lags, date, f_horizon=None):
    newdf1 = makeDirFcastDfAR(df,p_order0,p_order1,p_order2,q_order,s_lags,date, target_var)
    # print(newdf1.columns)
    useDF1 = newdf1.loc[:,[l for l in newdf1.columns if 'lag' in l]]
    useDF1[f'{target_var}_{f_horizon}'] = newdf1[f'{target_var}_{f_horizon}']

    train_Df = useDF1.iloc[:,:]
    X_train = train_Df.iloc[:,:]
    y_train = train_Df.loc[:,f'{target_var}_{f_horizon}']

    return X_train, y_train


def scaleValuesV(df):
    sdf = df.iloc[:,:-3]
    sdf1 = (sdf-sdf.mean())/sdf.std()
    sdf1['VCI'] = (df.VCI/100 - np.mean(df.VCI.values/100))
    sdf1['Season'] = df.Season
    sdf1['Date'] = df.Date

    return sdf1, np.mean(df.VCI.values/100)

def scaleValuesA(df):
    sdf = df.iloc[:,:-2]
    sdf1 = (sdf-sdf.mean())/sdf.std()
    sdf1['NDVI_Anom'] = df.NDVI_Anom
    sdf1['Season'] = df.Season

    return sdf1

def R2_adj(actual, predicted, n_params):
    n = len(actual)
    r2 = r2_score(actual, predicted)

    ss_r = (1-r2)*(n-1)
    ss_t = n-n_params-1

    return 1-(ss_r/ss_t), r2

def MAPE(actual, predicted):
    abs_err = np.mean(np.abs((actual - predicted) / actual))

    return 100 * abs_err

def calcIC(obs, preds, n_params):
    N = len(obs)
    mn_ssr = np.mean((obs-preds)**2)
    aic = np.log(mn_ssr)+2*(n_params+1)/N
    bic = np.log(mn_ssr)+(n_params+1)*np.log(N)/N
    adj_r2 = R2_adj(obs, preds, n_params)
    mae = mean_absolute_error(obs, preds)
    mape = MAPE(obs, preds)
    rmse = np.sqrt(mean_squared_error(obs, preds))

    return aic, bic, adj_r2[1], mae, mape, rmse
def addSeason(_df):
    df2 = _df
    df2 = df2.set_index('Date')

    df2['Month'] = df2.index.month

    jf = df2[(df2.Month>=1) & (df2.Month<=2)]
    jf['Season']  = 'jf'
    mam = df2[(df2.Month>=3) & (df2.Month<=5)]
    mam['Season']  = 'mam'
    jja = df2[(df2.Month>=6) & (df2.Month<=9)]
    jja['Season']  = 'jja'
    ond = df2[(df2.Month>=10) & (df2.Month<=12)]
    ond['Season']  = 'ond'
    new_sdf = pd.concat([jf, mam, jja, ond])
    new_sdf = new_sdf.sort_values(by='Date')
    return new_sdf.reset_index(drop=False)
