# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:59:42 2022

@author: slaven
"""

import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler


# helper function to get file names in a target directory
def getFilenamesInDirectory(contains, targetDirectory):
    # get list with all datasets in directory
    # define paths
    base = os.getcwd()
    dataPath = os.path.join(base, targetDirectory)
    
    # get all filenames in datasets folder
    filenames = []
    entries = os.scandir(dataPath)
    for entry in entries:
        if contains in entry.name:
            filenames.append(entry.name)
    return filenames


# get raw data from binance csv files
def getRawData(startRange, endRange, market):
    targetDirectory = 'datasets'
    filenames = getFilenamesInDirectory(market, targetDirectory)
    filename = [fname for fname in filenames if market in fname][0]
    
    # get dataset
    rawData = pd.read_csv(targetDirectory+'/'+filename)
    
    startIdx = rawData.loc[rawData['date'] == startRange.strftime('%Y-%m-%d %H:%M:%S')].index.tolist()[0]
    endIdx = rawData.loc[rawData['date'] == endRange.strftime('%Y-%m-%d %H:%M:%S')].index.tolist()[0]
    data = rawData.iloc[(startIdx):(endIdx)]
    return data          


# extract the features of interest
def extractFeaturesFromRawData(data):
    return pd.DataFrame(data=[data['close'], data['high'], data['low']]).T

# extract additional features
def extractAdditionalFeaturesFromRawData(data):
    return pd.DataFrame(data=data['transactions'])

# format raw data into the right shape to be used as tensors later on
def formatRawDataForInput(data, window):
    priceData = extractFeaturesFromRawData(data)
    additionalData = extractAdditionalFeaturesFromRawData(data)
    X_tensor = []             # formatted tensor X
    priceRelativeVector = []  # price relative vector
    
    # scale non-price data
    scaler = MinMaxScaler()
    scaler.fit(additionalData)
    
    for i in range(window+20, len(priceData)):
        stepData = []
        for j in range(len(priceData.iloc[i])):
            # normalize with close price
            stepData.append([np.divide(priceData.iloc[k][j], priceData.iloc[i-1]['close']) for k in range(i-window, i)])
 
        X_tensor.append(stepData)
        # EQUATION 1: y_t = elementWiseDivision(v_t, v_t-1)  # without 1 at the beginning
        priceRelativeVector.append(np.divide(priceData.iloc[i-1]['close'], priceData.iloc[i-2]['close']))
    return X_tensor, priceRelativeVector


# wrapper function to return the finalized data to be used in neural networks
def prepareData(startRange, endRange, markets, window):
    # final shape CPU mode: timesteps x markets x lookback x features, features = channels = (close, high, low, ...)
    data = []
    # EQUATION 1: y_t = (v_0,t/v_0,t-1 | v_BTC,t/v_BTC,t-1 | ... | v_ADA,t/v_ADA,t-1), with v_0 = 1 for all t (v_0 is the cash)
    priceRelativeVectors = []
    
    for market in markets:
        rawData = getRawData(startRange, endRange, market)
        formattedData, priceRelativeVector = formatRawDataForInput(rawData, window)
        data.append(formattedData)
        priceRelativeVectors.append(priceRelativeVector)
    
    # get them into the right shape
    data = np.swapaxes(np.swapaxes(data, 2, 3), 0, 1)  # the tensor X_t (EQUATION 18, page 9)
    priceRelativeVectors = np.transpose(priceRelativeVectors)
    return data, priceRelativeVectors


# mainly for debugging
if __name__ == '__main__':
    window = 50
    
    # prepare train data
    startRange = datetime.datetime(2022,6,17,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    market = 'BTCUSDT'
    
    rawData = getRawData(startRange, endRange, market)
    formatted, _ = formatRawDataForInput(rawData, window)
