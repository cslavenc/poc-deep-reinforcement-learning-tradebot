# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:59:42 2022

@author: slaven
"""

import os
import numpy as np
import pandas as pd


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


# get raw data from binance
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
    data = pd.DataFrame(data=[data['close'], data['high'], data['low']]).T
    return data


# format raw data into the right shape to be used as tensors later on
def formatRawDataForInput(data, window):
    data = extractFeaturesFromRawData(data)
    X_tensor = []             # final formatted tensor X
    priceRelativeVector = []  # price relative vector
    rates = []                # close price of each asset: [[close_BTC_0, close_ETH_0, ..., close_ADA_0], ..., [close_BTC_t, ..., close_ADA_t]]
    
    for i in range(window, len(data)):
        stepData = []
        for j in range(len(data.iloc[i])):
            # normalize with close price
            stepData.append([np.divide(data.iloc[k][j], data.iloc[i-1]['close']) for k in range(i-window, i)])
        X_tensor.append(stepData)
        # EQUATION 1: y_t = elementWiseDivision(v_t, v_t-1)  # without 1 at the beginning
        priceRelativeVector.append(np.divide(data.iloc[i-1]['close'], data.iloc[i-2]['close']))
        rates.append(data.iloc[i]['close'])
    return X_tensor, priceRelativeVector, rates


# wrapper function to return the finalized data to be used in neural networks
def prepareData():
    pass