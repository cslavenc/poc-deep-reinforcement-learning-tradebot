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

def getData(startRange, endRange, market):
    targetDirectory = 'datasets'
    filenames = getFilenamesInDirectory(market, targetDirectory)
    filename = [fname for fname in filenames if market in fname][0]
    
    # get dataset
    rawData = pd.read_csv(targetDirectory+'/'+filename)
    
    startIdx = rawData.loc[rawData['date'] == startRange.strftime('%Y-%m-%d %H:%M:%S')].index.tolist()[0]
    endIdx = rawData.loc[rawData['date'] == endRange.strftime('%Y-%m-%d %H:%M:%S')].index.tolist()[0]
    data = rawData.iloc[(startIdx):(endIdx)]
    return data          


# TODO(!) : i think transpose is not necessary here actually
# TODO : does it matter if close, high, low first or should close be at the end?
# TODO : handle the case where addBTC or addUSDT is true
def formatData(data, addCurrency=False):
    # transpose to get timestamps as rows, features as cols
    data = pd.DataFrame(data=[data['close'], data['high'], data['low']]).T
    return data


# TODO : understand variables properly, add a reference where they appear in the original paper
def formatDataForInput(data, window):
    data = formatData(data)
    x = []  # final formatted tensor X
    priceRelativeVector = []  # price relative vector
    rates = []  # TODO : what is this?
    for i in range(window, len(data)):
        stepData = []
        for j in range(len(data.iloc[i])):
            stepData.append([np.divide(data.iloc[k][j], data.iloc[i-1]['close']) for k in range(i-window, i)])
        x.append(stepData)
        priceRelativeVector.append(np.divide(data.iloc[i-1]['close'], data.iloc[i-2]['close']))
        rates.append(data.iloc[i]['close'])
    return x, priceRelativeVector, rates