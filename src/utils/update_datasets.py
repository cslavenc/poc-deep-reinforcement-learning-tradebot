# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:04:06 2022

@author: slaven
"""

# os, pathlib used to set the current working directory in a general way
import os
import pathlib

import time
import datetime
import requests
import pandas as pd

from utils import getFilenamesInDirectory
from binance_utils import structureData


def timestamp(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    diff = dt - epoch
    return diff.total_seconds()*1000


# helper function to update a market
def updateMarket(market, filename):
    rawData = pd.read_csv('datasets/'+filename)
    
    lastDatetime = datetime.datetime.strptime(rawData['date'].iloc[-1],
                                              '%Y-%m-%d %H:%M:%S')
    lastInMilliseconds = int(timestamp(lastDatetime))
    todayInMilliseconds = int(round(time.time() * 1000))
    
    url = 'https://api.binance.com/api/v3/klines?symbol='+market\
        + '&interval='+timeframe\
        + '&limit='+limit\
        + '&startTime='+str(lastInMilliseconds)\
        + "&endTime="+str(todayInMilliseconds)
    data = structureData(requests.get(url).json())
    
    # append data
    # remove last entry from rawData to get the last updated entry as well
    rawData = pd.concat([rawData[:-1], pd.DataFrame(data=data)], ignore_index=True)
    rawData.to_csv('datasets/'+filename, index=False)
    
    return rawData
    

if __name__ == '__main__':
    os.chdir(pathlib.Path(__file__).parent.parent)
    timeframes = ['15m']
    # simply use max, binance will return the proper number of values based on startTime and endTime
    limit = '1000'
    
    for _ in range(1):
        for timeframe in timeframes:
            print('Begin timeframe: ' + timeframe)
            filenames = getFilenamesInDirectory(timeframe, 'datasets')
            
            for filename in filenames:
                print('Updating for {}'.format(filename))
                market = filename.split('_')[0]
                rawData = pd.read_csv('datasets/'+filename)
                rawData = updateMarket(market, filename)