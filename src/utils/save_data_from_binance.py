# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:09:41 2022

@author: slaven
"""

# binance api exchange data
import time
import requests
import numpy as np
import pandas as pd  # for candlestick libraries

# custom imports
from binance_utils import getTimeInterval, structureDatapoint

# get data from binance api based on tick_interval and trading pair and save it to a csv file
def saveData(market, tick_interval, limit, flag=False):
    print('*** FETCHING DATA FOR {} WITH INTERVAL {} ***'.format(market, tick_interval))
    todayInMilliseconds = int(round(time.time() * 1000))
    timeIntervalInMilliseconds = getTimeInterval(tick_interval)
    maxYearsInMillisecondsSinceBinance = 24*3600*1000 * 365 * 6  # 6 years
    # how often repetitions are executed to get data further in the past
    reps = int(np.ceil(maxYearsInMillisecondsSinceBinance / (timeIntervalInMilliseconds * int(limit))))
    structured = []
    
    for rep in range(reps-1,-1,-1):  # start included, end excluded, -1 reverses sequence
        url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval\
            + '&limit='+limit\
            + "&endTime="+str(todayInMilliseconds - timeIntervalInMilliseconds * int(limit) * rep)
        data = requests.get(url).json()
        print(data)
        
        for entry in data:
            temp = structureDatapoint(entry)
            structured.append(temp)
        print("\tExecuting Epoch {}...".format(rep))
    
    print('*** DATA RETRIEVED! ***')  
    df = pd.DataFrame(data=structured)
    
    # save data
    print('*** NOW SAVING DATA WITH LENGTH {}... ***'.format(len(df)))  
    df.to_csv('../datasets/'+market+'_'+tick_interval+'_binance.csv', index=False)
    print('*** DATA SAVED! ***\n')
    
    if flag:
        return df
  
if __name__ == '__main__':
    market = 'BTCUSDT'
    tick_interval = '1w'
    limit='1000'
    
    # df = saveData(market, tick_interval, limit, flag=True)