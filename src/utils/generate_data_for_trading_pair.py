# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:22:33 2022

@author: slaven
"""

from save_data_from_binance import saveData

# generate datasets for every trading pair based on provided currency from binance
if __name__ == '__main__':
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'MATICUSDT']
    timeframes = ['15m']  # also supported: '1w', '3d', '1d', '4h', '1h' etc.
    limit='1000'
    
    for market in markets:
        for timeframe in timeframes:
            saveData(market, timeframe, limit)