# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:24:34 2022

@author: slaven
"""

# create a list with trading pairs in binance
import requests
import numpy as np
import pandas as pd

def getTimeInterval(tick_interval):
    baseInMilliseconds = 24*3600*1000  # 1 day in ms
    if tick_interval == '1d':
        return baseInMilliseconds
    elif tick_interval == '1w':
        return baseInMilliseconds * 7
    elif tick_interval == '3d':
        return baseInMilliseconds * 3
    elif tick_interval == '4h':
        return baseInMilliseconds // 6
    elif tick_interval == '1h':
        return baseInMilliseconds // 24
    elif tick_interval == '15m':
        return baseInMilliseconds // (24*4)
    elif tick_interval == '5m':
        return baseInMilliseconds // (24*12) 
    elif tick_interval == '3m':
        return baseInMilliseconds // (24*20)
    elif tick_interval == '1m':
        return baseInMilliseconds // (24*60)
    else:
        raise "Illegal Tick Interval! Your tick interval is: " + tick_interval

# PRE : data entry from binance klines api
# POST: prepares data to be saved later as csv
def structureDatapoint(entry):
    return {
        "date": pd.to_datetime(entry[0], unit="ms"),
        "open": float(entry[1]),
        "high": float(entry[2]),
        "low": float(entry[3]),
        "close": float(entry[4]),
        "volume": float(entry[5]),
        "close_time": entry[6],
        "transactions": float(entry[7]),
        "number_of_trades": entry[8],
        "taker_buy_base": entry[9],
        "taker_buy_quote": entry[10],
        "ignore": entry[11]
        }

def structureData(data):
    structured = []
    for entry in data:
        entry = structureDatapoint(entry)
        structured.append(entry)
    return structured

def logDatapoint(dp):
    for prop in ["open", "high", "low", "close"]:
        dp[prop] = np.log(dp[prop])
    return dp

def logData(data):
    logData = []
    for datapoint in data:
        logData.append(logDatapoint(datapoint))
    return logData


# PRE : currency - USDT, BTC etc. (valued against)
# POST: returns a list of all tradingPairs
def getTradingPairs(currency="USDT"):
    currency = currency
    tradingPairs = []
    exchangeDataUrl = "https://api.binance.com/api/v3/exchangeInfo" 
    exchangeInfoData = requests.get(exchangeDataUrl).json()  # also contains all other exchange data
    
    # get all symbols of trading pairs
    tradingPairsData = exchangeInfoData["symbols"]
    
    for tp in tradingPairsData:
        pair = tp["symbol"]
        status = tp["status"]
        
        if currency in pair and status != "BREAK" and "DOWNUSDT" not in pair\
            and "UPUSDT" not in pair and "TUSDT" not in pair and "GBPUSDT" not in pair\
            and "USDC" not in pair  and "USDP" not in pair and "EURUSDT" not in pair\
            and "AUDUSDT" not in pair and "PAXG" not in pair and "DAI" not in pair\
            and "BIDR" not in pair and "BRL" not in pair and "IDRT" not in pair\
            and "RUB" not in pair and "USDTTRY" not in pair and "USDTUAH" not in pair\
            and "LUNAUSDT" not in pair and "LUNCUSDT" not in pair and "BUSD" not in pair\
            and "UST" not in pair and "TUSD" not in pair and "NGNUSDT" not in pair:
            tradingPairs.append(pair)
    return tradingPairs


if __name__ == '__main__':
    tradingPairs = getTradingPairs()
