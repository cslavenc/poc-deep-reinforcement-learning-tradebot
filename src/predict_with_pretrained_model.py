# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:22:06 2023

@author: slaven.cvijetic
"""

import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


from pipeline import Portfolio, prepareData, findClosestQuarterlyMinute,\
    sanitizeCashValues

# only for simple predictions without any downside analysis
# if a tradestop was expected, run pipeline.py with the appropriate start and end dates
if __name__ == '__main__':
    # force CPU usage
    K.set_image_data_format('channels_last')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # required to fully enforce CPU usage
    
    window = 50
    learning_rate = 0.00019
    currentActualBalanceInUSDT = 10000
    sinceLastPrediction = 10  # can be anything, such as days, minutes, hours etc.
    
    now = datetime.datetime.utcnow()
    endRange = datetime.datetime(now.year, now.month, now.day, now.hour, 
                                 findClosestQuarterlyMinute(now.minute), 0)
    startRange = endRange - datetime.timedelta(days=sinceLastPrediction)
    
    markets = ['BUSDUSDT_15m', 'BTCUSDT_15m', 'ETHUSDT_15m', 'BNBUSDT_15m',
               'ADAUSDT_15m', 'MATICUSDT_15m']
    
    # datasets MUST be up-to-date!
    data, priceRelativeVectors = prepareData(startRange, endRange, markets, window)
    
    # create a new model
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data, priceRelativeVectors)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            run_eagerly=True,
                            loss=portfolio.model.cumulatedReturn,
                            metrics='accuracy')

    # load pretrained model
    pretrainedModel = tf.keras.models.load_model('models/MODEL_NAME', 
                                                  compile=False)
    portfolio.model.set_weights(pretrainedModel.get_weights())
    
    # predict
    priceRelativeVectors = sanitizeCashValues(priceRelativeVectors)
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    currentPortfolioWeights = portfolio.model.predict([data[-1:], 
                                                       optimalWeights[-1:]])
    currentPortfolioWeights = np.asarray(currentPortfolioWeights)
    
    # calculate current portfolio value
    portfolioWeights = np.loadtxt('weights.txt')
    portfolioValues = np.loadtxt('values.txt')
    
    # TODO : for loop to update add all the new values (do it efficiently somehow)
    value = portfolio.calculateCurrentPortfolioValue(
                portfolioValues[-1],
                priceRelativeVectors[-1:],
                portfolioWeights[-1]
            )
    
    # update portfolio weights and values
    portfolioValues = np.append(portfolioValues, value)
    portfolioWeights = np.append(portfolioWeights, currentPortfolioWeights, axis=0)
    
    # scale all data points to be inline with the actual balance
    scale = portfolioValues[-1] / currentActualBalanceInUSDT
    portfolioValues = portfolioValues / scale
    
    # save portfolio weights and values
    # DONT FORGET TO FIX values.txt
    np.savetxt('weights.txt', np.round(portfolioWeights, 3), fmt='%.3f')
    np.savetxt('values.txt', portfolioValues, fmt='%.5f')
