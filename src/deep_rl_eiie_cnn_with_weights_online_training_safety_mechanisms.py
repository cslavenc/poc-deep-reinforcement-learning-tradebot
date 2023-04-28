#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:35:36 2022

@author: slaven
"""

import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Flatten, Lambda, Concatenate, Softmax

from utils.utils import prepareData, analyzeCurrentDownside
from utils.plot_utils import plotPortfolioValueChange

def expandDimensions(weights):
    # CPU mode None x 4 x 1 x 1: add a new axis at the end of the tensor
    expandedWeights = tf.expand_dims(weights, axis=-1)
    expandedWeights = tf.expand_dims(expandedWeights, axis=-1)
    
    return expandedWeights

class Portfolio():
    """
    EQUATION 2: p_t = p_t-1 * <y_t, w_t-1>
    
    :param prevPortfolioValue, p_t-1
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: p_t, current portfolio value (current net worth)
    """
    def calculateCurrentPortfolioValue(self, prevPortfolioValue, currentPriceRelativeVector, prevPortfolioWeights):
        return prevPortfolioValue * (currentPriceRelativeVector @ prevPortfolioWeights)  # p_t = p_t-1 * <y_t, w_t-1>
    
    
    """
    Simplified implementation with weights of FIGURE 2, page 14
    
    :param X_tensor, input data (CPU mode: timesteps x markets x lookback x features), features = close, high, low, etc.
    :param weights, input weights to obtain the shapes for the weights input layer
    
    return: an EIIE CNN model with weights concatenated
    
    NOTE: if using GPU mode, shapes and kernel sizes need to be adapted
    NOTE: the weights are also known as the **portfolio vector memory** or portfolioVectorMemory in the paper
    NOTE: The portfolioVectorMemory excludes the cash weights. This implementation uses the cash weights anyway
          if it is present in the input data (often simulated as BUSDUSDT)
    """
    def createEiieCnnWithWeights(self, X_tensor, weights):
        mainInputShape = np.shape(X_tensor)[1:]
        weightsInputShape = np.shape(weights)[1:]
        
        # prefer functional API for its flexibility for future model extensions
        mainInputLayer = Input(shape=mainInputShape, name='main_input_layer')
        main = Conv2D(filters=2, kernel_size=(1, 3),
                      activation='relu', name='first_conv_layer')(mainInputLayer)
        main = Conv2D(filters=20, kernel_size=(1, 48),
                      activation='relu', name='second_conv_layer')(main)
        
        # create layers for input weights
        weightsInputLayer = Input(shape=weightsInputShape, name='weights_input_layer')
        weightsExpanded = Lambda(expandDimensions, name='weights_expansion_layer')(weightsInputLayer)
        
        # Concatenate the weightsLayer to the mainLayer
        main = Concatenate(axis=3,
                           name='weights_concatenation_layer')([main, weightsExpanded])
        
        main = Conv2D(filters=1, kernel_size=(1, 1), name='final_conv_layer')(main)
        outputLogits = Flatten()(main)  # bring it into the right shape
        outputWeights = Softmax()(outputLogits)
        
        eiieCnnWithWeightsModel = CustomModel(inputs=[mainInputLayer, weightsInputLayer],
                                              outputs=outputWeights,
                                              name='eiie_cnn_with_weights')
        self.model = eiieCnnWithWeightsModel
    
    
    
    """
    Generate the optimal weights, which is allocating everything into the asset that grew the most.
    Increase is usually >1. This is used as a simulated "y_true" for the cross entropy loss function.
    
    :param priceRelativeVectors, the highest increase/lowest decrease gets weight 1., others weight 0.
    
    return: optimized weights, where 1 for the asset with the highest increase and else 0
    
    NOTE: if cash is added, it will only get weight 1., if all other assets were decreasing (growth < 1)
    """
    def generateOptimalWeights(self, priceRelativeVectors):
        optimalWeights = []
        optimalAssetIds = np.argmax(priceRelativeVectors, axis=1)
        
        for i in range(np.shape(priceRelativeVectors)[0]):
            tempWeights = [0. for _ in range(np.shape(priceRelativeVectors)[1])]
            tempWeights[optimalAssetIds[i]] = 1.
            optimalWeights.append(tempWeights)
        
        return np.asarray(optimalWeights)


"""
This custom model implements the basis necessary for the RL enviromnent.
It uses the intermediate portfolio weights for a minibatch from the previous timestep
as input in the current timestep.

NOTE: ensuring that the custom train loop in train() is correct, it has been
      compared to a custom implementation of train_step() (which is used in fit()).
      train_step() in turn has been sanity checked on the basic fit() function.
NOTE: keep in mind that the portfolio vector memory for a minibatch has shape 
      (minibatchSize, marketsSize), e.g. (32, 6).
      Thus we get many pairs of portfolio weights.
"""
class CustomModel(tf.keras.Model):
    portfolioVectorMemory = []
    
    """
    CustomModel uses a custom loss function. By default, it will not be saved. 
    Thus, the custom loss function is explicitly saved into the base_config 
    which is available after loading the model.
    Alternatively, compile=False may be set to avoid auto-compilation. Then, 
    a loss function can be passed and the loaded model has to be compiled.
    """
    # def get_config(self):
    #     base_config = super(CustomModel, self).get_config()
    #     base_config['cumulatedReturn'] = self.cumulatedReturn
    #     return base_config
    
    """
    EQUATION 22: R = 1/t_f * sum(r_t, start=1, end=t_f+1)
    This cumulated reward function is used for optimization for gradient *ASCENT*
    This function is also used as "loss" when optimizing the neural network weights
    during learning. A normal loss function is usually minimized.
    This reward function is maximized, which can be simply achieved by inverting the sign.
    
    
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: R, average logarithmic cumulated reward (negative value for gradient ASCENT)
    
    NOTE: In preliminary experiments, simply taking the sum of the individual returns
          lead to better learning actually. Dividing by the length (1/t_f)
          resulted into a slightly worse performance in some scenarios.
          This can be somewhat specific based on the training samples and assets used
          and could also be affected by the number of epochs - more epochs could
          potentially better results again when averaging the sum of individual rewards.
    """
    def cumulatedReturn(self, currentPriceRelativeVector, prevPortfolioWeights):
        rewardPerMinibatch = []
        
        for j in range(currentPriceRelativeVector.shape[0]):
            portfolioValuePerMinibatch = tf.multiply(currentPriceRelativeVector[j], prevPortfolioWeights[j])
            individualRateOfReturn = tf.reduce_sum(portfolioValuePerMinibatch, axis=0)  # -1
            individualReward = -tf.math.log(tf.reduce_sum(portfolioValuePerMinibatch, axis=0))
            if individualRateOfReturn < 1:
                individualReward = tf.multiply(individualReward, individualRateOfReturn)
            rewardPerMinibatch.append(individualReward)
        averageCumulatedReturn = tf.math.reduce_sum(rewardPerMinibatch)
        return averageCumulatedReturn


    """
    Implementation of the custom training loop from scratch.
    This is necessary, because the intermediate weights are used as 
    additional inputs in the EIIE CNN.
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example    
    
    :param data, the full training data
    :param weights, weights from previous timestep are required as portfolio vector memory
    :param minibatchSize
    :param priceRelativeVectors, required to calculate reward for gradient ascent
    :param epochs, epochs to iterate over for the most outer for-loop
    """  
    def train(self, data, weights, priceRelativeVectors, minibatchSize, epochs):
        # convert to tensor so it can be used by GradientTape properly
        priceRelativeVectors = tf.convert_to_tensor(priceRelativeVectors, dtype=tf.float32)
        
        # prepare for minibatch evaluation
        originalMinibatchSize = minibatchSize
        dataSize = data.shape[0]  # size of the time series
        
        # if numOfMiniBatches = 13.4, it becomes 14, but the 14th minibatch has a smaller minibatch size
        numOfMiniBatches = int(np.ceil(dataSize/minibatchSize))
        
        for epoch in range(epochs):
            print("\nSTART OF EPOCH {}/{}".format(epoch, epochs-1))
            minibatchSize = originalMinibatchSize  # reset
            
            # reset and use optimal weights as default values
            self.portfolioVectorMemory.append(tf.convert_to_tensor(weights[0:minibatchSize], dtype=tf.float32))
            lossTracker = []
            
            for i in range(1, numOfMiniBatches-1):
                # check if minibatch size is not too big and make it smaller if it does not fit the dataset
                if (i+1)*minibatchSize >= dataSize:
                    minibatchSize = (i+1)*minibatchSize - dataSize - 1
                
                with tf.GradientTape() as tape:
                    predictedPortfolioWeights = self([data[(i*minibatchSize):((i+1)*minibatchSize)],
                                                      # w_t-1, weights from previous minibatch
                                                      self.portfolioVectorMemory[i-1][0:minibatchSize]],
                                                     training=True)

                    loss = self.compiled_loss(priceRelativeVectors[((i+1)*minibatchSize):((i+2)*minibatchSize)],
                                              predictedPortfolioWeights[0:minibatchSize],
                                              regularization_losses=self.losses)
                
                
                # FIGURE 3a: this adds the portfolio vector memory of the minibatch (shape: (minibatchSize, marketsSize))
                self.portfolioVectorMemory.append(predictedPortfolioWeights)
                
                lossTracker.append(loss)
                
                # compute the gradient now
                gradients = tape.gradient(loss, self.trainable_weights)
                
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                
                # Update metrics (includes the metric that tracks the loss)
                self.compiled_metrics.update_state(tf.convert_to_tensor(weights[(i*minibatchSize):((i+1)*minibatchSize)]),
                                                   tf.convert_to_tensor(predictedPortfolioWeights))
                # reset for the next minibatch
                self.rewardPerEpisode = []
                
            print('Current loss: {}'.format(np.mean(lossTracker)))
            self.compiled_metrics.reset_state()
            self.portfolioVectorMemory = []  # reset for the next epoch


"""
Sometimes, the cash is slightly below 1, since the current datasets actually looks at
BUSDUSDT which itself fluctuates a little bit. This should not be the case actually.
Also, a very small bias has been added to values = 1., because sometimes, a crypto asset
remains constant during a timestep and thus has its price relative value is 1. too and
the neural network could potentially prefer this one over cash.
In itself, this does not need to be a bad thing though.

:param priceRelativeVectors, first entry should be the cash (if applicable)

returns: priceRelativeVectors, with the first entry slightly modified
"""
def sanitizeCashValues(priceRelativeVectors):
    smallCashBias = 0.000001
    for i in range(priceRelativeVectors.shape[0]):
        if priceRelativeVectors[i, 0] <= 1.:
            priceRelativeVectors[i, 0] = 1. + smallCashBias
    
    return priceRelativeVectors


"""
Convenience function.
data and testData must have compatible shapes: axis=0 may have different shapes,
other axis must be the same shape

:param data, the train data to be updated for online training
:param testData, test data that was used for predicting in the current step and to be added at the end of train data

returns: updated train data
"""
def updateOnlineTrainData(data, testData):
    data = data[testData.shape[0]:]           # remove oldest datapoints
    data = np.append(data, testData, axis=0)  # add newest datapoints
    return data


if __name__ == '__main__':
    # enforce CPU mode
    K.set_image_data_format('channels_last')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # required to fully enforce CPU usage
    
    # define a few neural network specific variables
    epochs = 1
    window = 50
    minibatchSize = 32
    learning_rate = 0.00019
    
    # prepare train data
    startRange = datetime.datetime(2022,6,1,0,0,0)
    endRange = startRange + datetime.timedelta(weeks=3)
    markets = ['BUSDUSDT_15m', 'BTCUSDT_15m', 'ETHUSDT_15m', 'BNBUSDT_15m',
               'ADAUSDT_15m', 'MATICUSDT_15m']
    
    data, priceRelativeVectors = prepareData(startRange, endRange, markets, window)
        
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data, priceRelativeVectors)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            run_eagerly=True,
                            loss=portfolio.model.cumulatedReturn,
                            metrics='accuracy')
    
    # simulate y_true
    priceRelativeVectors = sanitizeCashValues(priceRelativeVectors)
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    portfolio.model.train(data, optimalWeights, priceRelativeVectors, minibatchSize, epochs)
    
    # save model weights and topology
    portfolio.model.save("models/saved_model_test", include_optimizer=True)
    
    # get predicted portfolio weights and perform online training
    portfolioWeights = []
    onlineTrainData = data
    onlinePriceRelativeVectors = priceRelativeVectors
    onlineOptimalWeights = optimalWeights
    onlineEpochs = 10
    
    weeksIncrement = 6
    startRangeTest = endRange
    endRangeTest = datetime.datetime(2022,8,22,0,0,0)
    currentLowerRangeTest = startRangeTest
    currentUpperRangeTest = startRangeTest + datetime.timedelta(weeks=weeksIncrement)
    
    # define variables for custom safety mechanism
    portfolioValue = [10000.]
    allCashWeights = np.array([1.] + [0. for _ in range(len(markets)-1)])
    tradestopCounter = 0
    shiftTradestopIdx = 0  # needed to shift idx in tradestopSignals
    longSMA = 2500
    shortSMA = 100
    fifteenMinsInOneDay = 4*24  # tradestop duration
    lookbackDownside = 200
    cutoffDrop = -0.08
    
    # always use N weeks test data for predictions to reduce computation time
    # during long periods when testing
    # it likely improves normalization too, as it is normalized on these test batches
    # instead of the entire dataset which includes data from bull and bear markets potentially
    while currentLowerRangeTest < endRangeTest:
        print('\nCurrent time interval: {} to {}'.format(
            currentLowerRangeTest.strftime('%Y-%m-%d'),
            currentUpperRangeTest.strftime('%Y-%m-%d')))
        
        # prepare test data
        testData, testPriceRelativeVectors = prepareData(currentLowerRangeTest, currentUpperRangeTest, markets, window)
        testPriceRelativeVectors = sanitizeCashValues(testPriceRelativeVectors)
        optimalTestWeights = portfolio.generateOptimalWeights(testPriceRelativeVectors)
        
        for i in range(int(np.ceil(testData.shape[0]/(window*5)))):
            print('\nTRAIN STEP: {}/{}'.format(i, int(np.ceil(testData.shape[0]/(window*5))-1)))
            
            # predict on smaller batch
            currentTestData = testData[(i*window*5):((i+1)*window*5)]
            currentOptimalTestWeights = optimalTestWeights[(i*window*5):((i+1)*window*5)]
            currentTestPriceRelativeVectors = testPriceRelativeVectors[(i*window*5):((i+1)*window*5)]
            currentPortfolioWeights = portfolio.model.predict([currentTestData, 
                                                               currentOptimalTestWeights])
            
            # tradestop should not continue past the very last datapoint (only relevant at the end)
            if tradestopCounter > currentPortfolioWeights.shape[0]:
                tradestopCounter = currentPortfolioWeights.shape[0]
            
            # check if there is an active tradestopCounter from the last time range
            ii = 0
            while tradestopCounter > 0:
                currentPortfolioWeights[ii] = allCashWeights
                tradestopCounter -= 1
                ii += 1
            
            # update portfolioWeights
            if np.size(portfolioWeights) > 0:
                portfolioWeights = np.append(portfolioWeights, currentPortfolioWeights, axis=0)
            else:
                portfolioWeights = currentPortfolioWeights
            
            # update current portfolioValues
            currentPortfolioValue = [portfolioValue[-1]]
            for j in range(1, len(currentTestPriceRelativeVectors)):
                if tradestopCounter > 0:
                    portfolioWeights[-len(currentPortfolioWeights)+j] = allCashWeights
                    tradestopCounter -= 1
                
                value = portfolio.calculateCurrentPortfolioValue(
                            currentPortfolioValue[j-1],
                            currentTestPriceRelativeVectors[j],
                            portfolioWeights[-len(currentPortfolioWeights)+j-1]
                        )
                currentPortfolioValue.append(value)
                portfolioValue.append(value)
                
                # identify tradestop signals based on portfolioValue
                if len(portfolioValue) > longSMA:
                    portfolioValueDF = pd.DataFrame(data={'value': portfolioValue})
                    portfolioValueSMA = ta.sma(portfolioValueDF['value'], length=longSMA)
                    
                    # the newest portfolio value should be smaller than the latest SMA value
                    if value <= portfolioValueSMA.iloc[-1]:
                        portfolioGrowth = np.diff(ta.sma(pd.DataFrame(data={'value': portfolioValue})['value'],
                                                         length=shortSMA))
                        growthInterval = portfolioGrowth[-lookbackDownside:]
                        tradestopSignals = analyzeCurrentDownside(pd.DataFrame(data={'growth': growthInterval})['growth'],
                                                                  shiftIdx=shiftTradestopIdx,
                                                                  cutoffDrop=cutoffDrop,
                                                                  lookback=lookbackDownside)
                        
                        # activate safety mechanism if necessary
                        if len(tradestopSignals) > 0:
                            # set weights to all cash for some time to simulate holding cash
                            for idx in tradestopSignals:
                                if (portfolioWeights[idx][0] != 1.) and (portfolioValue[idx] <= portfolioValueSMA[idx]):
                                    portfolioWeights[idx] = allCashWeights
                                    tradestopCounter = fifteenMinsInOneDay
            
                    # update index to shift tradestop signals
                    shiftTradestopIdx = len(portfolioValue)-lookbackDownside
                
            # update train data
            onlineTrainData = updateOnlineTrainData(onlineTrainData, currentTestData)
            onlinePriceRelativeVectors = updateOnlineTrainData(onlinePriceRelativeVectors, 
                                                               currentTestPriceRelativeVectors)
            onlineOptimalWeights = updateOnlineTrainData(onlineOptimalWeights, currentOptimalTestWeights)
            
            # perform online training
            portfolio.model.train(onlineTrainData, onlineOptimalWeights, onlinePriceRelativeVectors,
                                  minibatchSize, onlineEpochs)
        
        # update datetime interval
        currentLowerRangeTest += datetime.timedelta(weeks=weeksIncrement)
        currentUpperRangeTest += datetime.timedelta(weeks=weeksIncrement)
        if currentUpperRangeTest > endRangeTest:
            currentUpperRangeTest = endRangeTest  # ensure data is not out of bounds
        
    plotPortfolioValueChange(portfolioValue, startRangeTest, endRangeTest, startRange, endRange)

    