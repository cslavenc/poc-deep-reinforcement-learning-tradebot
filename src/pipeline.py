# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:37:20 2023

@author: slaven
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Flatten, Lambda, Concatenate, Softmax

from utils.utils import prepareData, analyzeCurrentDownside


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


def findClosestQuarterlyMinute(mins):
    multiple = mins//15
    return 15 * multiple

# TODO : how does the actual and calculated/predicted portfolio value diverge? can it be neglected?
if __name__ == '__main__':
    # enforce CPU mode
    K.set_image_data_format('channels_last')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # required to fully enforce CPU usage
    
    window = 50
    learning_rate = 0.00019
    performOnlineTraining = True if sys.argv[1] == 'true' else False
    
    now = datetime.datetime.utcnow()
    endRange = datetime.datetime(now.year, now.month, now.day, now.hour, 
                                 findClosestQuarterlyMinute(now.minute), 0)
    # longSMA has close to a month, so choose a large enough startRange
    startRange = endRange - datetime.timedelta(days=27)
    
    markets = ['BUSDUSDT_15m', 'BTCUSDT_15m', 'ETHUSDT_15m', 'BNBUSDT_15m',
               'ADAUSDT_15m', 'MATICUSDT_15m']
    
    data, priceRelativeVectors = prepareData(startRange, endRange, markets, window)
    
    # create a new model
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data, priceRelativeVectors)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            run_eagerly=True,
                            loss=portfolio.model.cumulatedReturn,
                            metrics='accuracy')
    
    # load saved model
    pretrainedModel = tf.keras.models.load_model('models/saved_model_3w_2d', 
                                                  compile=False)
    portfolio.model.set_weights(pretrainedModel.get_weights())
    
    # simulate y_true
    priceRelativeVectors = sanitizeCashValues(priceRelativeVectors)
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    
    onlineEpochs = 10
    minibatchSize = 32
    longSMA = 2500
    shortSMA = 100
    tradestopDuration = 4*24*2  # there are 4*24 15 mins per day
    last3Weeks = 4*24*7*3       # number of 15mins during n weeks
    lookbackDownside = 200
    cutoffDrop = -0.08
    allCashWeights = np.array([1.] + [0. for _ in range(len(markets)-1)])
    
    # predict on latest datapoint
    currentPortfolioWeights = portfolio.model.predict([data[-1:], 
                                                       optimalWeights[-1:]])
    currentPortfolioWeights = np.asarray(currentPortfolioWeights)
    
    # decrement tradestop counter (enables trading if tradestopCounter=0)
    tradestopCounter = int(np.loadtxt('tradestop.txt'))
    if tradestopCounter > 0:
        currentPortfolioWeights[0] = allCashWeights
        tradestopCounter -= 1
        np.savetxt('tradestop.txt', [tradestopCounter], fmt='%i')
    
    # calculate current portfolio value
    portfolioWeights = np.loadtxt('weights.txt')
    portfolioValues = np.loadtxt('values.txt')
    
    value = portfolio.calculateCurrentPortfolioValue(
                portfolioValues[-1],
                priceRelativeVectors[-1:],
                portfolioWeights[-1]
            )
    
    # update portfolio weights and values
    portfolioValues = np.append(portfolioValues, value)
    portfolioWeights = np.append(portfolioWeights, currentPortfolioWeights, axis=0)
    
    if len(portfolioValues) > longSMA:
        portfolioValuesDF = pd.DataFrame(data={'value': portfolioValues})
        portfolioValuesSMA = ta.sma(portfolioValuesDF['value'], length=longSMA)
        
        # the newest portfolio value should be smaller than the latest SMA value
        if value <= portfolioValuesSMA.iloc[-1]:
            portfolioGrowth = np.diff(ta.sma(pd.DataFrame(data={'value': portfolioValues})['value'],
                                             length=shortSMA))
            growthInterval = portfolioGrowth[-lookbackDownside:]
            tradestopSignals = analyzeCurrentDownside(pd.DataFrame(data={'growth': growthInterval})['growth'],
                                                      cutoffDrop=cutoffDrop,
                                                      lookback=lookbackDownside)
            
            # activate safety mechanism if necessary
            if len(tradestopSignals) > 0:
                # set weights to all cash for some time to simulate holding cash
                if (portfolioWeights[-1][0] != 1.) and (portfolioValues[-1] <= portfolioValuesSMA.iloc[-1]):
                    portfolioWeights[-1] = allCashWeights
                    np.savetxt('tradestop.txt', [tradestopDuration], fmt='%i')
    
    # save portfolio weights and values
    np.savetxt('weights.txt', np.round(portfolioWeights, 3))
    np.savetxt('values.txt', portfolioValues)
    
    if performOnlineTraining:
        onlineTrainData = data[-last3Weeks:]
        onlinePriceRelativeVectors = priceRelativeVectors[-last3Weeks:]
        onlineOptimalWeights = optimalWeights[-last3Weeks:]
        
        # perform online training
        portfolio.model.train(onlineTrainData, onlineOptimalWeights, onlinePriceRelativeVectors,
                              minibatchSize, onlineEpochs)
        # save trained model
        portfolio.model.save("models/saved_model_test")
