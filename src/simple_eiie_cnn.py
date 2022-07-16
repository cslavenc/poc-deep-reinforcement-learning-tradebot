# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:16:12 2022

@author: slaven
"""

import datetime
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dropout, BatchNormalization, Flatten, Softmax

from utils.utils import getData, formatDataForInput


class Portfolio:
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
    EQUATION 3: rho_t = <y_t, w_t-1> - 1
    
    :param prevPortfolioValue, p_t-1
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: rho_t, current rate of return
    """
    def calculateRateOfReturn(self, prevPortfolioValue, currentPriceRelativeVector, prevPortfolioWeights):
        return self.calculateCurrentPortfolioValue(
            prevPortfolioValue, currentPriceRelativeVector, prevPortfolioWeights) - 1
    
    """
    EQUATION 4: r_t = ln(<y_t, w_t-1>)
    
    :param prevPortfolioValue, p_t-1
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: rho_t, current rate of return
    
    Note: np.log is the NATURAL logarithm, often denoted as ln()
    """
    def calculateLnRateOfReturn(self, prevPortfolioValue, currentPriceRelativeVector, prevPortfolioWeights):
        return np.log(self.calculateRateOfReturn(prevPortfolioValue, currentPriceRelativeVector, prevPortfolioWeights))
    
    
    """
    Simplified implementation of FIGURE 2, page 11
    
    :param X_tensor, input data (CPU mode: timesteps x markets x lookback x features), features = close, high, low, etc.
    
    return: a simple EIIE CNN model
    
    NOTE: if using GPU mode, shapes and kernel sizes need to be adapted
    NOTE: this is a simple EIIE CNN without adding pvm_t-1 = w_t-1[1:] or the cash bias
    """
    def createSimpleEIIECNN(self, X_tensor):
        mainInputShape = np.shape(X_tensor)[1:]
        
        # prefer functional API for its flexibility for future model extensions
        mainInputLayer = Input(shape=mainInputShape, name='main_input_layer')
        main = Conv2D(filters=2, kernel_size=(1, 3), activation='relu', name='first_conv_layer')(mainInputLayer)
        main = Conv2D(filters=20, kernel_size=(1, 48), activation='relu', name='second_conv_layer')(main)
        main = Conv2D(filters=1, kernel_size=(1, 1), name='final_conv_layer')(main)
        # NOTE: no need to apply softmax. Use logits, they are more numerically stable in the CategoricalCrossentropy loss function
        outputWeights = Flatten()(main)  # bring it to the right shape before applying softmax
        # outputWeights = Softmax(axis=0, name='softmax_layer')(main)
        # outputWeights = tf.squeeze(tf.squeeze(outputWeights, 2), 2)  # bring it into the right shape
        
        simpleEiieCnnModel = Model(inputs=mainInputLayer, outputs=outputWeights, name='simple_eiie_cnn')
        
        self.model = simpleEiieCnnModel


"""
EQUATION 21, apparently, this is one of the loss functions that the original authors use in their (updated) repo
https://github.com/ZhengyaoJiang/PGPortfolio

:param y_true - futurePriceRelativeVector, y_t+1 of current period t (y_t+1 = v_t+1/v_t)
:param y_pred - weightsOutput, weights output of the network w_t

return: loss, something similar like a cross entropy loss

NOTE: w_t is NOT w_t-1 from EQUATION 2
NOTE: y_true and y_pred are used as names since its a requirement by keras to define a custom loss function
      Technically, this is incorrect, since y_true has the futureRelativePriceVectors
      while y_pred are the predicted portfolio weights 
"""
def custom_loss_fn(y_true, y_pred):
    return -tf.math.log(tf.math.reduce_sum(tf.multiply(y_true, y_pred), axis=0))


"""
:param priceRelativeVectors, current priceRelativeVectors

return futurePriceRelativeVectors

NOTE: this is basically shifting priceRelativeVectors by 1.
      The last entry is just a duplication. Correctly, the actual value has to be added somehow
      by either extending the priceRelativeVectors by 1 or in some other way
"""
def generateFuturePriceRelativeVectors(priceRelativeVectors):
    futurePriceRelativeVectors = priceRelativeVectors[1:]
    futurePriceRelativeVectors.append(priceRelativeVectors[-1])
    return np.asarray(futurePriceRelativeVectors)

"""
Generate the optimal weights, which is allocating everything into the asset that grew the most.
Increase is usually >1. This is used as a simulated "y_true" for a cross entropy loss function.

:param priceRelativeVectors

return: optimized weights, where 1 for the asset with the highest increase and else 0

NOTE(TODO) : Keep in mind that currently, this will also allocate everything into the
asset that decreased the least, since cash is not considered here
"""
def generateOptimalWeights(priceRelativeVectors):
    optimalWeights = []
    optimalAssetIds = np.argmax(priceRelativeVectors, axis=1)
    
    for i in range(np.shape(priceRelativeVectors)[0]):
        tempWeights = [0. for _ in range(np.shape(priceRelativeVectors)[1])]
        tempWeights[optimalAssetIds[i]] = 1.
        optimalWeights.append(tempWeights)
    
    return np.asarray(optimalWeights)


"""
BASIC implementation of the EIIE CNN at figure 2 (page 14) in the paper without concatenating other tensors:
    A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
This implementation is only for investigative purposes of the EIIE CNN and does not include
the reinforcement learning environment.
"""
if __name__ == '__main__':
    # define a few neural network specific variables
    epochs = 50000
    window = 50
    learning_rate = 0.00019
    
    # enforce CPU mode (for GPU mode, set 'channels_first' and modify tensor shapes accordingly)
    K.set_image_data_format('channels_last')
    
    # prepare data
    startRange = datetime.datetime(2022,6,16,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
    
    
    # final shape CPU mode: timesteps x markets x lookback x features, features = channels = (close, high, low, ...)
    data = []
    # EQUATION 1: y_t = (v_0,t/v_0,t-1 | v_BTC,t/v_BTC,t-1 | ... | v_ADA,t/v_ADA,t-1), with v_0 = 1 for all t (v_0 is the cash)
    priceRelativeVectors = []
    # rates are the closePrices of all the assets used to calculate the relativePriceVector for various t
    rates = []
    for market in markets:
        rawData = getData(startRange, endRange, market)
        formattedData, priceRelativeVector, ratesMarket = formatDataForInput(rawData, window)
        data.append(formattedData)
        priceRelativeVectors.append(priceRelativeVector)
        rates.append(ratesMarket)
    
    # get them into the right shape
    data = np.swapaxes(np.swapaxes(data, 2, 3), 0, 1)  # the tensor X_t (page 9, EQUATION 18)
    priceRelativeVectors = np.transpose(priceRelativeVectors).tolist()
    rates = np.transpose(rates).tolist()
    
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createSimpleEIIECNN(data)
    
    # with the simulated y_true, categorical crossentropy loss makes the most sense
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics='accuracy')
    
    # simulate y_true
    optimalWeights = generateOptimalWeights(priceRelativeVectors)
    portfolio.model.fit(x=data, y=optimalWeights, epochs=epochs)
    # predictions = portfolio.model.predict()