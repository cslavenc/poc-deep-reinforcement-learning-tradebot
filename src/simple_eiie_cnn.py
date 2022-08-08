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
from tensorflow.keras.layers import Conv2D, Input, Flatten

from utils.utils import prepareData
from utils.plot_utils import plotPortfolioValueChange


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
    
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: rho_t, current rate of return
    """
    def calculateRateOfReturn(self, currentPriceRelativeVector, prevPortfolioWeights):
        return (currentPriceRelativeVector @ prevPortfolioWeights) - 1
    
    """
    EQUATION 4: r_t = ln(<y_t, w_t-1>)
    
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: rho_t, current rate of return
    
    Note: np.log is the NATURAL logarithm, often denoted as ln()
    """
    def calculateLnRateOfReturn(self, currentPriceRelativeVector, prevPortfolioWeights):
        return np.log(self.calculateRateOfReturn(currentPriceRelativeVector, prevPortfolioWeights))
    
    
    """
    Simplified implementation of FIGURE 2, page 14
    
    :param X_tensor, input data (CPU mode: timesteps x markets x lookback x features), features = close, high, low, etc.
    
    return: a simple EIIE CNN model
    
    NOTE: if using GPU mode, shapes and kernel sizes need to be adapted
    NOTE: this is a simple EIIE CNN without adding pvm_t-1 = w_t-1[1:] or the cash bias
    NOTE: prefer functional API for its flexibility for future model extensions
    """
    def createSimpleEIIECNN(self, X_tensor):
        mainInputShape = np.shape(X_tensor)[1:]
        
        mainInputLayer = Input(shape=mainInputShape, name='main_input_layer')
        main = Conv2D(filters=2, kernel_size=(1, 3), activation='relu', name='first_conv_layer')(mainInputLayer)
        main = Conv2D(filters=20, kernel_size=(1, 48), activation='relu', name='second_conv_layer')(main)
        main = Conv2D(filters=1, kernel_size=(1, 1), name='final_conv_layer')(main)
        # NOTE: no need to apply softmax. Use logits, they are more numerically more stable in the CategoricalCrossentropy loss function
        # CategoricalCrossentropy applies a softmax too even if from_logits=False is set, because the output is not understood properly
        outputLogits = Flatten()(main)  # bring it into the right shape
        
        simpleEiieCnnModel = Model(inputs=mainInputLayer, outputs=outputLogits, name='simple_eiie_cnn')
        self.model = simpleEiieCnnModel


    """
    Generate the optimal weights, which is allocating everything into the asset that grew the most.
    Increase is usually >1. This is used as a simulated "y_true" for the cross entropy loss function.
    
    :param priceRelativeVectors, the highest increase/lowest decrease get weight 1., others weight 0.
    
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
BASIC implementation of the EIIE CNN at figure 2 (page 14) in the paper without concatenating other tensors:
    A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
This basic implementation is only for investigative purposes of the EIIE CNN and does not include
the reinforcement learning environment.

The labels for the CategoricalCrossentropy loss function are generated by taking
the argmax of the relativePriceVectors (y_t). This is NOT how the papers implements the actual learning process.

Essentially, the neural network tries to learn in what asset to allocate 100% of the portfolio.
Using BUSDUSDT simulates the allocation to cash. Ideally, the entire portfolio should be allocated
into stablecoins, when the market is trending down in a strong and/or sustained matter
(e.g. when BTC is rejected at some level).

Note that allocating the entire portfolio into stablecoins during sudden downtrends
is not fully established in this very basic version of a EIIE CNN as it does not
sufficiently take the history into account!
In any case, it tends to trend like the general market (in both directions).
"""
if __name__ == '__main__':
    # enforce CPU mode (for GPU mode, set 'channels_first' and modify tensor shapes accordingly)
    K.set_image_data_format('channels_last')
    
    # define a few neural network specific variables
    epochs = 1200
    window = 50
    learning_rate = 0.00019
    
    # prepare train data
    startRange = datetime.datetime(2022,6,17,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'MATICUSDT']
    
    data, priceRelativeVectors, _ = prepareData(startRange, endRange, markets, window)
        
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createSimpleEIIECNN(data)
    
    # with the simulated y_true (optimalWeights), categorical crossentropy loss makes the most sense
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics='accuracy')
    
    # simulate y_true
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    portfolio.model.fit(x=data, y=optimalWeights, epochs=epochs)
    
    # prepare test data
    startRangeTest = datetime.datetime(2022,6,24,0,0,0)
    endRangeTest = datetime.datetime(2022,6,25,0,0,0)
    
    testData, testPriceRelativeVector, _ = prepareData(startRangeTest, endRangeTest, markets, window)
    
    # get logits which are used to obtain the portfolio weights with tf.nn.softmax(logits)
    logits = portfolio.model.predict(testData)
    
    # Calculate and visualize how the portfolio value changes over time
    portfolioValue = [10000.]
    portfolioWeights = tf.nn.softmax(logits)
    for i in range(1,len(testPriceRelativeVector)):
        portfolioValue.append(
            portfolio.calculateCurrentPortfolioValue(portfolioValue[i-1], np.asarray(testPriceRelativeVector[i]), np.asarray(portfolioWeights[i-1])))
    
    plotPortfolioValueChange(portfolioValue, startRangeTest, endRangeTest)