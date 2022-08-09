# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 08:54:45 2022

@author: slaven
"""

import datetime
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Flatten, Lambda, Concatenate, Softmax

from utils.utils import prepareData
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
    :param previousPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: p_t, current portfolio value (current net worth)
    """
    def calculateCurrentPortfolioValue(self, prevPortfolioValue, currentPriceRelativeVector, previousPortfolioWeights):
        return prevPortfolioValue * (currentPriceRelativeVector @ previousPortfolioWeights)  # p_t = p_t-1 * <y_t, w_t-1>
    
    
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
        main = Conv2D(filters=2, kernel_size=(1, 3), activation='relu', name='first_conv_layer')(mainInputLayer)
        main = Conv2D(filters=20, kernel_size=(1, 48), activation='relu', name='second_conv_layer')(main)
        
        # create layers for input weights
        weightsInputLayer = Input(shape=weightsInputShape, name='weights_input_layer')
        weightsExpanded = Lambda(expandDimensions, name='weights_expansion_layer')(weightsInputLayer)
        
        # Concatenate the weightsLayer to the mainLayer
        main = Concatenate(axis=3, name='weights_concatenation_layer')([main, weightsExpanded])
        
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
    tradingFees = 0.001  # 0.1% is max rate at binance, which is c_s and c_p from the paper
    transactionRemainderFactor = tf.convert_to_tensor(1.)
    cutoffForConvergence = 15  # k in THEOREM 1 and EQUATION 15
    
    
    """
    Calculate mu.
    """
    def calculateTransactionRemainderFactor(self, priceRelativeVectors, previousPortfolioWeights,
                                            predictedPortfolioWeights):
        # EQUATION 7: w_t' = elwiseMultiplication(y_t, w_t-1) / <y_t, w_t-1>
        multiplied = tf.math.multiply(priceRelativeVectors, previousPortfolioWeights)
        # print('multiplied: {}'.format(multiplied))
        dotProduct = tf.math.reduce_sum(multiplied, axis=1, keepdims=True)
        # print('dotProduct: {}'.format(dotProduct))
        currentPortfolioWeightsPrime = tf.math.divide(multiplied, dotProduct)
        # print('previous portf weights[1]: {}'.format(previousPortfolioWeights[1]))
        # print('wPrime[1]: {}'.format(currentPortfolioWeightsPrime[1]))
        
        # EQUATION 16: initial guess for the first value of the sequence
        transactionRemainderFactor = self.tradingFees * tf.math.reduce_sum(tf.math.abs(currentPortfolioWeightsPrime - predictedPortfolioWeights), axis=0, keepdims=True)
        # print('initial guess mu: {}'.format(transactionRemainderFactor))
        
        # THEOREM 1: (1/(1-c_p*w_t,0)) * (1-c_p*w_t,0' - (c_s + c_p - c_s*c_p) * sum(relu(w_t,i' - mu*w_t,i), start=1, end=numOfAssets))
        for k in range(self.cutoffForConvergence):
            # SUM of EQUATION 14
            multipliedMax = tf.math.maximum(
                (currentPortfolioWeightsPrime - tf.math.multiply(
                    transactionRemainderFactor, predictedPortfolioWeights)), 0)
            # print('multipliedMax: {}'.format(multipliedMax))
            
            transactionRemainderFactorSuffix =  tf.math.reduce_sum(
                multipliedMax, axis=1, keepdims=True)
            # print('mu suffix: {}'.format(transactionRemainderFactorSuffix))
            
            transactionRemainderFactor = tf.math.multiply((1. / (1. - tf.math.multiply(self.tradingFees, predictedPortfolioWeights[0]))),
                                  (1. - tf.math.multiply(self.tradingFees, currentPortfolioWeightsPrime[0])\
                                   - tf.math.multiply((self.tradingFees + self.tradingFees - self.tradingFees*self.tradingFees), transactionRemainderFactorSuffix)))
        # print('mu: {}'.format(transactionRemainderFactor))
        
        return transactionRemainderFactor
    
    
    """
    EQUATION 22: R = 1/t_f * sum(r_t, start=1, end=t_f+1)
    This cumulated reward function is used for optimization for gradient *ASCENT*
    This function is also used as "loss" when optimizing the neural network weights
    during learning. A normal loss function is usually minimized.
    This reward function is maximized, which can be simply achieved by inverting the sign.
    
    
    :param currentPriceRelativeVector, y_t from the current period t
    :param previousPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: R, average logarithmic cumulated reward (negative value for gradient ASCENT)
    """
    def cumulatedReturn(self, currentPriceRelativeVector, previousPortfolioWeights):
        rewardPerEpisode = []
        
        for j in range(currentPriceRelativeVector.shape[0]):
            multiplied = tf.multiply(currentPriceRelativeVector[j], previousPortfolioWeights[j])
            # individualReward = -tf.math.log(tf.multiply(self.transactionRemainderFactor, tf.reduce_sum(multiplied, axis=0)))
            individualReward = -tf.math.log(tf.reduce_sum(multiplied, axis=0))
            rewardPerEpisode.append(individualReward)
        # averageCumulatedReturn = tf.math.reduce_sum(rewardPerEpisode)/len(rewardPerEpisode)
        averageCumulatedReturn = tf.math.reduce_sum(rewardPerEpisode)
        return averageCumulatedReturn


    """
    Implementation of the custom training loop from scratch.
    This is necessary, since we need the intermediate weights as well which
    are used as the additional inputs in the EIIE CNN.
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example    
    
    :param data, the full training data
    :param weights, only needed for crossentropy loss function
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
            print("\nSTART OF EPOCH {}".format(epoch))
            minibatchSize = originalMinibatchSize  # reset
            
            # reset and use optimal weights as default values
            self.portfolioVectorMemory.append(tf.convert_to_tensor(weights[0:minibatchSize], dtype=tf.float32))
            lossTracker = []
            
            for i in range(1, numOfMiniBatches-1):
                # check if minibatch size is not too big and make it smaller if it does not fit the dataset
                if (i+1)*minibatchSize >= dataSize:
                    minibatchSize = (i+1)*minibatchSize - dataSize - 1
                # minibatchSize for futurePortfolioWeights and transactionRemainderFactor
                futureMinibatchSize = priceRelativeVectors[((i+1)*minibatchSize):((i+2)*minibatchSize)].shape[0]
                print(futureMinibatchSize)
                
                # the predicted portfolio weights correspend to timestep t and not t-1 as in the paper
                # this is because when saving the portfolio weights to the portfolioVectorMemory,
                # tensorflow only sees the mere tf.Tensor, but not the operations that were needed to get the value
                # the operations are crucial for tensorflow, because it cannot do backpropagation otherwise
                # (tape.gradient() will not work because the loss would depend on a value
                # from a past iteration where the operations are not available to GradientTape anymore)
                # Therefore, the calculations have been shifted from t (paper) to t+1 (implementation)
                with tf.GradientTape() as tape:
                    predictedPortfolioWeights = self([data[(i*minibatchSize):((i+1)*minibatchSize)],
                                                      # w_t-1, weights from previous minibatch
                                                      self.portfolioVectorMemory[i-1][0:minibatchSize]],
                                                     training=True)
                    # get the portfolio weights at t+1. this is only an approximation,
                    # since the current neural network weights are used to predict the future weights
                    # actually, the gradient should be applied to update the neural network weights first
                    # before predicting the future portfolio weights at t+1
                    futurePortfolioWeights = self([data[((i+1)*minibatchSize):((i+2)*minibatchSize)],
                                                   predictedPortfolioWeights[0:futureMinibatchSize]],
                                                  training=True)
                    
                    # at step t+1, because y_t+1 is used as well with portfolio weights w_t and w_t+1
                    self.transactionRemainderFactor = self.calculateTransactionRemainderFactor(
                        priceRelativeVectors[((i+1)*minibatchSize):((i+2)*minibatchSize)],
                        predictedPortfolioWeights[0:futureMinibatchSize],
                        futurePortfolioWeights[0:futureMinibatchSize])

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
                
            # loss tracker and accuracy printer might be nice, but it costs more computational power so maybe just ignore it
            print('Current loss: {}'.format(np.mean(lossTracker)))
            self.compiled_metrics.reset_state()
            self.portfolioVectorMemory = []  # reset for the next epoch


"""
Sometimes, the cash is slightly below 1, since the current datasets actually looks at
USDCUSDT which itself fluctuates a little bit. This should not be the case actually.
Also, a very small bias has been added to values = 1., because sometimes, a crypto asset
remains constant during a timestep and thus has its price relative value is 1. too and
the neural network could potentially prefer this one over cash.
In itself, this does not need to be a bad thing though.

:param priceRelativeVectors, first entry should be the cash (if applicable)

returns: priceRelativeVectors, which the first entry slightly modified
"""
def sanitizeCashValues(priceRelativeVectors):
    smallCashBias = 0.000001
    for i in range(priceRelativeVectors.shape[0]):
        if priceRelativeVectors[i, 0] <= 1.:
            priceRelativeVectors[i, 0] = 1. + smallCashBias
    
    return priceRelativeVectors


if __name__ == '__main__':
    # enforce CPU mode (for GPU mode, set 'channels_first' and modify tensor shapes accordingly)
    K.set_image_data_format('channels_last')
    
    # define a few neural network specific variables
    epochs = 300
    window = 50
    minibatchSize = 32
    learning_rate = 0.00019
    
    # prepare train data
    startRange = datetime.datetime(2022,6,20,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'MATICUSDT']
    
    data, priceRelativeVectors = prepareData(startRange, endRange, markets, window)
        
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data, priceRelativeVectors)
    
    # with the simulated y_true (optimalWeights), categorical crossentropy loss makes the most sense
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            run_eagerly=True,
                            loss=portfolio.model.cumulatedReturn,
                            metrics='accuracy')
    
    # simulate y_true
    priceRelativeVectors = sanitizeCashValues(priceRelativeVectors)
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    portfolio.model.train(data, optimalWeights, priceRelativeVectors, minibatchSize, epochs)
    
    # prepare test data
    startRangeTest = datetime.datetime(2022,6,22,0,0,0)
    endRangeTest = datetime.datetime(2022,6,24,0,0,0)
    
    # update y_true for new time range
    testData, testPriceRelativeVectors = prepareData(startRangeTest, endRangeTest, markets, window)
    testPriceRelativeVectors = sanitizeCashValues(testPriceRelativeVectors)
    optimalTestWeights = portfolio.generateOptimalWeights(testPriceRelativeVectors)
    
    # # get logits which are used to obtain the portfolio weights with tf.nn.softmax(logits)
    portfolioWeights = portfolio.model.predict([testData, optimalTestWeights])
    
    # Calculate and visualize how the portfolio value changes over time
    portfolioValue = [10000.]
    for i in range(1,len(testPriceRelativeVectors)):
        portfolioValue.append(
            portfolio.calculateCurrentPortfolioValue(portfolioValue[i-1], np.asarray(testPriceRelativeVectors[i]), np.asarray(portfolioWeights[i-1])))
    
    plotPortfolioValueChange(portfolioValue, startRangeTest, endRangeTest, startRange, endRange)