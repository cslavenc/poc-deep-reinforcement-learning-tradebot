# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 17:41:46 2022

@author: slaven
"""

import datetime
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Flatten, Lambda, Concatenate

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
    def createEiieCnnWithWeights(self, X_tensor, weights, cashBias):
        mainInputShape = np.shape(X_tensor)[1:]
        weightsInputShape = np.shape(weights)[1:]
        cashBiasInputShape = np.shape(cashBias)[1:]
        
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
        
        cashBiasInputLayer = Input(shape=cashBiasInputShape, name='cash_bias_input_layer')
        cashBiasExpanded = Lambda(expandDimensions, name='cash_bias_expansion_layer')(cashBiasInputLayer)
        
        main = Concatenate(axis=1, name='cash_bias_concatenation_layer')([cashBiasExpanded, main])
        
        # NOTE: no need to apply softmax. Use logits, they are more numerically more stable in the CategoricalCrossentropy loss function
        # CategoricalCrossentropy applies a softmax too even if from_logits=False is set, because the output is not understood properly
        outputLogits = Flatten()(main)  # bring it into the right shape
        
        eiieCnnWithWeightsAndCashBiasModel = CustomModel(inputs=[mainInputLayer,weightsInputLayer,
                                                      cashBiasInputLayer],
                                              outputs=outputLogits,
                                              name='eiie_cnn_with_weights_and_cash_bias')
        self.model = eiieCnnWithWeightsAndCashBiasModel
    
    
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

    """
    Implementation of the custom training loop from scratch.
    This is necessary, since we need the intermediate weights as well which
    are used as the additional inputs in the EIIE CNN.
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example    
    
    :param data, the full training data
    :param weights, only needed for crossentropy loss function
    :param minibatchSize
    :param epochs, epochs to iterate over for the most outer for-loop
    
    """    
    def train(self, data, weights, cashBias, minibatchSize, epochs):
        # prepare for minibatch evaluation
        originalMinibatchSize = minibatchSize
        dataSize = data.shape[0]  # size of the time series
        
        # if numOfMiniBatches = 13.4, it becomes 14, but the 14th minibatch has a smaller minibatch size
        numOfMiniBatches = int(np.ceil(dataSize/minibatchSize))
        
        for epoch in range(epochs):
            print("\nSTART OF EPOCH {}".format(epoch))
            minibatchSize = originalMinibatchSize  # reset
            
            # TODO : are there better starting values? all in cash simply?
            # reset and use optimal weights as default values for now
            self.portfolioVectorMemory.append(weights[0:minibatchSize])
            lossTracker = []
            
            for i in range(1, numOfMiniBatches):
                # check if minibatch size is not too big and make it smaller if it does not fit the dataset
                if (i+1)*minibatchSize >= dataSize:
                    minibatchSize = (i+1)*minibatchSize - dataSize - 1
                
                with tf.GradientTape() as tape:
                    predictedPortfolioLogits = self([data[(i*minibatchSize):((i+1)*minibatchSize), 1:],
                                                     self.portfolioVectorMemory[i-1][0:minibatchSize, 1:],  # w_t-1, weights from previous minibatch
                                                     cashBias[(i*minibatchSize):((i+1)*minibatchSize)]],
                                                     training=True)
                    # logits are automatically softmaxed here
                    loss = self.compiled_loss(tf.convert_to_tensor(weights[(i*minibatchSize):((i+1)*minibatchSize)]),
                                              tf.convert_to_tensor(predictedPortfolioLogits),
                                              regularization_losses=self.losses)
                
                # FIGURE 3a: this adds the portfolio vector memory of the minibatch
                self.portfolioVectorMemory.append(tf.nn.softmax(predictedPortfolioLogits).numpy())
                
                lossTracker.append(loss)
                # compute the gradient now
                gradients = tape.gradient(loss, self.trainable_weights)
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                # Update metrics (includes the metric that tracks the loss)
                self.compiled_metrics.update_state(tf.convert_to_tensor(weights[(i*minibatchSize):((i+1)*minibatchSize)]),
                                                   tf.convert_to_tensor(predictedPortfolioLogits))
            
            # loss tracker and accuracy printer might be nice, but it costs more computational power so maybe just ignore it
            print('Current loss: {}'.format(np.mean(lossTracker)))
            self.compiled_metrics.reset_state()
            self.portfolioVectorMemory = []  # reset for the next epoch


if __name__ == '__main__':
    # enforce CPU mode (for GPU mode, set 'channels_first' and modify tensor shapes accordingly)
    K.set_image_data_format('channels_last')
    
    # define a few neural network specific variables
    epochs = 1200  # 1200
    window = 50
    minibatchSize = 32
    learning_rate = 0.00019
    
    # prepare train data
    startRange = datetime.datetime(2022,6,17,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    # TODO : remove BUSDUSDT to have smaller vectors...
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'MATICUSDT']
    
    data, priceRelativeVectors, _ = prepareData(startRange, endRange, markets, window)
    cashBias = np.ones(shape=(priceRelativeVectors.shape[0], 1), dtype=float)
        
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data[:, 1:], priceRelativeVectors[:, 1:], cashBias)
    
    # with the simulated y_true (optimalWeights), categorical crossentropy loss makes the most sense
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            run_eagerly=True,
                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics='accuracy')
    
    # simulate y_true
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    portfolio.model.train(data=data, weights=optimalWeights, cashBias=cashBias,
                          minibatchSize=minibatchSize, epochs=epochs)
    
    # prepare test data
    startRangeTest = datetime.datetime(2022,6,24,0,0,0)
    endRangeTest = datetime.datetime(2022,6,25,0,0,0)
    
    # update y_true for new time range
    testData, testPriceRelativeVectors, _ = prepareData(startRangeTest, endRangeTest, markets, window)
    optimalTestWeights = portfolio.generateOptimalWeights(testPriceRelativeVectors)
    testCashBias = np.ones(shape=(testPriceRelativeVectors.shape[0], 1), dtype=float)
    
    # get logits which are used to obtain the portfolio weights with tf.nn.softmax(logits)
    logits = portfolio.model.predict([testData[:, 1:], optimalTestWeights[:, 1:], testCashBias])
    
    # Calculate and visualize how the portfolio value changes over time
    portfolioValue = [10000.]
    portfolioWeights = tf.nn.softmax(logits)
    for i in range(1,len(testPriceRelativeVectors)):
        portfolioValue.append(
            portfolio.calculateCurrentPortfolioValue(portfolioValue[i-1], np.asarray(testPriceRelativeVectors[i]), np.asarray(portfolioWeights[i-1])))
    
    plotPortfolioValueChange(portfolioValue, startRangeTest, endRangeTest)