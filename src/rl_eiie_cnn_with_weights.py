# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:20:46 2022

@author: slaven
"""

import datetime
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Flatten, Lambda, Concatenate

from utils.utils import prepareData
from utils.plot_utils import plotPortfolioValueChange


def expandDimensions(weights):
    # CPU mode None x 4 x 1 x 1: add a new axis at the end of the tensor
    expandedWeights = tf.expand_dims(weights, axis=-1)
    expandedWeights = tf.expand_dims(expandedWeights, axis=-1)
    return expandedWeights

class Portfolio():
    # the logits of the portfolio weights are saved here (use tf.nn.softmax() to obtain the actual weights)
    portfolioVectorMemory = []
    
    def initializePortfolioVectorMemory(self):
        # Actually, it saves the logits as these are in fact glued for the weight input layer
        # logits need to be softmaxed to obtain the weights,
        # but softmaxing happens after the weight input layer has been concatenated to the$
        # main neural network at level of *logits* in fact
        # Also, logits offer a better numerical stability
        # Note: the first idx simply uses the weights 1 resp 0 to keep things simple
        self.portfolioVectorMemory = []  # TODO : first one is 1 for cash rest 0 (initially, all money is in cash simply)
    
    
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
    NOTE: the weights are also known as the **portfolio vector memory** or PVM in the paper
    NOTE: The PVM excludes the cash weights. This implementation uses the cash weights anyway
          if it is present in the input data (often simulated as BUSDUSDT)
    """
    # TODO : maybe refactor to use input shapes directly as params and not X_tensor etc, since i only need the shape
    def createEiieCnnWithWeights(self, X_tensor, weights):
        mainInputShape = np.shape(X_tensor)[1:]
        weightsInputShape = np.shape(weights)[1:]
        
        # prefer functional API for its flexibility for future model extensions
        mainInputLayer = Input(shape=mainInputShape, name='main_input_layer')
        main = Conv2D(filters=2, kernel_size=(1, 3), activation='relu', name='first_conv_layer')(mainInputLayer)
        main = Conv2D(filters=20, kernel_size=(1, 48), activation='relu', name='second_conv_layer')(main)
        # intermediateOutputs = Conv2D(filters=20, kernel_size=(1, 48), activation='relu', name='intermediate_output_layer')(main)
        
        # create layers for input weights
        weightsInputLayer = Input(shape=weightsInputShape, name='weights_input_layer')
        # weightsExpanded = Lambda(expandDimensions, name='weights_expansion_layer')(weightsInputLayer)
        
        # # Concatenate the weightsLayer to the mainLayer
        # main = Concatenate(axis=3, name='weights_concatenation_layer')([main, weightsExpanded])
        
        main = Conv2D(filters=1, kernel_size=(1, 1), name='final_conv_layer')(main)
        
        # NOTE: no need to apply softmax. Use logits, they are more numerically more stable in the CategoricalCrossentropy loss function
        # CategoricalCrossentropy applies a softmax too even if from_logits=False is set, because the output is not understood properly
        outputLogits = Flatten()(main)  # bring it into the right shape
        
        eiieCnnWithWeightsModel = CustomModel(inputs=[mainInputLayer, weightsInputLayer], outputs=outputLogits, name='eiie_cnn_with_weights')
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


# TODO : custom model is probably not needed anymore with the current setup
# TODO : custom train loop where i obtain the intermediate fitted weights and give them as input in the next epoch
# this is the concept of the portfolio vector memory (pvm)
class CustomModel(tf.keras.Model):
    pvm = [[1./6 for i in range(6)]]  # strictly uniform weights for all
    rewardPerEpisode = []  # all r_t
    
    
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
        print(currentPriceRelativeVector)
        print(prevPortfolioWeights)
        print(currentPriceRelativeVector @ prevPortfolioWeights)
        return np.log(currentPriceRelativeVector @ prevPortfolioWeights)
    
    
    """
    EQUATION 22: R = 1/t_f * sum(r_t, start=1, end=t_f+1)
    This cumulated reward function is used for optimization for gradient *ASCENT*
    This function is also used as "loss" when optimizing the neural network weights
    during learning
    
    :param currentPriceRelativeVector, y_t from the current period t
    :param prevPortfolioWeights, w_t-1 weights at the beginning of period t AFTER capital reallocation
    
    return: R, average logarithmic cumulated reward (negative value for gradient ASCENT)
    """
    def cumulatedReturn(self, currentPriceRelativeVector, prevPortfolioWeights):
        currentPriceRelativeVector = currentPriceRelativeVector.numpy()
        prevPortfolioWeights = prevPortfolioWeights.numpy()
        individualReward = -np.log(currentPriceRelativeVector @ prevPortfolioWeights)
        print('indiv. reward: {}'.format(individualReward))
        self.rewardPerEpisode.append(individualReward)
        averageCumulatedReturn = sum(self.rewardPerEpisode)/len(self.rewardPerEpisode)
        return -tf.cast(averageCumulatedReturn, tf.float32)


    """
    Implementation of the custom training loop from scratch.
    This is necessary, since we need the intermediate weights as well which
    are used as the additional inputs in the EIIE CNN.
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example    
    
    :param data, the full training data
    :param weights, only needed for crossentropy loss function  # TODO : remove with custom loss
    :param epochs, epochs to iterate over for the most outer for-loop
    
    """    
    def train(self, data, weights, epochs, priceRelativeVectors=None):
        print(np.shape(data))
        for epoch in range(epochs):
            print("\nSTART OF EPOCH {}".format(epoch))
            self.pvm = [[1./6 for i in range(6)]]
            
            for i, (train_batch, weights_batch)\
                in enumerate(zip(data, weights), start=1):
                
                # loss = tf.Variable(0.)
                with tf.GradientTape() as tape:
                    # tape.watch(loss)
                    predictedPortfolioWeights = self([np.expand_dims(train_batch, axis=0),
                                                      np.expand_dims(self.pvm[i-1], axis=0)],
                                                     training=True)

                    loss = self.compiled_loss(tf.convert_to_tensor(weights_batch),
                                              tf.convert_to_tensor(predictedPortfolioWeights[0]),
                                              regularization_losses=self.losses)
                    # loss = self.cumulatedReturn(price_relative_batch, self.pvm[-1])
                
                # print('simple loss: {}'.format(loss0))
                self.pvm.append(tf.nn.softmax(predictedPortfolioWeights[0]))
                print('\npredicted port. weights: \n{}'.format(tf.nn.softmax(predictedPortfolioWeights[0])))
                
                # compute the gradient now
                gradients = tape.gradient(loss, self.trainable_weights)
                # print('my gradients: {}'.format(gradients))
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                # Update metrics (includes the metric that tracks the loss)
                self.compiled_metrics.update_state(tf.convert_to_tensor(weights_batch),
                                                   tf.convert_to_tensor(predictedPortfolioWeights[0]))
                print('Current loss: {}'.format(loss))
    
    
    
    # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    def train_step(self, data):
        trainDataWithWeights, optimalPortfolioWeights = data
        
        with tf.GradientTape() as tape:
            predictedPortfolioWeights = self([trainDataWithWeights[0], trainDataWithWeights[1]],
                                              training=True)
            
            # loss = self.compiled_loss(optimalPortfolioWeights, predictedPortfolioWeights,
            #                           regularization_losses = self.losses)
            loss = self.compiled_loss(tf.convert_to_tensor(trainDataWithWeights[-1]),
                                      tf.convert_to_tensor(predictedPortfolioWeights),
                                      regularization_losses = self.losses)
        
        # compute the gradient now
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(optimalPortfolioWeights, predictedPortfolioWeights)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


if __name__ == '__main__':
    # enforce CPU mode (for GPU mode, set 'channels_first' and modify tensor shapes accordingly)
    K.set_image_data_format('channels_last')
    
    # define a few neural network specific variables
    epochs = 2  # 1200
    window = 50
    learning_rate = 0.00019
    
    # prepare train data
    startRange = datetime.datetime(2022,6,17,0,0,0)
    endRange = datetime.datetime(2022,6,22,0,0,0)
    markets = ['BUSDUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'MATICUSDT']
    
    data, priceRelativeVectors, _ = prepareData(startRange, endRange, markets, window)
        
    # start portfolio simulation
    portfolio = Portfolio()
    portfolio.createEiieCnnWithWeights(data, priceRelativeVectors)
    
    # with the simulated y_true (optimalWeights), categorical crossentropy loss makes the most sense
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    portfolio.model.compile(optimizer=optimizer,
                            loss=portfolio.model.cumulatedReturn,
                            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics='accuracy')
    
    # simulate y_true
    optimalWeights = portfolio.generateOptimalWeights(priceRelativeVectors)
    portfolio.model.fit(x=[data, optimalWeights, priceRelativeVectors],
                        y=optimalWeights,
                        epochs=epochs)
    # portfolio.model.train(data, optimalWeights, epochs, priceRelativeVectors)
    
    # prepare test data
    startRangeTest = datetime.datetime(2022,6,24,0,0,0)
    endRangeTest = datetime.datetime(2022,6,25,0,0,0)
    
    # update y_true for new time range
    testData, testPriceRelativeVector, _ = prepareData(startRangeTest, endRangeTest, markets, window)
    optimalTestWeights = portfolio.generateOptimalWeights(testPriceRelativeVector)
    
    # get logits which are used to obtain the portfolio weights with tf.nn.softmax(logits)
    logits = portfolio.model.predict([testData, optimalTestWeights])
    
    # Calculate and visualize how the portfolio value changes over time
    portfolioValue = [10000]
    portfolioWeights = tf.nn.softmax(logits)
    for i in range(1,len(testPriceRelativeVector)):
        portfolioValue.append(
            portfolio.calculateCurrentPortfolioValue(portfolioValue[i-1], np.asarray(testPriceRelativeVector[i]), np.asarray(portfolioWeights[i-1])))
    
    plotPortfolioValueChange(portfolioValue, startRangeTest, endRangeTest)