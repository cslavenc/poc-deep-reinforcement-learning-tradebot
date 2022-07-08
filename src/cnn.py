# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 03:49:14 2022

@author: slaven
"""

import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from numpy.random import geometric
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, Input, Dropout, BatchNormalization,\
    Activation, Softmax, Lambda, Concatenate

from utils.utils import getData, formatDataForInput


def expandDimensions(weights):
    expandedWeights = K.expand_dims(weights, axis=1)
    expandedWeights = K.expand_dims(expandedWeights, axis=1)
    return expandedWeights

# Simulated crypto portfolio
class Portfolio:
    def __init__(self, symbols, weights, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs, noop=False, failureChance=0):
        self.symbols = symbols
        self.setWeights(weights)
        self.pBeta = pBeta
        self.k = k
        self.learningRate = learningRate
        self.minibatchCount = minibatchCount
        self.minibatchSize = minibatchSize
        self.epochs = epochs
        self.reset()
        self.tradeFee = 0.0005
        self.tradePer = 1.0 - self.tradeFee
        self.value = 1.0
        self.noop = noop
        self.failureChance = failureChance  # needed?
    
    """
    :param inputTensor, data with shape TODO
    :param rates, TODO
    """
    def createEIIECNN(self, inputTensor, rates):
        # TODO : enable GPU support
        tf.compat.v1.disable_eager_execution()  # TODO : better to use GradientTape instead
        # TODO : copy CNN architecture from the paper
        # TODO(check) : they have a btc_bias var, which is their cash. what to do with it?
        # TODO(check) : these layers have a batch_size arg, does this help or do i need more fine grained control? 
        
        # define variables        
        mainInputShape = np.array(inputTensor).shape[1:]
        weightInputShape = np.array(rates).shape[1:]
        biasInputShape = (1, )
        print('main input shape: {}, weight input shape: {}'.format(mainInputShape, weightInputShape))
        
        # create EIIE CNN with the functional API due to its flexibility: https://www.tensorflow.org/guide/keras/functional
        # define the basic input tensor shape in a separate layer
        inputLayer = Input(shape=mainInputShape, name='main_input_layer')
        # print('After input     : {}'.format(tf.shape(inputLayer)))
        base = Conv2D(filters=2, kernel_size=(3, 1), name='first_conv')(inputLayer)
        # print('After first conv: {}'.format(tf.shape(base)))
        base = Activation('relu')(base)
        # print('After first relu: {}'.format(tf.shape(base)))
        base = Conv2D(filters=20, kernel_size=(48, 1), name='second_conv')(base)
        # print('After second conv: {}'.format(tf.shape(base)))
        base = Activation('relu')(base)
        # print('After second relu: {}'.format(tf.shape(base)))
        
        # create layer for portfolio weights
        weightsInput = Input(shape=weightInputShape, name='weights_input_layer')
        weightsExpanded = Lambda(expandDimensions, name='weights_expansion_layer')(weightsInput)  # layer that acts like a lambda function
        # print('After weights input: {}'.format(tf.shape(weightsExpanded)))
        # CPU mode: None x 1 x 1 x 4 to None x 1 x 4 x 1
        weightsExpanded = tf.transpose(weightsExpanded, [0,1,3,2])
        # concatenate base layer with portfolio weights layer
        # TODO : CPU: axis=3, GPU: axis=1
        base = Concatenate(axis=3, name='weights_concatenation_layer')([base, weightsExpanded])
        base = Conv2D(1, (1, 1))(base)
        
        # create bias layer
        biasInput = Input(shape=biasInputShape, name='bias_input_layer')
        biasExpanded = Lambda(expandDimensions, name='bias_expansion_layer')(biasInput)
        print('After bias expanded: {}'.format(tf.shape(biasExpanded)))
        
        # TODO : CPU axis=2, GPU axis=3
        base = Concatenate(axis=2, name='bias_concatenation_layer')([base, biasExpanded])
        baseOut = Activation('softmax', name='output_layer')(base)
        print('Final output: {}'.format(tf.shape(baseOut)))
        
        self.model = Model([inputLayer, weightsInput, biasInput], baseOut)
        
        # Instantiate custom symbolic gradient
        mu = K.placeholder(shape=(None, 1), name='mu')
        y = K.placeholder(shape=(None, len(self.symbols)+1), name='y')
        
        # tf.keras.backend.squeeze removes 1-dim (opposite of expand_dims) to fit the shape
        # GPU mode
        # sqOut = K.squeeze(K.squeeze(self.model.output, 1), 1)
        # CPU mode
        print(self.model.output)
        sqOut = K.squeeze(K.squeeze(self.model.output, 1), 2)
        yOutMult = tf.multiply(sqOut, y)
        
        yOutBatchDot = tf.reduce_sum(yOutMult, axis=1, keepdims=True)
        muDotMult = tf.multiply(mu, yOutBatchDot)
        
        loss = -K.log(muDotMult)
        grad = K.gradients(loss, self.model.trainable_weights)
        
        self.getGradient = K.function(inputs=[inputLayer, weightsInput, biasInput, mu, y, self.model.output], outputs=grad) 
    
    
    # Re-initialize portfolio state
    def reset(self):
        self.weights = [1.] + [0. for i in self.symbols[1:]]
        self.value = 1.0

    # Instantiate portfolio vector memory with initial values
    def initPvm(self, rates):
        self.pvm = [[1.] + [0. for i in self.symbols[1:]] for j in (rates + rates[:1])]

    # Determine change in weights and portfolio value due to price movement between trading periods
    def updateRateShift(self, prevRates, curRates): 
        xHat = np.divide([1.] + curRates, [1.] + prevRates)
        values = [self.getValue() * w * x for w, x in zip(self.getWeights(), xHat)]

        prevValue = self.getValue()
        self.setValue(sum(values))

        b = np.divide(values, self.getValue())
        prevWeights = self.getWeights()
        
        self.setWeights(b)
        return b, prevWeights, self.getValue()

    # Sample the start index of a training minibatch from a geometric distribution
    def getMinibatchInterval(self, i):
        k = geometric(self.pBeta)
        tB = np.clip(-k - self.minibatchSize + i + 2, 1, i - self.minibatchSize + 1)
        return tB

    # Ascend reward gradient of minibatch starting at idx
    def trainOnMinibatch(self, idx, inTensor, rates):
        pvmSeg = self.pvm[idx:(idx + self.minibatchSize)]  # TODO : might lack paranthesis

        truncPvmSeg = [q[1:] for q in pvmSeg]  # TODO : old: [q[1:] for q in pvmSeg]
        print('truncated pvm segment: {}'.format(truncPvmSeg))
        
        mIn = np.array(inTensor[idx:(idx + self.minibatchSize)])
        wIn = np.array(truncPvmSeg)
        bIn = np.array([[1.0]] * self.minibatchSize)
        print('minibatch: main input: {}'.format(mIn.shape))  # seems to have proper shape
        print('weight input as truncPvmSeg: {}'.format(wIn))
        
        # TODO : are the inputs MALFORMED now?
        out = self.model.predict([mIn, wIn, bIn], batch_size=self.minibatchSize) 
        squeezeOut = np.squeeze(out)

        pP = [[1.] + list(r) for r in rates[(idx-1):(idx+self.minibatchSize-1)]]
        pC = [[1.] + list(r) for r in rates[(idx):(idx+self.minibatchSize)]] 
        pN = [[1.] + list(r) for r in rates[(idx+1):(idx+self.minibatchSize+1)]] 

        # Previous and current market relative price matrices
        yP = np.divide(pC, pP)
        yC = np.divide(pN, pC)    
        
        wPrev = np.array(self.pvm[idx:idx + self.minibatchSize])
        
        wpNum = np.multiply(yP, wPrev)
        wpDen = np.array([np.dot(ypT, wpT) for (ypT, wpT) in zip(yP, wPrev)])
        wPrime = [np.divide(n, d) for (n, d) in zip(wpNum, wpDen)]

        mu = [[self.calculateMu(wPT, wT, self.k)] for (wPT, wT) in zip(wPrime, squeezeOut)]
        
        grad = self.getGradient(inputs=[mIn, wIn, bIn, mu, yC, out])  

        updates = [self.learningRate * g for g in grad]
        
        modelWeights = self.model.get_weights()
        updateWeights = [np.add(wT, uT) for (wT, uT) in zip(modelWeights, updates)]
        self.model.set_weights(updateWeights)
        print('MINIBATCH: updated weights: {}'.format(self.model.get_weights()))

    # RL agent training function
    def train(self, inTensor, rates):
        # print('data: {}'.format(inTensor))
        self.initPvm(rates)
        # For each epoch
        for epoch in range(self.epochs):
            print('\nBeginning epoch ' + str(epoch))
            self.reset()
            # For each trading period in the interval
            for i, (r, p, x) in enumerate(zip(rates[1:], self.pvm[1:-1], inTensor[1:])):
                # Determine eiie output at the current period
                mIn = np.array([x])
                wIn = np.array([np.squeeze(p)])  # TODO : np.squeeze necessary?
                bIn = np.array([1.])
                # print(p)  # size only 1, but should be 4...
                # print(wIn)
                modelOutput = self.model.predict([mIn, wIn, bIn])[0]
                print('model output: {}'.format(modelOutput))

                # Overwrite pvm at subsequent period
                self.pvm[i + 2] = modelOutput[0][0]
                
                # Update portfolio for current timestep
                newB, prevB, curValue = self.updateRateShift(rates[i], r)
                # print(newB)
                # print(curValue)
                self.updatePortfolio(modelOutput[0][0], newB, curValue, rates[i], r) 
                if i % 1000 == 0:
                    print('\ti (' + str(i) + ') value: ' + str(self.getValue()))
                 
                # Train EIIE over minibatches of historical data
                # if i - self.minibatchSize >= 0:
                if i < self.minibatchSize:
                    for j in range(self.minibatchCount):
                        # Sample minibatch interval from geometric distribution
                        idx = self.getMinibatchInterval(i)
                        self.trainOnMinibatch(idx, inTensor, rates)
            print('Epoch ' + str(epoch) + ' value: ' + str(self.getValue()))
    
    # Calculate current portfolio value and set portfolio weights
    def updatePortfolio(self, newWeights, prevWeights, prevValue, prevRates, curRates):
        # Calculate current portfolio value
        rateRatios = list(np.divide(curRates, prevRates)) # [1.] + list(np.divide(curRates, prevRates))
        prevValues = np.multiply(prevWeights, prevValue)
        print('rate ratios: {}'.format(rateRatios))
        print('prev values: {}'.format(prevValues))
        currentValues = np.multiply(rateRatios, prevValues)
        currentVal = sum(currentValues)
        
        # Calculate difference between current and new weights
        weightDelta = np.subtract(newWeights, self.weights)
        print('weight difference: {}'.format(weightDelta))

        valueDelta = [(self.value * delta) for delta in weightDelta]

        # Calculate BTC or USDT being bought
        buy = self.tradePer * -sum([v if (v < 0) else 0 for v in valueDelta])
        print('buy: {}'.format(buy))

        posValDeltas = {}
        for i, v in enumerate(valueDelta):
            if v > 0:
                posValDeltas[i] = v

        posValDeltaSum = sum(posValDeltas.values())
        posValDeltaPer = np.divide(list(posValDeltas.values()), posValDeltaSum)

        # Calculate actual positive value changes with trade fees
        realPosValDeltas = [per * self.tradePer * buy for per in posValDeltaPer]

        # Calculate overall value deltas
        realValDeltas = []
        for val in valueDelta:
            if val <= 0:
                realValDeltas.append(val)
            else:
                realValDeltas.append(realPosValDeltas.pop(0))

        # TODO : perhaps remove trade failure simulation as there shouldnt be such a possibility with enough liquidity
        # Simulate possiblility of trade failure
        # for i in range(1, len(realValDeltas)):
        #     if random.random() < self.failureChance:
        #         realValDeltas[0] += realValDeltas[i] / self.tradePer    
        #         realValDeltas[i] = 0
        

        # Calculate new value
        newValues = np.add(currentValues, realValDeltas)
        newValue = sum(newValues)
        self.setValue(newValue)
    
        self.setWeights(np.divide(newValues, newValue))


    # Iteratively calculate the transaction remainder factor for the period
    def calculateMu(self, wPrime, w, k):
        # Calculate initial mu value
        mu = self.tradeFee * sum([abs(wpI - wI) for wpI, wI in zip(wPrime, w)])     

        # Calculate iteration of mu
        for i in range(k):
            muSuffix = sum([max((wpI - mu * wI), 0) for (wpI, wI) in zip(wPrime, w)])
            mu = (1. / (1. - self.tradeFee * w[0])) * (1. - (self.tradeFee * wPrime[0]) - (2 * self.tradeFee - (self.tradeFee ** 2)) * muSuffix)
        return mu

    def getValue(self):
        return self.value

    def getWeights(self):
        return self.weights[:]

    def getValues(self):
        return self.values

    def setValue(self, value):
        self.value = value

    # Assign new portfolio weights
    def setWeights(self, weights):
        self.weights = weights[:]

if __name__ == '__main__':
    # The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW
    # channels_first seems to only make sense when using GPU which makes it a lot faster
    # TODO : figure out proper CPU shape and option to turn to GPU shape
    # GPU : timesteps x channels x height x width = timesteps x CloseHighLow x #markets x lengthOfWindow or timesteps x CloseHighLow x lengthOfWindow x #markets 
    # CPU : timesteps x height x width x channels = timesteps x CloseHighLow x #markets x lengthOfWindow
    K.set_image_data_format('channels_last')
    holdBTC = False  # TODO : be sure about the purpose of this variable...
    holdUSDT = True
    startRange = datetime.datetime(2022,6,20,0,0,0)
    endRange = datetime.datetime(2022,6,21,0,0,0)
    
    # TODO(investigate) : what about BTC markets as currency?
    # markets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'MATICUSDT', 'DOTUSDT', 'LINKUSDT',
    #            'BNBUSDT', 'SOLDUSDT', 'AVAXUSDT', 'ATOMUSDT', 'XRPUSDT',
    #            # less established ones
    #            'GMTUSDT', 'PAXGUSDT']
    markets = ['BTCUSDT_15m', 'ETHUSDT_15m', 'ADAUSDT_15m', 'BNBUSDT_15m']
    # market = 'BTCUSDT_15m'
    
    # TODO : give these vars better names
    b = [1.] + [0.] * (len(markets) - 1)  
    pBeta = 0.00005
    k = 15
    learningRate = 0.00019
    minibatchCount = 30
    minibatchSize = 50
    epochs = 50
    window = 50
    
    data = []  # final shape: timesteps x features x markets, features = (close, high, low, ...)
    priceRelativeVectors = []
    rates = []
    for market in markets:
        rawData = getData(startRange, endRange, market)
        formattedData, priceRelativeVector, ratesMarket = formatDataForInput(rawData, window)
        data.append(formattedData)
        priceRelativeVectors.append(priceRelativeVector)
        rates.append(ratesMarket)
    
    # get them into the right shape
    data = np.swapaxes(np.swapaxes(np.swapaxes(data, 2, 3), 0, 1), 1, 2)  # swap markets to the end
    # data = np.swapaxes(data, 1, 2)  # swap features with lookback, s.t. rows=lookback, cols=features    
    priceRelativeVectors = np.transpose(priceRelativeVectors).tolist()
    rates = np.transpose(rates).tolist()
    
    
    portfolio = Portfolio(markets, b, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs)
    portfolio.createEIIECNN(data, priceRelativeVectors)
    portfolio.train(data, rates)
