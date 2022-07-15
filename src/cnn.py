# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 03:49:14 2022

@author: slaven
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from numpy.random import geometric
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dropout, BatchNormalization,\
    Activation, Softmax, Lambda, Concatenate, InputLayer, Permute

from utils.utils import getData, formatDataForInput


def expandDimensions(weights):
    expandedWeights = tf.expand_dims(weights, axis=1)
    expandedWeights = tf.expand_dims(expandedWeights, axis=1)
    return expandedWeights

# Simulated crypto portfolio
class Portfolio:
    def __init__(self, symbols, weights, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs, noop=False, failureChance=0):
        self.symbols = symbols
        self.setWeights(weights)  # how are these weights different from the PVM?
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
        self.noop = noop  # TODO : what is this?
        self.failureChance = failureChance  # TODO : needed?
    
    """
    :param inputTensor, data with shape TODO
    :param rates, TODO
    """
    def createEIIECNN(self, inputTensor, rates):
        # TODO : enable GPU support
        # TODO(check) : they have a btc_bias var, which is their cash. what to do with it? i might need a usdt_bias var or something
        # TODO(check) : these layers have a batch_size arg, does this help or do i need more fine grained control? 
        
        # define variables        
        mainInputShape = np.array(inputTensor).shape[1:]
        weightInputShape = np.array(rates).shape[1:]
        biasInputShape = (1, )
        print('main input shape: {}, weight input shape: {}'.format(mainInputShape, weightInputShape))
        
        # TODO : GPU mode will be with if-stmt to create a new neural network
        # create EIIE CNN with the functional API because of its flexibility: https://www.tensorflow.org/guide/keras/functional
        # define the basic input tensor shape in a separate layer
        inputLayer = Input(shape=mainInputShape, name='main_input_layer')
        # print('After input     : {}'.format(inputLayer))
        base = Conv2D(filters=2, kernel_size=(1, 3), name='first_conv')(inputLayer)
        # print('After first conv: {}'.format(tf.shape(base)))
        base = Activation('relu')(base)
        # print('After first relu: {}'.format(tf.shape(base)))
        base = Conv2D(filters=20, kernel_size=(1, 48), name='second_conv')(base)
        # print('After second conv: {}'.format(tf.shape(base)))
        base = Activation('relu')(base)
        # print('After second relu: {}'.format(tf.shape(base)))
        
        # create layer for portfolio weights
        weightsInput = Input(shape=weightInputShape, name='weights_input_layer')
        weightsExpanded = Lambda(expandDimensions, name='weights_expansion_layer')(weightsInput)  # layer that acts like a lambda function
        # print('After weights input: {}'.format(tf.shape(weightsExpanded)))
        # CPU mode: None x 1 x 1 x 4 to None x 4 x 1 x 1
        weightsExpanded = tf.transpose(weightsExpanded, [0,3,2,1])
        # concatenate base layer with portfolio weights layer
        # TODO : CPU: axis=3, GPU: axis=1
        base = Concatenate(axis=3, name='weights_concatenation_layer')([base, weightsExpanded])
        base = Conv2D(1, (1, 1))(base)
        
        # create bias layer
        biasInput = Input(shape=biasInputShape, name='bias_input_layer')
        biasExpanded = Lambda(expandDimensions, name='bias_expansion_layer')(biasInput)
        # print('After bias expanded: {}'.format(tf.shape(biasExpanded)))
        
        # TODO : CPU axis=1, GPU axis=3
        base = Concatenate(axis=1, name='bias_concatenation_layer')([base, biasExpanded])
        baseOut = Activation('softmax', name='output_layer')(base)
        baseOut = tf.squeeze(tf.squeeze(baseOut, 2), 2)  # bring it into the right shape
        # baseOut = tf.transpose(baseOut, [0,2,1,3])
        print('Final output: {}'.format(baseOut))
        
        self.model = Model([inputLayer, weightsInput, biasInput], baseOut)
        print('final self.model: {}'.format(self.model.output))
        
        # self.model.add_loss()  # TODO : do this?, put the tape part or so in its own function?
        
        ### INSTANTIATE CUSTOM SYMBOLIC GRADIENT - "Symbolic" gradient does not work, as I do EAGER mode not "graph" mode...
        ### TODO : maybe dont use these things and no symbolic gradient but "more direct"
        ### TODO : mu, y and potentially out/self.model.output should be given directly as inputs somehow and not instantiated as a placeholder var or so...
        # mu = Input(shape=(1,), name='mu')
        # y = Input(shape=(len(self.symbols)+1,), name='y')
        # out = Input(shape=(len(self.symbols)+1,), name='out')
        # print('y with Input: {}'.format(y))
        
        # print('\nExecuting eagerly outside of tape: {}'.format(tf.executing_eagerly()))
        # # TODO : use gradient tape for TF2 - requires deep inspection in how the loss can be calculated properly!
        # with tf.GradientTape() as tape:
        #     print('Executing eagerly INSIDE of tape: {}'.format(tf.executing_eagerly()))
        #     print('y input: {}'.format(y))
        #     print('out input: {}'.format(out))
        #     print('self.model.output input: {}'.format(self.model.output))
        #     yOutMult = tf.multiply(self.model.output, y)  # TODO : or use out?
        #     print('yOutMult: {}'.format(yOutMult))
        #     yOutBatchDot = tf.reduce_sum(yOutMult, axis=0, keepdims=True)
        #     print('yOutBatchDot:  {}'.format(yOutBatchDot))
        #     muDotMult = tf.multiply(mu, yOutBatchDot)
        #     print('muDotMult: {}'.format(muDotMult))
        #     loss = -tf.math.log(muDotMult)  # minus sign needed for gradient ASCENT as descent is default for neural networks
        #     print('\nloss after the tapeing block: {}'.format(loss))
            
        
        # tf.keras.backend.squeeze removes 1-dim (opposite of expand_dims) to fit the shape
        # GPU mode
        # sqOut = K.squeeze(K.squeeze(self.model.output, 1), 1)
        # CPU mode
        # print('model output: {}'.format(self.model.output))
        # sqOut = K.squeeze(K.squeeze(K.squeeze(self.model.output, 2), 2), 0)
        # sqOut = self.model.output
        # sqOut = K.squeeze(self.model.output, 0)
        #
        # sqOut = out
        # print('sqOut: {}'.format(sqOut))
        # yOutMult = tf.multiply(sqOut, y)
        # print('yOutMult: {}'.format(yOutMult))        
        
        # yOutBatchDot = tf.reduce_sum(yOutMult, axis=0, keepdims=True)
        # print('yOutBatchDot: {}'.format(yOutBatchDot))
        # muDotMult = tf.multiply(mu, yOutBatchDot)
        # print('muDotMult: {}'.format(muDotMult))
        
        # loss = -tf.math.log(muDotMult)  # minus sign needed for gradient ASCENT as descent is default for neural networks
        # print('\nloss after the tapeing block: {}'.format(loss))
        # print('model trainable weights: {}'.format(self.model.trainable_weights))
        # grad = tape.gradient(loss, [self.model.trainable_weights])
        
        # # I can probably just do: self.getGradient = grad
        # self.getGradient = tf.function(inputs=[inputLayer, weightsInput, biasInput, mu, y, self.model.output], outputs=grad) 
    
    
    # TODO : give better naming
    # define custom loss for gradient ASCENT
    # TODO : should all be TENSORS
    """
    TODO : extend description
    :param y, the priceRelativeVectors (batch_size x assets)
    :param mu, transaction remainder factor (? x ?)
    :param out, TODO : the weights? (batch_size x assets)
    """
    def custom_loss(self, y, mu, out):
        # print('mu input: {}'.format(mu))
        # print('y input: {}'.format(y))
        # print('out input: {}'.format(out))  # TODO : self.model.output is this supposed to be out simply?
        yOutMult = tf.multiply(y, out)
        # print('yOutMult: {}'.format(yOutMult))
        yOutBatchDot = tf.reduce_sum(yOutMult, axis=0, keepdims=True)
        # print('yOutBatchDot:  {}'.format(yOutBatchDot))
        muDotMult = tf.multiply(mu, yOutBatchDot)  # p11 formula 21
        # print('muDotMult: {}'.format(muDotMult))
        loss = -tf.math.log(muDotMult)  # minus sign needed for gradient ASCENT as descent is default for neural networks
        # print('\nloss aftäer the tapeing block: {}'.format(loss))
        return loss
    
    # Re-initialize portfolio state
    def reset(self):
        # TODO : probably assumed that the first symbol is cash simply
        self.weights = [1.] + [0. for i in self.symbols[1:]]  # TODO : is this correct? cash bias 1, rest 0, so all is in cash? I think for these weights it might be OK
        self.value = 1.0

    # Instantiate portfolio vector memory with initial values
    def initPvm(self, rates):  # TODO : init with the cash bias?
        # EQUATION 5: w_0 is initialized as [1,0,...,0]
        self.pvm = [[1.] + [0. for i in self.symbols[1:]] for j in (rates + rates[:1])]  # TODO : is this correct?

    # Determine change in weights and portfolio value due to price movement between trading periods
    def updateRateShift(self, prevRates, curRates):
        print('\nENTERED RATESHIFT')
        print('prevRates: {}'.format(prevRates))
        print('currRates: {}'.format(curRates))
        xHat = np.divide([1.] + curRates, [1.] + prevRates)  # TODO : clearly understand where the [1.] comes from and if its even necessary (probably yes)
        values = [self.getValue() * w * x for w, x in zip(self.getWeights(), xHat)]
        print('self.getWeights(): {}'.format(self.getWeights()))
        print('self.getValue(): {}'.format(self.getValue()))
        print('xHat: {}'.format(xHat))
        print('values: {}'.format(values))
        
        prevValue = self.getValue()  # TODO : why is this unused
        self.setValue(sum(values))
        print('sum values: {}'.format(sum(values)))

        b = np.divide(values, self.getValue())
        prevWeights = self.getWeights()
        print('b = np.divide(values, self.getValue()): {}'.format(b))
        print('prevWeights: {}'.format(prevWeights))
        
        self.setWeights(b)
        return b, prevWeights, self.getValue()

    # Sample the start index of a training minibatch from a geometric distribution
    def getMinibatchInterval(self, i):
        k = geometric(self.pBeta)
        tB = np.clip(-k - self.minibatchSize + i + 2, 1, i - self.minibatchSize + 1)
        return tB

    # Ascend reward gradient of minibatch starting at idx
    def trainOnMinibatch(self, idx, inTensor, rates):
        # TODO : why is index negative, positive would be nicer? causes problems at pvmSeg
        print('\nENTER MINIBATCH ({})'.format(idx))
        # only require the relevant subset based on the minibatchSize of the full PVMs at each timestep (idx)
        pvmSeg = self.pvm[idx:(idx+self.minibatchSize)]  # TODO(THIS) : something is a miss when truncating...
        # TODO : something is going wrong when adding the cash bias array, i think it is on its own row....
        # print('IN minibatch: segm. pvm: {}'.format(pvmSeg))

        # TODO(FIX) : starting at pvmSeg[1:], bc elem at idx 0 has 1 size less
        truncPvmSeg = [pvmSeg[0]] + [q[1:] for q in pvmSeg[1:]]  # TODO : q[1:] is supposed to remove the cash weight in the PVM, this should be OK
        # print('truncated pvm segment: {}'.format(truncPvmSeg))
        
        mIn = np.array(inTensor[idx:(idx+self.minibatchSize)])
        wIn = np.array(truncPvmSeg)  # TODO : its a long array containing lists with weights excl. the cash weight
        bIn = np.array([[1.0]] * self.minibatchSize)  # TODO : cash bias? why [1.0]? array containing a list with only 1 element
        print('minibatch: main input: {}'.format(mIn.shape))  # seems to have proper shape
        print('weight input as truncPvmSeg: {}'.format(wIn.shape))
        
        # TODO : are the inputs MALFORMED now?
        out = self.model.predict([mIn, wIn, bIn], batch_size=self.minibatchSize)
        # print('out: {}'.format(out))
        squeezeOut = np.squeeze(out)
        print('squeezeOut: {}'.format(squeezeOut.shape))

        # TODO : what vars are these exactly?
        pP = [[1.] + list(r) for r in rates[(idx-1):(idx+self.minibatchSize-1)]]  # prev t-1
        pC = [[1.] + list(r) for r in rates[(idx):(idx+self.minibatchSize)]]      # curr t
        pN = [[1.] + list(r) for r in rates[(idx+1):(idx+self.minibatchSize+1)]]  # next t+1

        # Previous and current market relative price matrices
        yP = np.divide(pC, pP)
        print('yP: {}'.format(np.shape(yP)))
        yC = np.divide(pN, pC)    
        
        # long matrix with weights up to the current time period
        wPrev = self.pvm[idx:(idx+self.minibatchSize)]  # TODO(FIX) : first elem seems to make problems (only len 4 instead of 5 - it's the one which was init'ed)
        # fix first entry manually
        wPrev[0].append(0.)
        print('wPrev shape: {}'.format(np.shape(wPrev)))
        print('wPrev: \n{}'.format(wPrev))
        
        wpNum = np.multiply(yP, wPrev)
        wpDen = np.array([np.dot(ypT, wpT) for (ypT, wpT) in zip(yP, wPrev)])
        wPrime = [np.divide(n, d) for (n, d) in zip(wpNum, wpDen)]
        print('wpNum  : \n{}'.format(wpNum.shape))
        print('wpDen  : \n{}'.format(wpDen.shape))
        print('wPrime : \n{}'.format(np.shape(wPrime)))

        mu = [[self.calculateMu(wPT, wT, self.k)] for (wPT, wT) in zip(wPrime, squeezeOut)]
        print('mu: \n{}'.format(np.shape(mu)))
        print('yC: \n{}'.format(np.shape(yC)))
        print('out: \n{}'.format(np.shape(out)))
        
        mu = tf.Variable(mu, dtype=tf.float32, name='mu Variable')
        yC = tf.Variable(yC, dtype=tf.float32, name='yC variable')
        out = tf.Variable(out, dtype=tf.float32, name='out variable')
        # TODO : use custom loss
        with tf.GradientTape() as tape:
            loss = tf.Variable(self.custom_loss(yC, mu, out))
        
        print('watched variables: {}'.format(tape.watched_variables()))
        print('loss after casting to tf.Variable: {}'.format(loss))
        print('trainable weights: {}'.format(self.model.trainable_weights))
        
        # TODO(!!!) : which weights do i actually need here for the gradient calculation?
        grad = tape.gradient(loss, self.model.trainable_weights)
        # grad = self.getGradient(inputs=[mIn, wIn, bIn, mu, yC, out])
        print('minibatch GRAD: {}'.format(grad))

        updates = [self.learningRate * g for g in grad]
        print('Updates: {}'.format(updates))
        
        modelWeights = self.model.get_weights()
        updateWeights = [np.add(wT, uT) for (wT, uT) in zip(modelWeights, updates)]
        self.model.set_weights(updateWeights)
        print('MINIBATCH: updated model weights: {}'.format(self.model.get_weights()))

    # RL agent training function
    def train(self, inTensor, rates):
        self.initPvm(rates)
        # For each epoch
        for epoch in range(self.epochs):
            print('\nBeginning epoch ' + str(epoch))
            self.reset()
            # For each trading period in the interval
            for i, (r, p, x) in enumerate(zip(rates[1:], self.pvm[1:-1], inTensor[1:])):  # TODO : why is the last pvm ignored? what about the inTensor?
                # Determine eiie output at the current period
                print('\nCURRENT ITERATION: {}'.format(i))
                print('current p: {}'.format(p))
                print('current pvm: {}'.format(self.pvm[i+1]))
                
                mIn = np.array([x])
                wIn = np.array([np.squeeze(self.pvm[i+1])])  # TODO : np.squeeze necessary?
                bIn = np.array([1.])
                print('pvm in train func: {}'.format(self.pvm[i+1]))  # TODO : size only 1, but should be 4...
                print('wIn in train func: {}'.format(wIn))
                print('wIn shape in train func: {}'.format(wIn.shape))
                modelOutput = self.model.predict([mIn, wIn, bIn])[0]
                print('model output: {}'.format(modelOutput))
                # print('model output[0]: {}'.format(modelOutput[0]))
                
                # Overwrite pvm at subsequent period
                # TODO : am i missing the cash bias to have length 5? length 4? without cash bias?
                # self.pvm is not updated in the loop from above like this - it retains the init'ed values...
                self.pvm[i+2] = modelOutput.tolist()[1:]  # TODO : correct to start at idx 1?
                print('self.pvm[i+2]: {}'.format(self.pvm[i+2]))
                
                # Update portfolio for current timestep
                newB, prevB, curValue = self.updateRateShift(rates[i], r)
                print('newB: {}'.format(newB))
                print('curValue: {}'.format(curValue))
                
                self.updatePortfolio(modelOutput.tolist(), newB, curValue, rates[i], r)
                if i % 1000 == 0:
                    print('\ti (' + str(i) + ') value: ' + str(self.getValue()))
                 
                # Train EIIE over minibatches of historical data
                # "i" should be bigger than the minibatchSize for the lookback to work properly
                # if i - self.minibatchSize >= 0:
                # # if i < self.minibatchSize:
                #     for j in range(self.minibatchCount):
                        
                #         # Sample minibatch interval from geometric distribution
                #         idx = self.getMinibatchInterval(i)
                #         self.trainOnMinibatch(idx, inTensor, rates)
            print('Epoch ' + str(epoch) + ' value: ' + str(self.getValue()))
            
            if epoch == 6: break
    
    # Calculate current portfolio value and set portfolio weights
    def updatePortfolio(self, newWeights, prevWeights, prevValue, prevRates, curRates):
        # Calculate current portfolio value
        print('\nENTERED UPDATE PORTFOLIO')
        rateRatios = list(np.divide(curRates, prevRates)) # [1.] + list(np.divide(curRates, prevRates))
        prevValues = np.multiply(prevWeights, prevValue)
        print('rate ratios: {}'.format(rateRatios))
        print('prev values: {}'.format(prevValues))
        currentValues = np.multiply(rateRatios, prevValues)
        currentVal = sum(currentValues)
        
        # Calculate difference between current and new weights 
        # TODO : np.subtract(newWeights[1:], self.weights) ??
        weightDelta = np.subtract(self.weights, newWeights[1:])  # TODO : start from 1?
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
        print('newValues: {}'.format(newValues))
        print('newValue : {}'.format(newValue))
        print('np.divide(newValues, newValue): {}'.format(np.divide(newValues, newValue)))
    
        self.setWeights(np.divide(newValues, newValue))  # what kind of weights are these exactly?
        print('newly set weights: {}'.format(self.getWeights()))


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
    # tf.compat.v1.enable_eager_execution()
    # The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW
    # channels_first seems to only make sense when using GPU which makes it a lot faster
    # TODO : figure out proper CPU shape and option to turn to GPU shape
    # GPU : timesteps x channels x height x width = timesteps x CloseHighLow x #markets x lengthOfWindow or timesteps x CloseHighLow x lengthOfWindow x #markets 
    # CPU : timesteps x height x width x channels = timesteps x CloseHighLow x #markets x lengthOfWindow
    K.set_image_data_format('channels_last')
    holdBTC = False  # TODO : be sure about the purpose of this variable...
    holdUSDT = True
    startRange = datetime.datetime(2022,6,19,0,0,0)
    endRange = datetime.datetime(2022,6,21,0,0,0)
    
    # markets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'MATICUSDT', 'DOTUSDT', 'LINKUSDT',
    #            'BNBUSDT', 'SOLDUSDT', 'AVAXUSDT', 'ATOMUSDT', 'XRPUSDT',
    #            # less established ones
    #            'GMTUSDT', 'PAXGUSDT']
    markets = ['BTCUSDT_15m', 'ETHUSDT_15m', 'ADAUSDT_15m', 'BNBUSDT_15m']
    # market = 'BTCUSDT_15m'
    
    # TODO : give these vars better names
    b = [1.] + [0.] * (len(markets) - 1)  # TODO : -1 correct here?
    pBeta = 0.00005
    k = 15
    learningRate = 0.00019
    minibatchCount = 30
    minibatchSize = 50
    epochs = 500
    window = 50  # TODO : when using a bigger window, i need to adapt the kernel size or add a new conv layer with a bigger kernel size s.t. shapes are correct again
    
    data = []  # final shape: timesteps x features x markets, features = channels = (close, high, low, ...)
    priceRelativeVectors = []
    rates = []
    for market in markets:
        rawData = getData(startRange, endRange, market)
        formattedData, priceRelativeVector, ratesMarket = formatDataForInput(rawData, window)
        data.append(formattedData)
        priceRelativeVectors.append(priceRelativeVector)
        rates.append(ratesMarket)
    
    # get them into the right shape
    data = np.swapaxes(np.swapaxes(data, 2, 3), 0, 1)
    # data = np.swapaxes(np.swapaxes(np.swapaxes(data, 2, 3), 0, 1), 1, 2)  # swap markets to the end
    # data = np.swapaxes(data, 1, 2)  # swap features with lookback, s.t. rows=lookback, cols=features    
    priceRelativeVectors = np.transpose(priceRelativeVectors).tolist()
    rates = np.transpose(rates).tolist()
    
    
    portfolio = Portfolio(markets, b, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs)
    portfolio.createEIIECNN(data, priceRelativeVectors)
    portfolio.train(data, rates)
