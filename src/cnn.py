# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 03:49:14 2022

@author: slaven
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.models import Models
from tensorflow.keras.layers import Conv2D, Conv3D, Input, Dropout, BatchNormalization,\
    Activation, Softmax, Lambda, Concatenate


def expandDimensions(weights):
    expandedWeights = K.expand_dims(weights, axis=1)
    expandedWeights = K.expand_dims(expandedWeights, axis=1)
    return expandedWeights
    

if __name__ == '__main__':
    # TODO : copy CNN architecture from the paper
    # TODO(check) : they have a btc_bias var, which is their cash. what to do with it?
    # !INFO : keras.layers.Input is not necessary, bc it only removes input var of the next layer
    # TODO(check) : these layers have a batch_size arg, does this help or do i need more fine grained control? 
    # TODO(check) : functional API vs Sequential() API
    
    # define variables
    features = 3
    filterNumber = 3
    inputTensor = 0
    rates = 0
    
    mainInputShape = np.array(inputTensor).shape[1:]
    weightInputShape = np.array(rates).shape[1:]
    biasInputShape = (1, )
    
    # create EIIE CNN with the functional API due to its flexibility: https://www.tensorflow.org/guide/keras/functional
    # define the basic input tensor shape in a separate layer
    inputLayer = Input(shape=mainInputShape)
    base = Conv2D(filters=2, kernel_size=(3, 1))(inputLayer)
    base = Activation('relu')(base)
    base = Conv2D(filters=20, kernel_size=(48, 1))(base)
    base = Activation('relu')(base)
    
    # create layer for portfolio weights
    weightsInput = Input(shape=weightInputShape)
    weightsExpanded = Lambda(expandDimensions)(weightsInpit)  # layer that acts like a lambda function
    
    # concatenate base layer with portfolio weights layer
    base = Concatenate(axis=1)([base, weightsExpanded])
    base = Conv2D(1, (1, 1))(base)
    
    # create bias layer
    biasInput = Input(shape=biasInputShape)
    biasExpanded = Lambda(expandDimensions)(biasInput)
    
    base = Concatenate(axis=3)([base, biasExpanded])
    baseOut = Activation('softmax')(base)
    
    model = Model([inputLayer, weightsInput, biasInput], baseOut)
    
    # Instantiate custom symbolic gradient
    mu = K.placeholder(shape=(None, 1), name='mu')
    y = K.placeholder(shape=(None, len(self.symbols)), name='y')
    
    # tf.keras.backend.squeeze removes 1-dim (opposite of expand_dims) to fit the shape
    sqOut = K.squeeze(K.squeeze(self.model.output, 1), 1)
    yOutMult = tf.multiply(sqOut, y)
    
    yOutBatchDot = tf.reduce_sum(yOutMult, axis=1, keep_dims=True)
    muDotMult = tf.multiply(mu, yOutBatchDot)
    
    loss = -K.log(muDotMult)
    grad = K.gradients(loss, model.trainable_weights)
    
    # TODO : function has to be placed somewhere
    getGradient = K.function(inputs=[mIn, wIn, bIn, mu, y, self.model.output], outputs=grad) 
    