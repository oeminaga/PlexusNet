"""
Copyright by Okyaz Eminaga. 2020
"""
import random
import os
from keras import layers, models
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.constraints import min_max_norm
from keras import regularizers, initializers
from keras.regularizers import l2
import utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D,Activation, Dropout, Dense, GlobalMaxPooling2D

def ColorIntensityNormalisationSection(x):
    """
    x : tensor
    """
    x_y_o = utils.RotationThetaWeightLayer()([x,x])
    x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
    return x_y_o

def Conv2DBNSLU(x, filters=2, kernel_size=1, strides=1, padding='same', activation="relu", name="", activity_regularizer=None, kernel_regularizer=None, use_bias=False, scale=False):
    '''
    x : tensor
    '''
    x = layers.Conv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer)(x)
    x = layers.BatchNormalization(scale=False)(x)
    if activation not in [None, ""]:
        x = layers.Activation(activation)(x)
    return x
        
def SepConv2DBNSLU(x, filters=2, kernel_size=1, strides=1, padding='same', activation=None, name=None, scale=True, BN=True):
    '''
    x : tensor
    '''
    x = layers.SeparableConv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=False)(x)
    if BN:
        x = layers.BatchNormalization(scale=scale)(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block,kernel_initializer="he_normal", seed=0):
    """
    The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer=initializers.he_normal(seed=seed+1)
                      )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer=initializers.he_normal(seed=seed+2)
                      )(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer=initializers.he_normal(seed=seed+3))(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = layers.LeakyReLU()(x)
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),seed=0):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer=initializers.he_uniform(seed=seed+8)
                      )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer=initializers.he_uniform(seed=seed+8))(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer=initializers.he_uniform(seed=seed+8))(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer=initializers.he_uniform(seed=seed+8))(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.LeakyReLU()(x)
    return x