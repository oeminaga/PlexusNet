"""
Copyright by Okyaz Eminaga. 2020
"""
import math, os, re, warnings, random, time
from tensorflow.keras import layers, models,optimizers
from tensorflow.keras.layers import LeakyReLU
import keras.backend as K
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.regularizers import l2
import random
import os
from . import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D,Activation, Dropout, Dense, GlobalMaxPooling2D
from .functions import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

# Mixed precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')

HEIGHT = 512
WIDTH = 512
CHANNELS = 3
AUTO = tf.data.experimental.AUTOTUNE

def transform_rotation(image, height, rotation):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated
    DIM = height
    XDIM = DIM%2 #fix for size 331
    
    rotation = rotation * tf.random.uniform([1],dtype='float32')
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3])

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(rotation_matrix,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

def transform_shear(image, height, shear):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly sheared
    DIM = height
    XDIM = DIM%2 #fix for size 331
    
    shear = shear * tf.random.uniform([1],dtype='float32')
    shear = math.pi * shear / 180.
        
    # SHEAR MATRIX
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3])    

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(shear_matrix,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

# CutOut
def data_augment_cutout(image, min_mask_size=(int(HEIGHT * .1), int(HEIGHT * .1)), 
                        max_mask_size=(int(HEIGHT * .125), int(HEIGHT * .125))):
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if p_cutout > .85: # 10~15 cut outs
        n_cutout = tf.random.uniform([], 10, 15, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .6: # 5~10 cut outs
        n_cutout = tf.random.uniform([], 5, 10, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .25: # 2~5 cut outs
        n_cutout = tf.random.uniform([], 2, 5, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    else: # 1 cut out
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=1)

    return image

def random_cutout(image, height, width, channels=3, min_mask_size=(10, 10), max_mask_size=(80, 80), k=1):
    assert height > min_mask_size[0]
    assert width > min_mask_size[1]
    assert height > max_mask_size[0]
    assert width > max_mask_size[1]

    for i in range(k):
      mask_height = tf.random.uniform(shape=[], minval=min_mask_size[0], maxval=max_mask_size[0], dtype=tf.int32)
      mask_width = tf.random.uniform(shape=[], minval=min_mask_size[1], maxval=max_mask_size[1], dtype=tf.int32)

      pad_h = height - mask_height
      pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
      pad_bottom = pad_h - pad_top

      pad_w = width - mask_width
      pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
      pad_right = pad_w - pad_left

      cutout_area = tf.zeros(shape=[mask_height, mask_width, channels], dtype=tf.uint8)

      cutout_mask = tf.pad([cutout_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
      cutout_mask = tf.squeeze(cutout_mask, axis=0)
      image = tf.multiply(tf.cast(image, tf.float32), tf.cast(cutout_mask, tf.float32))

    return image
def data_augment(image, label):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    # Shear
    if p_shear > .2:
        if p_shear > .6:
            image = transform_shear(image, HEIGHT, shear=20.)
        else:
            image = transform_shear(image, HEIGHT, shear=-20.)
            
    # Rotation
    if p_rotation > .2:
        if p_rotation > .6:
            image = transform_rotation(image, HEIGHT, rotation=45.)
        else:
            image = transform_rotation(image, HEIGHT, rotation=-45.)
            
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > .75:
        image = tf.image.transpose(image)
        
    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3) # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2) # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1) # rotate 90º
        
    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)
        
    # Crops
    if p_crop > .6:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.5)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.6)
        elif p_crop > .7:
            image = tf.image.central_crop(image, central_fraction=.7)
        else:
            image = tf.image.central_crop(image, central_fraction=.8)
    elif p_crop > .3:
        crop_size = tf.random.uniform([], int(HEIGHT*.6), HEIGHT, dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, CHANNELS])
            
    image = tf.image.resize(image, size=[HEIGHT, WIDTH])

    if p_cutout > .5:
        image = data_augment_cutout(image)
        
    return image, label

def LoadModel(filename, custom_objects={},optimizer= optimizers.Adam(), loss="categorical_crossentropy"):
    custom_objects_internal = {'JunctionWeightLayer':utils.JunctionWeightLayer, 'RotationThetaWeightLayer': utils.RotationThetaWeightLayer, "Last_Sigmoid":Last_Sigmoid, "Mil_Attention":Mil_Attention, "RandomFourierFeatures": RandomFourierFeatures, "UnitNormLayer":UnitNormLayer, "MultiHeadSelfAttention":MultiHeadSelfAttention, "TransformerBlock":TransformerBlock}
    for  key in custom_objects:
        custom_objects_internal[key]=custom_objects[key]

    model_=load_model(filename, custom_objects=custom_objects_internal, compile=False)
    model_.compile(optimizer=optimizer, metrics=[""], loss=loss)
    return model_
Configuration={}
Configuration["num_heads"]=4
Configuration["number_of_transformer_blocks"]=1
def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''
    x = tf.keras.layers.GRU(units=256, return_sequences=False)(x)
    return x
class PlexusNet():
    def __init__(self, input_shape=(512,512), number_inputs=1,initial_filter=2, length=2, depth=7, junction=3, n_class=2, number_input_channel=3, compression_rate=0.5,final_activation="softmax", random_junctions=True, run_all_BN=True ,type_of_block="inception", run_normalization=True, run_rescale=True, filter_num_for_first_convlayer=32, kernel_size_for_first_convlayer=(5,5),stride_for_first_convlayer=2,activation_for_first_convlayer="relu", add_crop_layer=False, crop_boundary=((5,5),(5,5)), get_last_conv=False, normalize_by_factor=1.0/255.0, apply_RandomFourierFeatures=False,MIL_mode=False, MIL_CONV_mode=False, MIL_FC_percentage_of_feature=0.01, MIL_useGated=False,SCL=False,CPC=False, terms=4, predict_terms=4, code_size=256, GlobalPooling="max", RunLayerNormalizationInSCL=True, ApplyTransformer=False, number_of_transformer_blocks=1, propogate_img=False,apply_augmentation=False, lanewise_augmentation=False, ApplyLayerNormalization=False, ApplyLaneForAugmentation=[0]):
        """
        Architecture hyperparameter are:
        initial_filter (Default: 2)
        length (Default: 2)
        depth (Default: 7)
        junction (Default: 3)
        Multiple configurations are available
        You can modify or search the optimal architecture using these hyperparameters. Please be advised that the training set should be representative of different variation to build sophisticated models.

        """
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        # XLA
        tf.config.optimizer.set_jit(True)
        self.input_shape= input_shape
        HEIGHT = self.input_shape[1]
        WIDTH = self.input_shape[0]
        self.n_class = n_class
        self.lanewise_augmentation = lanewise_augmentation
        self.initial_filter=initial_filter
        self.length = length
        self.depth =depth
        self.junction = junction
        self.propogate_img=propogate_img
        self.number_input_channel=CHANNELS=number_input_channel
        self.final_activation=final_activation
        self.compression_rate = compression_rate
        self.random_junctions =random_junctions
        self.normalize_by_factor=normalize_by_factor
        self.type_of_block =type_of_block
        self.get_last_conv = get_last_conv
        self.run_all_BN =run_all_BN
        self.MIL_mode=MIL_mode
        self.ApplyLayerNormalization=ApplyLayerNormalization
        Configuration["number_of_transformer_blocks"]=number_of_transformer_blocks 
        self.RunLayerNormalizationInSCL=RunLayerNormalizationInSCL
        self.apply_augmentation=apply_augmentation
        self.MIL_FC_percentage_of_feature=MIL_FC_percentage_of_feature
        self.SCL=SCL
        self.CPC=CPC
        self.terms=terms
        self.predict_terms=predict_terms
        self.code_size=code_size
        self.GlobalPooling =GlobalPooling
        self.useGated = MIL_useGated
        self.ApplyLaneForAugmentation=ApplyLaneForAugmentation
        self.MIL_CONV_mode = MIL_CONV_mode
        self.apply_RandomFourierFeatures = apply_RandomFourierFeatures
        self.number_inputs=number_inputs
        self.ApplyTransformer = ApplyTransformer
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        if number_inputs ==1:
            x = layers.Input(shape=shape_default)
        else:
            x = []
            for k in range(number_inputs):
                x.append(layers.Input(shape=shape_default))
                
        if self.lanewise_augmentation:
            '''
            self.data_augmentation = []
            def NoAug(x):
                return x
            self.data_augmentation.append(keras.Sequential([
            layers.experimental.preprocessing.RandomRotation(0.2)]))
            
            self.data_augmentation.append(keras.Sequential([
            layers.experimental.preprocessing.RandomContrast(0.1)]))
            
            self.data_augmentation.append(NoAug)
            
            self.data_augmentation.append(keras.Sequential([
            layers.experimental.preprocessing.RandomTranslation(0.2,0.2)]))
            '''
            self.data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomFlip("vertical"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom (0.1)])


        if self.apply_augmentation:
            data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.02),
            layers.experimental.preprocessing.RandomWidth(0.2),
            layers.experimental.preprocessing.RandomHeight(0.2),])
            x_x=data_augmentation(x)
        else:
            x_x=x
        
        if propogate_img:
            x_a = layers.experimental.preprocessing.RandomRotation(0.1)(x_x)
            x_b = layers.experimental.preprocessing.RandomZoom (0.1)(x_x)
            x_c = x_x
            x_x = layers.Concatenate()([x_a,x_b,x_c])
        x_y_o = layers.Lambda(lambda x: x*(1/255))(x_x)
        if run_normalization:
            x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])
            #rescale
        if run_rescale:
            x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        if add_crop_layer:
            x_y_o = layers.Cropping2D(cropping=crop_boundary)(x_y_o)
        #Generate multiple channels from the image
        x_y = Conv2DBNSLU(x_y_o, filters= filter_num_for_first_convlayer, kernel_size=kernel_size_for_first_convlayer, strides=stride_for_first_convlayer, activation=activation_for_first_convlayer, padding='same')
        y = self.Core(x_y, initial_filter = self.initial_filter, length=self.length, depth=self.depth, number_of_junctions=self.junction, compression=self.compression_rate, type_of_block=self.type_of_block)
        if self.SCL:
            self.model = models.Model(inputs=x, outputs=y)
            self.projector_z = projector_net()
        elif self.CPC:
            K.set_learning_phase(1)
            self.encoder_model = models.Model(inputs=x, outputs=y, name='encoder')
            x_input = layers.Input((self.terms, self.input_shape[0], self.input_shape[1], self.number_input_channel))
            x_encoded = layers.TimeDistributed(self.encoder_model)(x_input)
            context = network_autoregressive(x_encoded)
            preds = network_prediction(context, self.code_size, self.predict_terms)
            y_input = layers.Input((self.predict_terms, self.input_shape[0], self.input_shape[1], self.number_input_channel))
            y_encoded = layers.TimeDistributed(self.encoder_model)(y_input)
            # Loss
            dot_product_probs = CPCLayer()([preds, y_encoded])
            # Model
            self.model = models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)
        else:
            self.model = models.Model(inputs=x, outputs=y)
    
    def _conv_block(self, x, initial_filter, reduction_channel_ratio=0.5, kernel_regularizer=None, seed=0, type_of_block="inception", stage=0, initial_image=None):
        """
        x : tensor
        initial_filter : integer
        type_of_block selection: inception,resnet,vgg,vgg_short
        """
        if self.MIL_mode:
            kernel_regularizer=l2(0.000001)
            x = layers.LayerNormalization(scale=True, center=True)(x)
            x_y = layers.Conv2D(int(round(initial_filter*1.5)), (1,16),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
            x_y_u = layers.Conv2D(int(round(initial_filter*1.5)), (1,16), kernel_initializer=initializers.glorot_normal(seed=seed+5),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,8))(x)
            x_y_t = layers.Conv2D(int(round(initial_filter*1.5)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+9),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,1))(x)
               
            x_y_v = layers.Softmax(axis=-1)(x_y_u)
            x_y_v = layers.Lambda(lambda x: x/K.max(x))(x_y_v)
            x_y = layers.Multiply()([x_y_v, x_y])
            x_y = layers.Add()([x_y_t,x_y])
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,16), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8),kernel_constraint=min_max_norm(-1,1,rate=0.001))(x_y)
            x_y = layers.LeakyReLU()(x_y)
            return x_y
        if type_of_block=="inception":
            x_v_0 = layers.Conv2D(initial_filter, (1,1),kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed))(x)
            
            x_v_1_0 = layers.Conv2D(int(round(initial_filter*1.5)), (1,1), kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.he_uniform(seed=seed+1))(x)
            x_v_1_1 = layers.Conv2D(initial_filter, (1,3),padding='same')(x_v_1_0)
            x_v_1_2 = layers.Conv2D(initial_filter, (3,1),padding='same')(x_v_1_0)
            
            x_v_2 = layers.Conv2D(int(round(initial_filter*1.5)), (1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+2))(x)
            x_v_2 = layers.Conv2D(int(round(initial_filter*1.75)), (1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+3))(x_v_2)
            x_v_2 = layers.Conv2D(int(round(initial_filter*2)), (3,1),padding='same',kernel_initializer=initializers.glorot_uniform(seed=seed+4))(x_v_2)
            x_v_2_0 = layers.Conv2D(initial_filter, (3,1),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+5))(x_v_2)
            x_v_2_1 = layers.Conv2D(initial_filter, (1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+6))(x_v_2)
            
            x_v_3 = layers.AveragePooling2D((2, 2), strides=(1,1),padding='same')(x)
            x_v_3 = layers.Conv2D(initial_filter, (1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+7))(x_v_3)
            
            x_y = layers.Concatenate()([x_v_0, x_v_1_1,x_v_1_2, x_v_2_0,x_v_2_1, x_v_3])
            if self.ApplyLayerNormalization:
                x_y = layers.LayerNormalization(scale=True, center=True)(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation('relu')(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_uniform(seed=seed+8))(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(scale=False)(x_y)
	    
            x_y = layers.Activation('relu')(x_y)
        if type_of_block=="resnet":
            x_y = conv_block(x, 3, [initial_filter, int(round(initial_filter*1.5)), initial_filter*1], stage=stage, block='a', strides=(1, 1))
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*1)), initial_filter*1], stage=stage, block='b')
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*1)), initial_filter*1], stage=stage, block='c')
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*1.5)), initial_filter*1], stage=stage, block='d')
        if type_of_block=="vgg":
            x_y = layers.Conv2D(int(round(initial_filter)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+8), kernel_regularizer=kernel_regularizer, padding='same')(x)
            x_y = layers.Conv2D(int(round(initial_filter*1.5)), (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+10),kernel_regularizer=kernel_regularizer, padding='same')(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
            x_y = layers.Activation("relu")(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8))(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation("relu")(x_y)
        if type_of_block=="soft_att":
            kernel_regularizer=l2(0.000001)
            x = layers.LayerNormalization(scale=True, center=True)(x)
            if x.shape.as_list()[2]<14:
                x_y = layers.Conv2D(int(round(initial_filter*1.5)), (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
                x_y_u = layers.Conv2D(int(round(initial_filter*1.5)), (3,3), kernel_initializer=initializers.glorot_normal(seed=seed+5),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,1))(x)
                x_y_t = layers.Conv2D(int(round(initial_filter*1.5)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+9),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,1))(x)
                
                x_y_v = layers.Softmax(axis=-1)(x_y_u)
                x_y_v = layers.Lambda(lambda x: x/K.max(x))(x_y_v)
                x_y = layers.Multiply()([x_y_v, x_y])
                x_y = layers.Add()([x_y_t,x_y])
            else:
                x_y_1 = layers.Conv2D(int(round(initial_filter))*1, (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(3,3))(x)
                x_y_2 = layers.Conv2D(int(round(initial_filter))*1, (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(2,2))(x)
                x_y = layers.concatenate([x_y_1,x_y_2], axis=-1)
                x_y_t = layers.Conv2D(int(round(initial_filter*1))*2, (1,1),kernel_initializer=initializers.he_normal(seed=seed+9),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,1))(x)
                x_y = layers.Add()([x_y_t,x_y])
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8),kernel_constraint=min_max_norm(-1,1,rate=0.001))(x_y)
            x_y = layers.LeakyReLU()(x_y)
        if type_of_block=="soft_att_all":
            kernel_regularizer=l2(0.000001)
            x = layers.LayerNormalization(scale=True, center=True)(x)
            x_y = layers.Conv2D(int(round(initial_filter*1.5)), (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
            x_y_u = layers.Conv2D(int(round(initial_filter*1.5)), (3,3), kernel_initializer=initializers.glorot_normal(seed=seed+5),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(3,3))(x)
            x_y_t = layers.Conv2D(int(round(initial_filter*1.5)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+9),kernel_regularizer=kernel_regularizer, padding='same',kernel_constraint=min_max_norm(-1,1,rate=0.001), dilation_rate=(1,1))(x)
               
            x_y_v = layers.Softmax(axis=-1)(x_y_u)
            x_y_v = layers.Lambda(lambda x: x/K.max(x))(x_y_v)
            x_y = layers.Multiply()([x_y_v, x_y])
            x_y = layers.Add()([x_y_t,x_y])
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8),kernel_constraint=min_max_norm(-1,1,rate=0.001))(x_y)
            x_y = layers.LeakyReLU()(x_y)
        if type_of_block=="vgg_short":
            x_y = layers.Conv2D(int(round(initial_filter)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+8), kernel_regularizer=kernel_regularizer, padding='same')(x)
            x_y = layers.Conv2D(int(round(initial_filter*1.5)), (3,3),kernel_initializer=initializers.lecun_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same')(x_y)
            x_y = layers.LeakyReLU()(x_y)
            shape_v=x_y.shape.as_list()[1:3]
            shape_z=initial_image.shape.as_list()[1:3]
            height_factor=shape_z[0]//shape_v[0]
            width_factor=shape_z[1]//shape_v[1]
            if height_factor!=1 and width_factor!=1:
                initial_img_=layers.MaxPooling2D((height_factor,width_factor), strides=(height_factor,width_factor))(initial_image)
                x_y = layers.Concatenate()([initial_img_,x_y])
            x_y = layers.Conv2D(int(round(initial_filter*2)), (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+8),kernel_regularizer=kernel_regularizer, padding='same')(x_y)
            x_y = layers.AveragePooling2D((2, 2),strides=(1,1), padding='same')(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8))(x_y)
            if self.run_all_BN:
                x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.LeakyReLU()(x_y)
        return x_y

    def Spider_Node(self, x_input, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0, type_of_block="inception", initial_image=None):
        node = []
        x = x_input
        #            for j, layer in enumerate(vb):
        #        vb[j]._name = f"C_{i}_{layer.name}"
        for i in range(depth):
            x = self._conv_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter), type_of_block=type_of_block, initial_image=initial_image)
            node.append(x)
            if self.MIL_mode:
                x = layers.Conv2D(filter*(i+1)+2, (1,4),strides=(1,4), padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
            else:
                x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
        return node

    def Spider_Node_w_Junction(self, x_input, node, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0, type_of_block="inception", initial_image=None):
        node_tmp = []
        x = x_input
        for i in range(depth):
            x = self._conv_block(layers.concatenate([x,node[i]]), filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter), type_of_block=type_of_block, initial_image=initial_image)
            node_tmp.append(x)
            if self.MIL_mode:
                x = layers.Conv2D(filter*(i+1)+2, (1,4),strides=(1,4), padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
            else:
                x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node_tmp

    def Spider_Node_w_Junction_list(self, x_input, node, filter,compression=0.5, depth=5, junction_list=None, kernel_regularizer=regularizers.l2(0.00001), counter=0, type_of_block="inception",initial_image=None ):
        node_tmp = []
        x = x_input
        for i in range(depth):
            if junction_list is None:
                pass
            elif i in junction_list:
                x = layers.concatenate([x,node[i]])
            else:
                pass
            x = self._conv_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter),type_of_block=type_of_block, initial_image=initial_image)
            
            node_tmp.append(x)
            if self.MIL_mode:
                x = layers.Conv2D(filter*(i+1)+2, (1,4),strides=(1,4), padding='same', kernel_constraint=min_max_norm(-1,1,rate=0.001))(x)
            else:
                x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node_tmp

    def Connect_Nodes(self, nodes, center_node_id=0):
        number_of_nodes = len(nodes)
        length_of_node = len(nodes[0])
        connection_A =[]
        connection_B = []
        type_of_junction =[]
        for i in range(length_of_node):
            for j in range(number_of_nodes):
                if j != center_node_id:
                    connection_A.append(nodes[center_node_id][i])
                    connection_B.append(nodes[j][i])
                    type_of_junction.append(layers.add)
        return connection_A, connection_B, type_of_junction

    def Connect_Different_Nodes(self, nodes, junctions, iteration_data):
        itm_nbr = iteration_data.shape[0]
        for i in range(itm_nbr):
            row = iteration_data[i]
            node_id, level_id, junc_id = row[0], row[1], row[2]
            print(junctions[junc_id])
            print(nodes[node_id][level_id])
            nodes[node_id][level_id] = layers.Concatenate()([junctions[junc_id],nodes[node_id][level_id]])
        return nodes

    def GetIterationConnectionKeys(self,number_of_nodes, depth, random_junctions=True, number_of_junctions=12):
        nodes = list(range(1, number_of_nodes))
        depth_lst = list(range(0,depth))
        counter = list(range(0,(number_of_nodes-1)*depth))
        
        import numpy as np
        nodes_ = np.array(nodes*depth)
        depth_ = np.array(depth_lst*(number_of_nodes-1))
        counter = np.array(counter)
        np_ = np.zeros((((number_of_nodes-1)*depth),3), dtype=np.int)

        np_[...,0] = nodes_
        np_[...,1] = depth_
        np_[...,2] = counter
        for i in range(depth):
            np_[i*(number_of_nodes-1):(i+1)*(number_of_nodes-1),1] = i
        if random_junctions and number_of_junctions>0:
            indexes = list(range(np_.shape[0]))
            random.seed=1
            indexes_selected = random.sample(indexes,number_of_junctions)
            random.seed=None
            np_ = np_[indexes_selected]
            print(np_.shape)
        return np_
    def Core(self, x, initial_filter=32, compression=0.5, length=5, depth=7, center_node_id=0, kernel_regularizer=None,random_junctions=True, number_of_junctions=5, junction_only_the_last_layers=False, type_of_block="inception", initial_image=None):
        nodes = []
        #Generate nodes
        for i in range(length):
            if self.lanewise_augmentation:
                if i in self.ApplyLaneForAugmentation:
                    x = tf.image.decode_image(x)
                    x = self.data_augmentation(tf.cast(x, dtype=tf.float32))
            vb = self.Spider_Node(x, initial_filter, compression,depth, kernel_regularizer, counter=i, type_of_block=type_of_block, initial_image=initial_image)


            nodes.append(vb)

        #Generate Connection between Nodes
        #Generate the junctions between nodes
        connected_layers = []
        con_A, con_B, _ = self.Connect_Nodes(nodes)
        with open('./junction.txt', 'w') as f:
            for (conv_A, conv_B) in zip(con_A,con_B):
                connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
                f.write("%s %s\n" % (conv_A, conv_B))
        '''
        if os.path.exists('./junction.txt'):
            with open('./junction.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    conv_A, conv_B = line.split(' ')
                    connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
        else:
            con_A, con_B, _ = self.Connect_Nodes(nodes)
            with open('./junction.txt', 'w') as f:
                for (conv_A, conv_B) in zip(con_A,con_B):
                    connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
                    f.write("%s %s\n" % (conv_A, conv_B))
        '''
        iteration_data = self.GetIterationConnectionKeys(number_of_nodes=length, depth=depth, random_junctions=random_junctions,number_of_junctions=number_of_junctions)
        
        #Update the node structure:
        itm_nbr = iteration_data.shape[0]
        for i in range(itm_nbr):
            row = iteration_data[i]
            node_id, level_id, junc_id = row[0], row[1], row[2]
            if junction_only_the_last_layers:
                nodes[node_id][level_id] = layers.Concatenate()([connected_layers[junc_id],nodes[node_id][level_id]])
            else:
                nodes[node_id][level_id] = connected_layers[junc_id]

        if junction_only_the_last_layers:
            y = connected_layers[-(length-1):]
            y = layers.concatenate(y)
        elif random_junctions==False:
            second_nodes = []
            for i in range(length):
                if i != center_node_id:
                    second_nodes.append(self.Spider_Node_w_Junction(x, nodes[i], initial_filter,compression,depth, kernel_regularizer, counter=i, type_of_block=type_of_block, initial_image=initial_image))
                else:
                    second_nodes.append(nodes[i])
            last_connection = []
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            if len(last_connection)>0:
                y = layers.concatenate(last_connection)

        elif random_junctions:
            second_nodes = []
            node_id_junctions = iteration_data[:,0].tolist()
            print(node_id_junctions)
            for i in range(length):
                if i == center_node_id:
                    second_nodes.append(nodes[i])
                else:
                    if i in node_id_junctions:
                        junction_levels = iteration_data[iteration_data[:,0]==i][:,1].tolist()
                    else:
                        junction_levels = None
                    second_nodes.append(self.Spider_Node_w_Junction_list(x, nodes[i], initial_filter, compression, depth, junction_levels,kernel_regularizer, counter=i, type_of_block=type_of_block, initial_image=initial_image))
            last_connection = []
            print(len(second_nodes))
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            
            y = layers.concatenate(last_connection)
        else:
            raise NameError('Please specify the arguments including junction_only_the_last_layers, random_junctions')
        #Apply Transformer
        if self.ApplyTransformer:
            dense_shape = y.shape.as_list()
            num_heads = Configuration["num_heads"]
            y=layers.Reshape((dense_shape[1]*dense_shape[2],dense_shape[-1]))(y)
            embed_dim=y.shape.as_list()[-1]
            ff_dim = y.shape.as_list()[-1]
            transformer_blocks = []
            for _ in range(Configuration["number_of_transformer_blocks"]):
                transformer_blocks.append(TransformerBlock(embed_dim, num_heads, ff_dim))
                
            for transformer_block in transformer_blocks:
                y = transformer_block(y)
            y=layers.Reshape((dense_shape[1],dense_shape[2],dense_shape[-1]))(y)
        #FC: You can change here whatever you want.
        if self.get_last_conv:
            return y
        if self.GlobalPooling is None or self.MIL_mode:
            y = layers.Flatten()(y)
        elif self.GlobalPooling=="max":
            y = layers.GlobalMaxPooling2D()(y)
        elif self.GlobalPooling=="avg":
            y = layers.GlobalAveragePooling2D()(y)

        #Supervised-Contrastive-Learning
        if self.SCL:
            if self.RunLayerNormalizationInSCL:
                y = layers.LayerNormalization()(y)
            return y
        #Contrastive Predictive Coding
        if self.CPC:
            y = layers.Flatten()(y)
            y = layers.Dense(units=256, activation='linear')(y)
            y = layers.LayerNormalization()(x)
            y = layers.LeakyReLU()(x)
            y = layers.Dense(units=self.code_size, activation='linear', name='encoder_embedding')(x)
            return y
            
        dense_shape = y.shape.as_list()[-1]
        #dense_shape = 1024
        if self.MIL_mode or self.MIL_CONV_mode:
            #weight_decay=0.00001
            y = layers.Dense(int(round(self.MIL_FC_percentage_of_feature*dense_shape)), activation= 'relu')(y)
            alpha = Mil_Attention(L_dim=int(round(self.MIL_FC_percentage_of_feature*dense_shape)), output_dim=1, name='alpha', use_gated=self.useGated)(y)
            x_mul = layers.Multiply()([alpha, y])
            y = layers.Dense(self.n_class, activation=self.final_activation)(x_mul)
            #y = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
            return y
            
        if self.apply_RandomFourierFeatures:
            y = layers.Dense(dense_shape)(y)
            y_ = RandomFourierFeatures(output_dim=dense_shape, scale=10.0, kernel_initializer="gaussian")(y)
            y = layers.Dense(self.n_class,activation=self.final_activation)(y_)
            return y
        else:
            if self.ApplyTransformer:
                y = layers.Dense(dense_shape, activation=tfa.activations.gelu)(y)
            else:
                y = layers.Dense(dense_shape, activation= 'selu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        return y
    def Save(self, filename):
        self.model.save(filename)
        print("saved...")              
from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers, regularizers
def add_projection_head(encoder,input_shape,projection_units):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
    )
    return model

class Mil_Attention(Layer):
    """
    Mil Attention Mechanism
    This layer contains Mil Attention Mechanism
    # Input Shape
        2D tensor with shape: (batch_size, input_dim)
    # Output Shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=False, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.tanh(K.dot(x, self.V)) # (2,64)

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(ori_x, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        alpha = K.softmax(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Last_Sigmoid(Layer):
    """
    Attention Activation
    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.
    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class CPCLayer(Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output
# Projector Network
def projector_net(hiddenunits=256, activation="relu", ApplyUnitNormLayer=False):
	projector = tf.keras.models.Sequential([
		Dense(hiddenunits, activation=activation),
		UnitNormLayer()
	])
	return projector

# Reference: https://github.com/wangz10/contrastive_loss/blob/master/model.py
class UnitNormLayer(layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,)
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


#Transformer sectopn
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self,embed_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    def get_config(self):
        config = super().get_config().copy()
        config.update({"embed_dim": self.embed_dim,
            "num_heads": self.num_heads
            #"projection_dim" : self.projection_dim,
            #"query_dense" : self.query_dense,
            #"key_dense" : self.key_dense,
            #"value_dense" : self.value_dense,
            #"combine_heads" : self.combine_heads
            })
        return config
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim=embed_dim
        self.ff_dim = ff_dim
        self.num_heads=num_heads
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            #"att": self.att,
            #"ffn": self.ffn,
            "embed_dim": self.embed_dim,
            "ff_dim" : self.ff_dim,
            "num_heads" : self.num_heads
            #"layernorm1" : self.layernorm1,
            #"layernorm2" : self.layernorm2,
            #"dropout1" : self.dropout1,
            #"dropout2" : self.dropout2
            })
        return config
