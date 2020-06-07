"""
Copyright by Okyaz Eminaga. 2020
"""
from keras import layers, models,optimizers
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.constraints import min_max_norm
from keras import regularizers, initializers
from keras.regularizers import l2
import random
import os
from . import utils
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D,Activation, Dropout, Dense, GlobalMaxPooling2D
from .functions import *

def LoadModel(filename, custom_objects={},optimizer= optimizers.Adam(), loss="categorical_crossentropy"):
    custom_objects_internal = {'JunctionWeightLayer':utils.JunctionWeightLayer, 'RotationThetaWeightLayer': utils.RotationThetaWeightLayer}
    for  key in custom_objects:
        custom_objects_internal[key]=custom_objects[key]

    model_=load_model(filename, custom_objects=custom_objects_internal, compile=False)
    model_.compile(optimizer=optimizer, metrics=[""], loss=loss)
    return model_
    
class PlexusNet():
    def __init__(self, input_shape=(512,512), initial_filter=2, length=2, depth=7, junction=3, n_class=2, number_input_channel=3, compression_rate=0.5,final_activation="softmax", random_junctions=True, type_of_block="inception", normalize_by_factor=1.0/255.0):
        """
        Architecture hyperparameter are:
        initial_filter (Default: 2)
        length (Default: 2)
        depth (Default: 6)
        junction (Default: 3)
        
        You can modify or search the optimal architecture using these hyperparameters. Please be advised that the training set should be representative of different variation to build sophisticated models.

        """
        self.input_shape= input_shape
        self.n_class = n_class
        self.initial_filter=initial_filter
        self.length = length
        self.depth =depth
        self.junction = junction
        self.number_input_channel=number_input_channel
        self.final_activation=final_activation
        self.compression_rate = compression_rate
        self.random_junctions =random_junctions
        self.normalize_by_factor=normalize_by_factor
        self.type_of_block =type_of_block
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x*self.normalize_by_factor)(x)
        x_y_o = ColorIntensityNormalisationSection(x_y_o)
        #rescale
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        #Generate multiple channels from the image
        x_y = Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        y = self.Core(x_y, initial_filter = self.initial_filter, length=self.length, depth=self.depth, number_of_junctions=self.junction, compression=self.compression_rate, type_of_block=self.type_of_block)
        self.model = models.Model(inputs=x, outputs=y)
    
    def _conv_block(self, x, initial_filter, reduction_channel_ratio=0.5, kernel_regularizer=None, seed=0, type_of_block="inception", stage=0, initial_image=None):
        """
        x : tensor
        initial_filter : integer
        type_of_block selection: inception,resnet,vgg,vgg_short
        """
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
            x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation('relu')(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_uniform(seed=seed+8))(x_y)
            x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation('relu')(x_y)
        if type_of_block=="resnet":
            x_y = conv_block(x, 3, [initial_filter, int(round(initial_filter*1.5)), initial_filter*2], stage=stage, block='a', strides=(1, 1))
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*2)), initial_filter*2], stage=stage, block='b')
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*2)), initial_filter*2], stage=stage, block='c')
            x_y = identity_block(x_y, 3, [initial_filter, int(round(initial_filter*1.5)), initial_filter*2], stage=stage, block='d')
        if type_of_block=="vgg":
            x_y = layers.Conv2D(int(round(initial_filter)), (1,1),kernel_initializer=initializers.he_normal(seed=seed+8), kernel_regularizer=kernel_regularizer, padding='same')(x)
            x_y = layers.Conv2D(int(round(initial_filter*1.5)), (3,3),kernel_initializer=initializers.glorot_normal(seed=seed+10),kernel_regularizer=kernel_regularizer, padding='same')(x_y)
            x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
            x_y = layers.Activation("relu")(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8))(x_y)
            x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation("relu")(x_y)
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
            x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
            shape_c = x_y.shape.as_list()[-1]
            x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=seed+8))(x_y)
            x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.LeakyReLU()(x_y)
        return x_y

    def Spider_Node(self, x_input, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0, type_of_block="inception", initial_image=None):
        node = []
        x = x_input
        for i in range(depth):
            x = self._conv_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter), type_of_block=type_of_block, initial_image=initial_image)
            node.append(x)
            x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node

    def Spider_Node_w_Junction(self, x_input, node, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0, type_of_block="inception", initial_image=None):
        node_tmp = []
        x = x_input
        for i in range(depth):
            x = self._conv_block(layers.concatenate([x,node[i]]), filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter), type_of_block=type_of_block, initial_image=initial_image)
            node_tmp.append(x)
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
            nodes.append(self.Spider_Node(x, initial_filter, compression,depth, kernel_regularizer, counter=i, type_of_block=type_of_block, initial_image=initial_image))

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
            raise NameError('Please specify the arguments, junction_only_the_last_layers, random_junctions')

        #FC
        y = layers.GlobalMaxPooling2D()(y)
        dense_shape = y.shape.as_list()[-1]
        #dense_shape = 1024
        y = layers.Dense(dense_shape, activation= 'relu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        return y
