"""
Copyright by Okyaz Eminaga. 2020
"""
from keras.engine.topology import Layer
from keras.constraints import min_max_norm
import keras.backend as K
class RotationThetaWeightLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(RotationThetaWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        
        self.W1 = self.add_weight(name='kernel_01', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        self.W2 = self.add_weight(name='kernel_02', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        
        super(RotationThetaWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x

        return K.cos(self.W1*90) * (-2) * K.exp(-(a**2+b**2)) + K.sin(self.W2*90) * (-2) * b * K.exp(-(a**2+b**2))
class JunctionWeightLayer(Layer): # a junction layer
    def __init__(self,  **kwargs):
        self.func_junction = layers.add
        super(JunctionWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape describes the number of the junctions.
        '''
        assert isinstance(input_shape, list)
        self.W1 = self.add_weight(name='junction_weight_first_element', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0, max_value=1))
        self.W2 = self.add_weight(name='junction_weight_second_element', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0, max_value=1))
        super(JunctionWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        a = a * self.W1 
        b = b * self.W2
        v = self.func_junction([a,b])
        return v