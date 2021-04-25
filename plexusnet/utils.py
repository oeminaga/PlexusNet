"""
Copyright by Okyaz Eminaga. 2020
"""
from keras.engine.topology import Layer
from keras.constraints import min_max_norm
from keras import layers
import keras.backend as K
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
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

# Visualization utility functions
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label,CLASSES):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', 
                  fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

def display_batch_of_images(databatch, CLASSES, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
    
# Visualize model predictions
def dataset_to_numpy_util(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break;  
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    label = np.argmax(label, axis=-1)
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(label, str(correct), ', shoud be ' if not correct else '',
                                correct_label if not correct else ''), correct

def display_one_flower_eval(image, title, subplot, red=False):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=14, color='red' if red else 'black')
    return subplot+1

def display_9_images_with_predictions(images, predictions, labels):
    subplot=331
    plt.figure(figsize=(13,13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i])
        subplot = display_one_flower_eval(image, title, subplot, not correct)
        if i >= 8:
            break;
              
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


# Model evaluation
def plot_metrics(history):
    fig, axes = plt.subplots(2, 1, sharex='col', figsize=(20, 8))
    axes = axes.flatten()
    
    axes[0].plot(history['loss'], label='Train loss')
    axes[0].plot(history['val_loss'], label='Validation loss')
    axes[0].legend(loc='best', fontsize=16)
    axes[0].set_title('Loss')
    axes[0].axvline(np.argmin(history['loss']), linestyle='dashed')
    axes[0].axvline(np.argmin(history['val_loss']), linestyle='dashed', color='orange')
    
    axes[1].plot(history['sparse_categorical_accuracy'], label='Train accuracy')
    axes[1].plot(history['val_sparse_categorical_accuracy'], label='Validation accuracy')
    axes[1].legend(loc='best', fontsize=16)
    axes[1].set_title('Accuracy')
    axes[1].axvline(np.argmax(history['sparse_categorical_accuracy']), linestyle='dashed')
    axes[1].axvline(np.argmax(history['val_sparse_categorical_accuracy']), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
    
def visualize_embeddings(embeddings, labels, seed=123, figsize=(16, 16)):
    # Extract TSNE values from embeddings
    embed2D = TSNE(n_components=2, n_jobs=-1, random_state=seed).fit_transform(embeddings)
    embed2D_x = embed2D[:,0]
    embed2D_y = embed2D[:,1]

    # Create dataframe with labels and TSNE values
    df_embed = pd.DataFrame({'labels': labels})
    df_embed = df_embed.assign(x=embed2D_x, y=embed2D_y)

    # Create classes dataframes
    df_embed_cbb = df_embed[df_embed['labels'] == 0]
    df_embed_cbsd = df_embed[df_embed['labels'] == 1]
    df_embed_cgm = df_embed[df_embed['labels'] == 2]
    df_embed_cmd = df_embed[df_embed['labels'] == 3]
    df_embed_healthy = df_embed[df_embed['labels'] == 4]
    
    # Plot embeddings
    plt.figure(figsize=figsize)
    plt.scatter(df_embed_cbb['x'], df_embed_cbb['y'],color='yellow',s=10,label='CBB')
    plt.scatter(df_embed_cbsd['x'], df_embed_cbsd['y'],color='blue',s=10,label='CBSD')
    plt.scatter(df_embed_cgm['x'], df_embed_cgm['y'],color='red',s=10,label='CGM')
    plt.scatter(df_embed_cmd['x'], df_embed_cmd['y'],color='orange',s=10,label='CMD')
    plt.scatter(df_embed_healthy['x'], df_embed_healthy['y'],color='green',s=10,label='Healthy')

    plt.legend()
    plt.show()