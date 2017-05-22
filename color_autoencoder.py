#!/usr/bin/env python

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import scipy
import scipy.ndimage
import random
from pathlib import Path
from tqdm import tqdm

def load_raw_data_set():
    """
Load the COIL database as an example of toy set of images. Return a list of
(class, object) labels
    """
    pathPattern = 'coil-100/obj{}__{}.png'
    numObjects = 100
    numViews = 72
    degreesPerView = 5

    samples = []
    
    for obj in tqdm(range(1, numObjects + 1), unit='item', leave=False, desc='Object'):
        for view in range(0, numViews * degreesPerView, degreesPerView):
            path = pathPattern.format(obj, view)
            img = (scipy.ndimage.imread(path).astype(np.float_).flatten() - 128.0) / 256.0
            label = obj

            samples.append((label, img))


    print('Loaded {} samples with length {}'.format(len(samples), samples[0][1].size))
    return samples


def split_samples_for_autoencoder_training(samples):
    """
    Given a set of (label, feature) samples, product a 80%/20% split of just features
"""
    just_features = [x[1] for x in samples]
    n = len(just_features)
    split = int(4*n/5);
    
    random.shuffle(just_features)

    train_split = np.array(just_features[:split])
    valid_split = np.array(just_features[split:])
    
    return (train_split, valid_split)

def main():
    samples = load_raw_data_set()
    (autoencoder_training, autoencoder_validation) = split_samples_for_autoencoder_training(samples)

    # Original size
    original_dimension = samples[0][1].size
        
    # 5:1 'compression' 
    encoding_dimension = int(20 * samples[0][1].size / 100)
    
    # https://blog.keras.io/building-autoencoders-in-keras.html
    input_image = Input(shape = (original_dimension,))
    
    encoded = Dense(encoding_dimension,
                    activation='relu')(input_image)
    
    decoded = Dense(original_dimension,
                    activation='sigmoid')(encoded)
    
    autoencoder = Model(input_image, decoded)
    
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    
    history = autoencoder.fit(autoencoder_training,
                         autoencoder_training,
                         epochs=100,
                         batch_size=100,
                         shuffle=True,
                         verbose=2,
                         validation_data=(autoencoder_validation, autoencoder_validation))
    
    
if '__main__' == __name__:
    main()
