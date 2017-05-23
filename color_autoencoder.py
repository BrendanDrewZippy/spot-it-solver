#!/usr/bin/env python

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import CSVLogger, LambdaCallback
from keras import regularizers

import math
import collections
import keras.models as models
import numpy as np
import scipy
import scipy.ndimage
import random
from pathlib import Path
from tqdm import tqdm
import shutil
import os
import json

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
            img = scipy.ndimage.imread(path).astype(np.float_) / 256.0
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

    print('Training data shape: {}, validation data shape: {}'.format(str(train_split.shape), str(valid_split.shape)))

    return (train_split, valid_split)


def create_autoencoder_model(kernel_lambda, activity_lambda, input_shape):
    """
All the network creation magic happens here.
Parameters:
kernel_lambda:   L2 regularizer weight for the kernel (not sure what this actually means)
activity_lambda: L1 regularizer weight for activations (also not sure what this actually means)
input_shape:     Shape of input images
"""
    num_filters = 8
    filter_size = (16, 16)
    pooling_size = (2, 2)
    input_image = Input(shape = input_shape)

    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda),
               input_shape=input_shape)(input_image)

    x = MaxPooling2D(pooling_size, padding='same')(x)
    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda))(x)
    x = MaxPooling2D(pooling_size, padding='same')(x)
    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda))(x)
    encoded = MaxPooling2D(pooling_size, padding='same')(x)

    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda))(encoded)
    x = UpSampling2D(pooling_size)(x)
    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda))(x)
    x = UpSampling2D(pooling_size)(x)
    x = Conv2D(num_filters,
               filter_size,
               use_bias=True,
               activation='relu',
               padding='same',
               data_format='channels_last',
               kernel_regularizer=regularizers.l2(kernel_lambda),
               activity_regularizer=regularizers.l1(activity_lambda))(x)
    x = UpSampling2D(pooling_size)(x)
    decoded = Conv2D(3,
                     filter_size,
                     use_bias=True,
                     activation='sigmoid',
                     padding='same',
                     data_format='channels_last',
                     kernel_regularizer=regularizers.l2(kernel_lambda),
                     activity_regularizer=regularizers.l1(activity_lambda))(x)
    return Model(input_image, decoded)

def create_progress_callback(batch_size, training_size, num_epochs):
    epochs_progress = None
    this_epoch_progress = None

    def on_epoch_begin(epoch, logs):
        nonlocal epochs_progress, this_epoch_progress
        this_epoch_progress = tqdm(miniters=1, total=training_size, desc='Epoch {} Progress'.format(epoch), leave=False, unit='Samples')

    def on_epoch_end(epoch, logs):
        nonlocal epochs_progress, this_epoch_progress
        epochs_progress.update(1)
        epochs_progress.set_description('MAE {:<7.06f} / Loss {:<8.03f}'.format(logs.get('mean_absolute_error'), logs.get('loss')))

        this_epoch_progress.close()
        this_epoch_progress = None

    def on_batch_begin(batch, logs):
        pass

    def on_batch_end(batch, logs):
        nonlocal this_epoch_progress
        this_epoch_progress.update(min(batch_size, training_size - this_epoch_progress.n))

    def on_train_begin(logs):
        nonlocal epochs_progress
        epochs_progress = tqdm(miniters=1, total=num_epochs, desc='Training Epochs', leave=False, unit='Epoch')

    def on_train_end(logs):
        nonlocal epochs_progress
        epochs_progress.close()
        epochs_progress = None

    return LambdaCallback(on_epoch_begin = on_epoch_begin,
                          on_epoch_end = on_epoch_end,
                          on_batch_begin = on_batch_begin,
                          on_batch_end = on_batch_end,
                          on_train_begin = on_train_begin,
                          on_train_end = on_train_end)


def main():
    samples = load_raw_data_set()
    (autoencoder_training, autoencoder_validation) = split_samples_for_autoencoder_training(samples)

    kernel_lambdas = [10, 8, 6, 4, 2, 1]
    activity_lambdas = [10, 8, 6, 4, 2, 1]

    trainingEpochs = 100
    batchSize = 100

    performance = collections.defaultdict(dict)

    print('Performing grid search on kernel_lambdas x activity_lambdas')
    for kernel_lambda in tqdm(kernel_lambdas, desc='Kernel Lambda', leave=False):
        for activity_lambda in tqdm(activity_lambdas, desc='Activity Lambda', leave=False):
            output_base = Path('.').resolve() / 'K{:03d}-A{:03d}'.format(kernel_lambda, activity_lambda)

            if output_base.exists():
                shutil.rmtree(str(output_base))
            os.makedirs(str(output_base))
            

            autoencoder = create_autoencoder_model(math.pow(10, -kernel_lambda), math.pow(10, -activity_lambda), samples[0][1].shape)
            autoencoder.compile(optimizer='nadam',
                                metrics=['mean_absolute_percentage_error',
                                         'mean_squared_error',
                                         'mean_absolute_error'],
                                loss='mean_absolute_error')

            csv_file = str(output_base / 'solver.csv')
            csv_callback = CSVLogger(csv_file, separator=',', append='False')
            lambda_callback = create_progress_callback(batchSize, autoencoder_training.shape[0], trainingEpochs)

            history = autoencoder.fit(autoencoder_training,
                                      autoencoder_training,
                                      epochs=trainingEpochs,
                                      batch_size=batchSize,
                                      shuffle=True,
                                      verbose=0,
                                      callbacks=[csv_callback, lambda_callback])

            loss_and_metrics = autoencoder.evaluate(autoencoder_validation, autoencoder_validation, batch_size=128)
            performance[kernel_lambda][activity_lambda] = loss_and_metrics
            models.save_model(autoencoder, output_base / 'model', overwrite=True)

            metrics_file = str(output_base / 'metrics.json')
            with open(metrics_file, 'wt') as metrics:
                json.dump(loss_and_metrics, metrics, ensure_ascii=True, sort_keys=True, indent=2)            

if '__main__' == __name__:
    main()
