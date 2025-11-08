
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import time
import gc
import csv
import argparse
import numpy as np


def create_cnn(input_shape, num_classes, cnn_config):
    conv_params_list = cnn_config['conv_params_list']
    pool_layer_template = cnn_config['pool_layer_template']
    pool_flags_list = cnn_config['pool_flags_list']
    dropout_rates_list = cnn_config['dropout_rates_list']
    dense_layer_nodes = cnn_config['dense_layer_nodes']
    optimizer = cnn_config['optimizer']
    
    if len(conv_params_list) != len(pool_flags_list):
        raise ValueError("conv_params_list and pool_flags_list must have the same length.")
    if len(conv_params_list) + 1 != len(dropout_rates_list):
        raise ValueError("dropout_rates_list must have N+1 elements (N conv blocks + 1 dense block).")

    model = Sequential()
    
    first_conv_params = conv_params_list[0]
    filters, kernel_size, strides = first_conv_params
    model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu',
                     input_shape=input_shape, padding='valid'))

    if pool_flags_list[0]:
        model.add(pool_layer_template)
        model.add(Dropout(dropout_rates_list[0]))

    for i in range(1, len(conv_params_list)):
        filters, kernel_size, strides = conv_params_list[i]
        model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu'))
        
        if pool_flags_list[i]:
            model.add(pool_layer_template)
            model.add(Dropout(dropout_rates_list[i]))
            
    model.add(Flatten())
    model.add(Dense(dense_layer_nodes, activation='relu'))
    model.add(Dropout(dropout_rates_list[-1]))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

BEST_LR = 0.01 
FIXED_OPTIMIZER = optimizers.Adam(learning_rate=BEST_LR)

CNN_ARCHITECTURES = {
    "Baseline_Git_Repo": {
        "conv_params_list": [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True],
        "dropout_rates_list": [0.0, 0.25, 0.5],
        "dense_layer_nodes": 128,
        "optimizer": FIXED_OPTIMIZER
    },
    "Wide_Model": {
        "conv_params_list": [(64, (3, 3), (1, 1)), (128, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True],
        "dropout_rates_list": [0.0, 0.25, 0.5],
        "dense_layer_nodes": 256,
        "optimizer": FIXED_OPTIMIZER   
    },
    "Aggressive_Reduction": {
        "conv_params_list": [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [True, True],
        "dropout_rates_list": [0.3, 0.3, 0.5],
        "dense_layer_nodes": 128,
        "optimizer": FIXED_OPTIMIZER
    },
    "Large_Kernel_Start": {
        "conv_params_list": [(32, (5, 5), (1, 1)), (64, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True],
        "dropout_rates_list": [0.0, 0.25, 0.5],
        "dense_layer_nodes": 128,
        "optimizer": FIXED_OPTIMIZER
    },
    "Deeper_3Conv_Model": {
        "conv_params_list": [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (128, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True, True],
        "dropout_rates_list": [0.0, 0.2, 0.2, 0.5],
        "dense_layer_nodes": 256,
        "optimizer": FIXED_OPTIMIZER
    },
    "Average_Pooling_Model": {
        "conv_params_list": [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1))],
        "pool_layer_template": AveragePooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True],
        "dropout_rates_list": [0.0, 0.25, 0.5],
        "dense_layer_nodes": 128,
        "optimizer": FIXED_OPTIMIZER
    },
    "Extra_Deep_5Conv": {
        "conv_params_list": [(32, (3, 3), (1, 1)), (32, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (128, (3, 3), (1, 1))],
        "pool_layer_template": MaxPooling2D(pool_size=(2, 2)),
        "pool_flags_list": [False, True, False, True, True],
        "dropout_rates_list": [0.0, 0.2, 0.0, 0.3, 0.4, 0.5],
        "dense_layer_nodes": 512,
        "optimizer": FIXED_OPTIMIZER
    }
}


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--arch_name", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()


X_train_main = np.load('data_X_train.npy')
y_train_main = np.load('data_y_train.npy')
X_val = np.load('data_X_val.npy')
y_val = np.load('data_y_val.npy')
num_classes = 10
input_shape = (28, 28, 1) 


try:
    t_start = time.time()
    
    cnn_config = CNN_ARCHITECTURES[args.arch_name]
    
    model = create_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        cnn_config=cnn_config
    )
    t_created = time.time()

    history = model.fit(
        X_train_main, y_train_main,
        epochs=20, 
        verbose=0,
        validation_data=(X_val, y_val),
        batch_size=1024
    )
    t_fit = time.time()
    
    val_accuracy = max(history.history['val_accuracy'])
    train_accuracy = max(history.history['accuracy'])

  
    result_row = [
        args.run_id, args.arch_name,
        train_accuracy, val_accuracy, t_fit - t_created
    ]
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

except Exception as e:
    print(f"!!! ERROR on run {args.run_id} (Arch={args.arch_name}): {e}")
    result_row = [args.run_id, args.arch_name, 0.0, 0.0, 0.0]
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

finally:
    del model
    del history
    K.clear_session()
    gc.collect()