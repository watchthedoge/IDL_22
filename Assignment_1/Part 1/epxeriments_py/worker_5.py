
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
from keras import regularizers 
import time
import gc
import csv
import argparse
import numpy as np
import ast 


def create_cnn(input_shape, num_classes, cnn_config):
    

    conv_params_list = cnn_config['conv_params_list']
    pool_layer_template = cnn_config['pool_layer_template']
    pool_flags_list = cnn_config['pool_flags_list']
    dropout_rates_list = cnn_config['dropout_rates_list']
    dense_layer_nodes = cnn_config['dense_layer_nodes']
    optimizer = cnn_config['optimizer']

    l1_reg = cnn_config.get('l1_reg', 0.0)
    l2_reg = cnn_config.get('l2_reg', 0.0)

    kernel_reg = None
    if l1_reg > 0.0 or l2_reg > 0.0:
        kernel_reg = regularizers.L1L2(l1=l1_reg, l2=l2_reg)

    model = Sequential()
    
 
    first_conv_params = conv_params_list[0]
    filters, kernel_size, strides = first_conv_params
    model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu',
                     input_shape=input_shape, padding='valid',
                     kernel_regularizer=kernel_reg))


    if pool_flags_list[0]:
        model.add(pool_layer_template)
        model.add(Dropout(dropout_rates_list[0]))

    for i in range(1, len(conv_params_list)):
        filters, kernel_size, strides = conv_params_list[i]
        model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu',
                         kernel_regularizer=kernel_reg)) 
        if pool_flags_list[i]:
            model.add(pool_layer_template)
            model.add(Dropout(dropout_rates_list[i]))
            

    model.add(Flatten())
    model.add(Dense(dense_layer_nodes, activation='relu',
                    kernel_regularizer=kernel_reg)) 
    model.add(Dropout(dropout_rates_list[-1]))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


parser = argparse.ArgumentParser()

parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--conv_dropout", type=float, required=True)
parser.add_argument("--dense_dropout", type=float, required=True)
parser.add_argument("--l1_reg", type=float, required=True)
parser.add_argument("--l2_reg", type=float, required=True)
parser.add_argument("--output_file", type=str, required=True)

parser.add_argument("--conv_params_str", type=str, required=True)
parser.add_argument("--pool_flags_str", type=str, required=True)
parser.add_argument("--dense_nodes", type=int, required=True)
parser.add_argument("--base_lr", type=float, required=True)
parser.add_argument("--epochs", type=int, required=True)
args = parser.parse_args()


X_train_main = np.load('data_X_train.npy')
y_train_main = np.load('data_y_train.npy')
X_val = np.load('data_X_val.npy')
y_val = np.load('data_y_val.npy')
num_classes = 10
input_shape = (28, 28, 1)


try:
    t_start = time.time()
 
    

    conv_params_list = ast.literal_eval(args.conv_params_str)
    pool_flags_list = ast.literal_eval(args.pool_flags_str)
    

    dropout_rates_list = []

    if pool_flags_list[0]:
         dropout_rates_list.append(args.conv_dropout) 
    else:
         dropout_rates_list.append(0.0) 
            
    if pool_flags_list[1]:
         dropout_rates_list.append(args.conv_dropout) 
    else:
         dropout_rates_list.append(0.0)
    
   
    dropout_rates_list.append(args.dense_dropout)
    
    cnn_config = {
        'conv_params_list': conv_params_list,
        'pool_layer_template': MaxPooling2D(pool_size=(2, 2)),
        'pool_flags_list': pool_flags_list,
        'dropout_rates_list': dropout_rates_list,
        'dense_layer_nodes': args.dense_nodes,
        'optimizer': optimizers.Adam(learning_rate=args.base_lr),
        'l1_reg': args.l1_reg,
        'l2_reg': args.l2_reg
    }
    
    model = create_cnn(input_shape, num_classes, cnn_config)
    t_created = time.time()

    history = model.fit(
        X_train_main, y_train_main,
        epochs=args.epochs,
        verbose=0,
        validation_data=(X_val, y_val),
        batch_size=1024
    )
    t_fit = time.time()
    
    val_accuracy = max(history.history['val_accuracy'])
    train_accuracy = max(history.history['accuracy'])


    result_row = [
        args.run_id, args.conv_dropout, args.dense_dropout, args.l1_reg, args.l2_reg,
        train_accuracy, val_accuracy, t_fit - t_created
    ]
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

except Exception as e:
    print(f"!!! ERROR on run {args.run_id} (ConvDrop={args.conv_dropout}, L1={args.l1_reg}): {e}")
    result_row = [args.run_id, args.conv_dropout, args.dense_dropout, args.l1_reg, args.l2_reg, 0.0, 0.0, 0.0]
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

finally:

    del model
    del history
    K.clear_session()
    gc.collect()