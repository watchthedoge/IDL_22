
import os
import warnings

# Suppress standard Python warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import tensorflow as tf 
from keras import optimizers
import time
import gc
import csv
import argparse
import numpy as np



def create_a_dnn_sequential(input_shape, num_classes, dnn_config):
    model = Sequential()
    if len(input_shape) > 1:
        model.add(Flatten(input_shape=input_shape))
    else:
        model.add(keras.layers.Input(shape=input_shape))
    
    # Rest of model function is unchanged
    num_hidden_layers = dnn_config['num_hidden_layers']
    nodes = dnn_config['nodes_per_layer']
    activation = dnn_config.get('activation', 'relu')
    initializer = dnn_config.get('initializer', 'glorot_uniform')
    dropout_rate = dnn_config.get('dropout_rate', 0.0)
    
    for _ in range(num_hidden_layers):
        model.add(Dense(units=nodes, activation=activation, kernel_initializer=initializer))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))
            
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=dnn_config['loss'],
                  optimizer=dnn_config['optimizer'],
                  metrics=['accuracy'])
    return model

# Define and Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--hl", type=int, required=True)
parser.add_argument("--nodes", type=int, required=True)
parser.add_argument("--dropout", type=float, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

#Load Pre-split Data
X_train_main = np.load('data_X_train.npy')
y_train_main = np.load('data_y_train.npy')
X_val = np.load('data_X_val.npy')
y_val = np.load('data_y_val.npy')
num_classes = 10

# Run the single experiment
try:
    t_start = time.time()
    config = {
        'optimizer': optimizers.Adam(learning_rate=args.lr),
        'loss': 'categorical_crossentropy',
        'num_hidden_layers': args.hl,
        'nodes_per_layer': args.nodes,
        'activation': 'relu',
        'initializer': 'he_uniform',
        'dropout_rate': args.dropout
    }
    model = create_a_dnn_sequential((28, 28), num_classes, config)
    t_created = time.time()

    history = model.fit(
        X_train_main, y_train_main,
        epochs=15,
        verbose=0, 
        validation_data=(X_val, y_val),
        batch_size=1024
    )
    t_fit = time.time()
    val_accuracy = max(history.history['val_accuracy'])

    #Append the single result to the CSV
    result_row = [
        args.run_id, args.hl, args.nodes, args.dropout, args.lr,
        val_accuracy, t_fit - t_created
    ]
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

except Exception as e:
    # see critical errors
    print(f"!!! ERROR on run {args.run_id} (hl={args.hl}, nodes={args.nodes}): {e}")
    result_row = [args.run_id, args.hl, args.nodes, args.dropout, args.lr, 0.0, 0.0]
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

finally:
    del model
    del history
    K.clear_session()
    gc.collect()