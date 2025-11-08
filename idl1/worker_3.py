
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import time
import gc
import csv
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--optimizer_name", type=str, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()


X_train_main = np.load('data_X_train.npy')
y_train_main = np.load('data_y_train.npy')
X_val = np.load('data_X_val.npy')
y_val = np.load('data_y_val.npy')
num_classes = 10
input_shape = (28, 28, 1) 
epochs_from_repo = 12

batch_size_efficient = 1024 


optimizer_obj = None
if args.optimizer_name == "Adam":
    optimizer_obj = optimizers.Adam(learning_rate=args.lr)
elif args.optimizer_name == "SGD_Momentum":
    optimizer_obj = optimizers.SGD(learning_rate=args.lr, momentum=0.9)
elif args.optimizer_name == "RMSprop":
    optimizer_obj = optimizers.RMSprop(learning_rate=args.lr)
elif args.optimizer_name == "Adadelta":
    optimizer_obj = optimizers.Adadelta(learning_rate=args.lr)
else:
    raise ValueError(f"Unknown optimizer name: {args.optimizer_name}")


try:
    t_start = time.time()
    

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer_obj, 
                  metrics=['accuracy'])

    
    t_created = time.time()

    history = model.fit(
        X_train_main, y_train_main,
        epochs=epochs_from_repo, # Use 12 epochs from repo
        verbose=0,
        validation_data=(X_val, y_val),
        batch_size=batch_size_efficient
    )
    t_fit = time.time()
    
    val_accuracy = max(history.history['val_accuracy'])
    train_accuracy = max(history.history['accuracy'])


    result_row = [
        args.run_id, args.optimizer_name, args.lr,
        train_accuracy, val_accuracy, t_fit - t_created
    ]
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

except Exception as e:
    print(f"!!! ERROR on run {args.run_id} (Opt={args.optimizer_name}, LR={args.lr}): {e}")
    result_row = [args.run_id, args.optimizer_name, args.lr, 0.0, 0.0, 0.0]
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

finally:

    del model
    del history
    K.clear_session()
    gc.collect()