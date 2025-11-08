import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
from sklearn.model_selection import train_test_split
import time
import subprocess
import sys
import csv
import keras
from keras.datasets import fashion_mnist
import numpy as np


print("Loading and preprocessing data for CNN...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


X_train_main, X_val, y_train_main, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=67
)

if not os.path.exists('data_X_train.npy'):
    np.save('data_X_train.npy', X_train_main)
    np.save('data_y_train.npy', y_train_main)
    np.save('data_X_val.npy', X_val)
    np.save('data_y_val.npy', y_val)
    print("Data files saved.")
else:
    print("Data files found.")

optimizer_list = ["Adadelta", "Adam", "SGD_Momentum", "RMSprop"]
lr_list = [1.0, 0.01, 0.001, 0.0001] 


RUNS_PER_SET = 3          

total_param_sets = (
    len(optimizer_list) *
    len(lr_list)
)
current_param_index = 0


output_filename = 'cnn_baseline_opt_search.csv'
csv_header = [
    'run_id', 'optimizer', 'lr',
    'train_accuracy', 'val_accuracy', 'fit_time_sec'
]
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

print("--- Starting Experiment Manager (CNN Baseline) ---")
my_env = os.environ.copy() 

for optimizer_name in optimizer_list:
    for lr in lr_list:
        for run_id in range(1, RUNS_PER_SET + 1):
            
      
            command = [
                sys.executable,
                'worker_3.py', 
                '--run_id', str(run_id),
                '--optimizer_name', optimizer_name,
                '--lr', str(lr),
                '--output_file', output_filename
            ]
            
          
            subprocess.run(command, env=my_env)

        current_param_index += 1
        percent = (current_param_index / total_param_sets) * 100
        print(
            f"Completed set {current_param_index}/{total_param_sets} "
            f"({percent:.1f}%): Optimizer={optimizer_name}, LR={lr}"
        )


print(f"\n--- Experiment complete. Results saved to {output_filename} ---")
results_df = pd.read_csv(output_filename)

print("\n--- Top Results (Raw) ---")
print(results_df.sort_values(by='val_accuracy', ascending=False).head(20))

print("\n--- Best Performing Optimizers (Averaged) ---")
avg_results = results_df.groupby(
    ['optimizer', 'lr']
)[['train_accuracy', 'val_accuracy']].mean()
print(avg_results.sort_values(by='val_accuracy', ascending=False).head(10))