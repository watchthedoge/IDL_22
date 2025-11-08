
import os
import warnings


# Suppress all warnings and TF info logs
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

# Load Data and Save it to Disk for the Worker 
print("Loading and saving data for worker processes...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

X_train_main, X_val, y_train_main, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=67
)
# Save to .npy files
np.save('data_X_train.npy', X_train_main)
np.save('data_y_train.npy', y_train_main)
np.save('data_X_val.npy', X_val)
np.save('data_y_val.npy', y_val)
print("Data saved.")

#Define Parameter Lists 
nodes_list = [64, 128, 256, 512]
dropout_list = [0, 0.2, 0.4, 0.5]
lr_list = [1.0, 0.1, 0.01, 0.001, 0.0005]
hl_list = [1, 2, 3, 4]

runs_per_set = 5
total_param_sets = (
    len(nodes_list) *
    len(dropout_list) *
    len(lr_list) *
    len(hl_list)
)
current_param_index = 0

#Set up the CSV file 
output_filename = 'dnn_grid_search_results_GPU.csv'
csv_header = [
    'run_id', 'hidden_layers', 'nodes', 'dropout', 'lr',
    'val_accuracy', 'fit_time_sec'
]
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

print("--- Starting Experiment Manager ---")

#The Loop 
for hl in hl_list:
    for nodes in nodes_list:
        for dropout in dropout_list:
            for lr in lr_list:
                for run_id in range(1, runs_per_set + 1):
                    
                    # Build the command to run the worker script
                    command = [
                        sys.executable,
                        'worker_1.py',
                        '--run_id', str(run_id),
                        '--hl', str(hl),
                        '--nodes', str(nodes),
                        '--dropout', str(dropout),
                        '--lr', str(lr),
                        '--output_file', output_filename
                    ]
                    
                    # Run the subprocess
                    # This runs the command and waits for it to finish.
                    # A new, clean 0MB process is created.
                    # It leaks up to 7GB.
                    # It finishes, and the OS destroys it.
                    # Get a copy of the parent's (this script's) environment
                    my_env = os.environ.copy()

                    # Explicitly pass that environment to the child process
                    subprocess.run(command, env=my_env)

                #Update progress
                current_param_index += 1
                percent = (current_param_index / total_param_sets) * 100
                print(
                    f"Completed param set {current_param_index}/{total_param_sets} "
                    f"({percent:.1f}%): hl={hl}, nodes={nodes}, dropout={dropout}, lr={lr}"
                )

#Read results back for analysis 
print(f"\n--- Experiment complete. Results saved to {output_filename} ---")
results_df = pd.read_csv(output_filename)

print("\n--- Top Results (Raw) ---")
print(results_df.sort_values(by='val_accuracy', ascending=False).head(20))