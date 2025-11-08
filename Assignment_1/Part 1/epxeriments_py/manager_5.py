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

if not os.path.exists('data_X_train.npy'):
    print("Data files (e.g., data_X_train.npy) not found.")
    print("Please run the Phase 1 'run_experiment_cnn_phase1.py' script first to generate these.")
    sys.exit()
else:
    print("Data files found.")


BASE_CONV_PARAMS_STR = "[(64, (3, 3), (1, 1)), (128, (3, 3), (1, 1))]"
BASE_POOL_FLAGS_STR = "[False, True]"
BASE_DENSE_NODES = 256
BASE_LR = 0.01


conv_block_2_dropout_list = [0.1, 0.25, 0.4]
dense_dropout_list = [0.3, 0.5, 0.7]

reg_list = [
    (0.0, 0.0),      # No regularization (baseline)
    (0.001, 0.0),    # L1 only
    (0.0, 0.001),    # L2 only
]


RUNS_PER_SET = 3
EPOCHS = 20
total_param_sets = (
    len(conv_block_2_dropout_list) *
    len(dense_dropout_list) *
    len(reg_list)
)
current_param_index = 0


output_filename = 'cnn_phase2_regularization_search.csv'
csv_header = [
    'run_id', 'conv_dropout', 'dense_dropout', 'l1_reg', 'l2_reg',
    'train_accuracy', 'val_accuracy', 'fit_time_sec'
]
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

print("--- Starting Experiment Manager (CNN Phase 2: Regularization) ---")
my_env = os.environ.copy()


for conv_dropout in conv_block_2_dropout_list:
    for dense_dropout in dense_dropout_list:
        for l1, l2 in reg_list:
            for run_id in range(1, RUNS_PER_SET + 1):
                
         
                command = [
                    sys.executable,
                    'worker_5.py', 
                    '--run_id', str(run_id),
                    '--conv_dropout', str(conv_dropout),
                    '--dense_dropout', str(dense_dropout),
                    '--l1_reg', str(l1),
                    '--l2_reg', str(l2),
                    '--output_file', output_filename,
                    '--conv_params_str', BASE_CONV_PARAMS_STR,
                    '--pool_flags_str', BASE_POOL_FLAGS_STR,
                    '--dense_nodes', str(BASE_DENSE_NODES),
                    '--base_lr', str(BASE_LR),
                    '--epochs', str(EPOCHS)
                ]
                
                subprocess.run(command, env=my_env)


            current_param_index += 1
            percent = (current_param_index / total_param_sets) * 100
            print(
                f"Completed set {current_param_index}/{total_param_sets} "
                f"({percent:.1f}%): ConvDrop={conv_dropout}, DenseDrop={dense_dropout}, L1={l1}, L2={l2}"
            )


print(f"\n--- Experiment complete. Results saved to {output_filename} ---")
results_df = pd.read_csv(output_filename)

print("\n--- Top Results (Raw) ---")
print(results_df.sort_values(by='val_accuracy', ascending=False).head(20))

print("\n--- Best Performing Regularization (Averaged) ---")
avg_results = results_df.groupby(
    ['conv_dropout', 'dense_dropout', 'l1_reg', 'l2_reg']
)[['train_accuracy', 'val_accuracy']].mean()
print(avg_results.sort_values(by='val_accuracy', ascending=False).head(10))