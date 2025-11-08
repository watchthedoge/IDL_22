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
import numpy as np

#CHECK FOR DATA
if not os.path.exists('data_X_train.npy'):
    print("Data files (e.g., data_X_train.npy) not found.")
    print("Please run the Phase 1 'run_experiment.py' script first to generate these.")
    sys.exit()
else:
    print("Data files found.")

BEST_HL = 4
BEST_NODES = 512
BEST_DROPOUT = 0.0
BEST_LR = 0.001 

optimizer_list = ["Adam", "SGD_Momentum", "RMSprop"]
activation_list = ["relu", "tanh"]
initializer_list = ["glorot_uniform", "he_uniform"]


#test (L1, L2) pairs
reg_list = [
    (0.0, 0.0),      # No regularization (baseline)
    (0.001, 0.0),    # L1 only
    (0.0, 0.001),    # L2 only
    (0.001, 0.001)   # L1 + L2
]

runs_per_set = 3
total_param_sets = (
    len(optimizer_list) *
    len(activation_list) *
    len(initializer_list) *
    len(reg_list) 
)
current_param_index = 0

#Set up the CSV file
output_filename = 'dnn_part2_search_results.csv'
csv_header = [
    'run_id', 'optimizer', 'activation', 'initializer', 'l1_reg', 'l2_reg', 
    'train_accuracy', 'val_accuracy', 'fit_time_sec'
]
with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

print("--- Starting Experiment Manager (Phase 2 - Now with Regularization) ---")
my_env = os.environ.copy() 

# The Loop 
for optimizer_name in optimizer_list:
    for activation in activation_list:
        for initializer in initializer_list:
            for l1, l2 in reg_list: 
                for run_id in range(1, runs_per_set + 1):
                    
                  
                    command = [
                        sys.executable,
                        'worker_2.py',
                        '--run_id', str(run_id),
                        '--optimizer_name', optimizer_name,
                        '--activation', activation,
                        '--initializer', initializer,
                        '--l1_reg', str(l1), 
                        '--l2_reg', str(l2),
                        '--output_file', output_filename,
                        '--best_hl', str(BEST_HL),
                        '--best_nodes', str(BEST_NODES),
                        '--best_dropout', str(BEST_DROPOUT),
                        '--best_lr', str(BEST_LR)
                    ]
                    
                    with open(os.devnull, 'w') as FNULL:
                        subprocess.run(
                            command, 
                            env=my_env,
                            stdout=subprocess.PIPE,
                            stderr=FNULL
                        )

             
                current_param_index += 1
                percent = (current_param_index / total_param_sets) * 100
                print(
                    f"Completed set {current_param_index}/{total_param_sets} "
                    f"({percent:.1f}%): Opt={optimizer_name}, Act={activation}, "
                    f"Init={initializer}, L1={l1}, L2={l2}" 
                )

#Read results back for analysis
print(f"\n--- Experiment complete. Results saved to {output_filename} ---")
results_df = pd.read_csv(output_filename)

print("\n--- Top Results (Raw) ---")
print(results_df.sort_values(by='val_accuracy', ascending=False).head(20))

#Aggregate results
print("\n--- Best Performing Combinations (Averaged) ---")
avg_results = results_df.groupby(
    ['optimizer', 'activation', 'initializer', 'l1_reg', 'l2_reg']
)[['train_accuracy', 'val_accuracy']].mean()
print(avg_results.sort_values(by='val_accuracy', ascending=False).head(10))