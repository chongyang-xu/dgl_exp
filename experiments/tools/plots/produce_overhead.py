#!/usr/bin/python3

import subprocess # for run command
import glob       # for enumetating files
import os         # create directory

import re           # for regex
import numpy as np
import pandas as pd # for data processing
import tabulate     # convert to markdown

import multiprocessing as mp
from handle_log import handle_train_time, handle_train_acc, handle_val_test_acc, handle_train_loss

from plot_util import plot_lines, plot_bars

# https://stackoverflow.com/questions/4256107/running-bash-commands-in-python
def shell_or(cmd, default=None):
    try:
        normal = subprocess.run([cmd],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                            shell=True,
                            text=True)
        return normal
    except:
        print(f"FAIL  : {cmd}")
        return default

def df_dump(f_name, df, directory="./dump_result"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/{f_name}.latex", 'w') as f:
        f.write(df.to_latex())
    
    with open(f"{directory}/{f_name}.markdown", 'w') as f:
        f.write(df.to_markdown())
    
    with open(f"{directory}/{f_name}.html", 'w') as f:
        f.write(df.to_html())

# ##################
# plots
# ##################
BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]

import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = [ 'DGL-Random', 'DGL-Metis', 'Triskelion'] #, 'NeutronStar', 'BNS-GCN' ]
values1 = [2637.5, 1304.1, 823.2]
values2 = [0.0, 0.0, 164.6]
values3 = [73.2, 175.8, 732]

# Plotting the stacked bar chart
fig, ax = plt.subplots()
# Plotting the first set of bars
ax.bar(categories, values1, label='Mini-Batch Training')
# Plotting the second set of bars on top of the first
ax.bar(categories, values2, bottom=values1, label='Training Controller')
# Plotting the third set of bars on top of the first two
ax.bar(categories, values3, bottom=[i + j for i, j in zip(values1, values2)], label='Partition Time')

# Adding labels and legend
ax.set_xlabel('Systems')
ax.set_ylabel('Overall Training Time (seconds)')
ax.set_title('OGBN products dataset, sage-2, 16 partitions (4 gpus)')
ax.legend()

# Display the plot
plt.savefig("overhead.pdf")

