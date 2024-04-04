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
import matplotlib.pyplot as plt

models = ['sage_2', 'gcn_2', 'gat_2',  'sage_3', 'gcn_3' ]
parts = [ 4 , 16, 64]
legends = [ 'DGL-Random', 'DGL-Metis', 'Triskelion'] #, 'NeutronStar', 'BNS-GCN' ]
BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]
fig, axs = plt.subplots(3, 5, figsize=(14, 6))
for i in range(3):
    for j in range(5):
        axs[i, j].set_title(f"{models[j]}, {parts[i]} partitions")
        MODEL=models[j]
        N_PART=parts[i]
        f0 = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_random_{MODEL}_dft.log"
        f1 = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_metis_{MODEL}_dft.log"
        f2 = f"../../logs/trainctrl_ogbpr_{N_PART}_{N_PART}_20fmode4_{MODEL}_bdr.log"
        fs = [ f0, f1, f2]
        for idx,f in enumerate(fs):
            try:
                a0 = handle_train_acc(f)
            except:
                a0 = [ (0, 0)]
                print(f"Error: {f}")
            a0 =  [ (epoch, v) for epoch, v in enumerate(a0) ]
            #if j == 0 and i == 2:
            #    print(MODEL, N_PART, a0)

            l = list(zip(*a0))
            x, y = list(l[0]), list(l[1])
            axs[i, j].plot(x,y,color[idx], linewidth=2.0)
            axs[i, j].set(xlabel='Epoch Number', ylabel='Train Accuracy')
            if i+ j == 0:
                axs[i, j].legend(legends, loc='best', shadow=False, fontsize="10", markerscale=2.0)

fig.tight_layout()
plt.savefig("mini_batch_train_accuracy_curve.pdf")


