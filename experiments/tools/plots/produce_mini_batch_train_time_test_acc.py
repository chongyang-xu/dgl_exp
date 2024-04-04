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
parts = [ 4 , 16, 64, 256 ]

legends = [ 'DGL-Random', 'DGL-Metis', 'Triskelion'] #, 'NeutronStar', 'BNS-GCN' ]

table1= {
        'model': [],
        'systems': [],
}
table2= {
        'model': [],
        'systems': [],
}
cols1 = []
cols2 = []

for p in parts:
    cols1.append(f"time-{p}")
for p in parts:
    cols2.append(f"acc-{p}")

for c in cols1:
    table1[c] = []
for c in cols2:
    table2[c] = []

for MODEL in models:
    for cfg in legends:

        # Train time
        for N_PART in parts:
            if cfg == legends[0]:
                f = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_random_{MODEL}_dft.log"
            elif cfg == legends[1]:
                f = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_metis_{MODEL}_dft.log"
            else:
                assert cfg == legends[2]
                f = f"../../logs/trainctrl_ogbpr_{N_PART}_{N_PART}_20fmode4_{MODEL}_bdr.log"
            if N_PART == 4:
                table1['model'].append(MODEL)
                table1['systems'].append(cfg)
            try:
                tot_t = handle_train_time(f)
                #tot_t = tot_t / 500.0
            except:
                tot_t = -1
                
            table1[f"time-{N_PART}"].append(tot_t)

for MODEL in models:
    for cfg in legends:
        # test accuracy
        for N_PART in parts:
            if cfg == legends[0]:
                f = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_random_{MODEL}_dft.log"
            elif cfg == legends[1]:
                f = f"../../logs/train_ogbpr_{N_PART}_{N_PART}_metis_{MODEL}_dft.log"
            else:
                assert cfg == legends[2]
                f = f"../../logs/trainctrl_ogbpr_{N_PART}_{N_PART}_20fmode4_{MODEL}_bdr.log"
            if N_PART == 4:
                table2['model'].append(MODEL)
                table2['systems'].append(cfg)
            try:
                _, t_acc = handle_val_test_acc(f)
            except:
                t_acc = -1
            table2[f"acc-{N_PART}"].append(t_acc)

df = pd.DataFrame(table1)
df=df.round(1)
df_dump(f"mini_batch_train_time", df)
print(df)
df = pd.DataFrame(table2)
df=df.round(4)
df_dump(f"mini_batch_test_accuracy", df)
print(df)
