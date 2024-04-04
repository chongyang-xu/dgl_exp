#!/usr/bin/python3

import subprocess # for run command
import glob       # for enumetating files
import os         # create directory

import re           # for regex
import numpy as np
import pandas as pd # for data processing
import tabulate     # convert to markdown

import multiprocessing as mp
from handle_log import handle_train_time, handle_train_acc, handle_val_test_acc

def shell_or(cmd, default=None):
# https://stackoverflow.com/questions/4256107/running-bash-commands-in-python
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

################################
# print val_test_acc
################################
data = {
        "model"  : [],
        "config" : [],
        "val-4" : [],
        "test-4" : [],
        "val-16" : [],
        "test-16" : [],
        "val-64" : [],
        "test-64" : [],
        "val-128" : [],
        "test-128" : [],
        "val-256" : [],
        "test-256" : [],
}

model = 'gcn'
layer = 2

for n_part in [4, 16, 64, 128, 256]:
    f0=f"../../../experiments/logs/train_ogbpr_{n_part}_{n_part}_random_{model}_{layer}_dft.log"
    f1=f"../../../experiments/logs/train_ogbpr_{n_part}_{n_part}_metis_{model}_{layer}_dft.log"
    f2=f"../../../experiments/logs/trainctrl_ogbpr_{n_part}_{n_part}_test_{model}_{layer}_bdr.log"
    
    for f in [f0, f1, f2]:
        v, t = handle_val_test_acc(f, n_part) 
        if n_part == 4:
            data['model'].append(f"{model}-{layer}")
            data['config'].append(f.split('_')[4])
        data[f"val-{n_part}"].append(v)
        data[f"test-{n_part}"].append(t)

df = pd.DataFrame(data)
print(df)
#df_dump(f"{model}-{layer}", df)
