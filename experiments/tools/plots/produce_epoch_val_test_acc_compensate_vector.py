#!/usr/bin/python3

import subprocess # for run command
import glob       # for enumetating files
import os         # create directory

import re           # for regex
import numpy as np
import pandas as pd # for data processing
import tabulate     # convert to markdown

import multiprocessing as mp
from handle_log import handle_train_time, handle_train_acc, handle_val_test_acc, handle_epoch_val_test_acc

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
import matplotlib.pyplot as plt

datasets = [ 'reddit']
config   = ['xno_part_bias_500x', 'xpart_bias_500x', 'xpart_biasx']

models = [ 'gcn_2', 'sage_2', 'gat_2' ]
#legends = ['refer-1-val', 'refer-1-test', 'refer-2', 'refer-2', 'refer-3', 'refer-3']
#legends = ['dyn50-1-val', 'dyn50-1-test', 'dyn50-2', 'dyn50-2', 'dyn50-3', 'dyn50-3']
legends = ['run-1-val', 'run-1-test', 'run-2-val', 'run-2-test', 'run-3-val', 'run-3-test']
BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]
fig, axs = plt.subplots(3, 3, figsize=(14, 8))

for i in range(3):
    for j in range(3):
        tag = "-part_bias -repartition" if i == 0 else "+part_bias -repartition" if i == 1 else "+part_bias +repartition"
        axs[i, j].set_title(f"{models[j]}_{tag}")
        refer  = f"../../logs/trainctrl_reddit_16_16_{config[i]}_{models[j]}_bdr.log"
 
        file_s = refer
        try:
            res_val, res_test = handle_epoch_val_test_acc(file_s)

            print(f"{tag} : {models[j]}")
            for rep in range(1,4):
                refer_v, refer_t = res_val[rep], res_test[rep]
                print(f"run-{rep}")
                try:
                    x, y = list(zip(*refer_v.items()))
                    axs[i, j].plot(x,y,color[rep-1], linewidth=0.5, marker='o')
                    #print("val: ", np.max(y), " ")
                except:
                    print("except-1:", file_s)
                try:
                    x, y = list(zip(*refer_t.items()))
                    axs[i, j].plot(x,y,color[rep-1], linewidth=0.5, marker='^')
                    print("tst: ", np.max(y), "\n")
                except:
                    print("except-2:", file_s)
        except:
            print("parse error: ", file_s)

        axs[i, j].set(xlabel='Epoch Number', ylabel='Accuracy')
        if i + j == 0:
            axs[i, j].legend(legends, loc='best', shadow=False, fontsize="10", markerscale=1.0)

fig.tight_layout()
plt.savefig("epoch_validation_test_accuracy_compensate_vector.png")

#df_dump(f"{model}-{layer}", df)
