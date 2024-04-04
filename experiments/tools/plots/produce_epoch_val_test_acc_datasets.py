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

datasets = ['ogbar'] 
datasets = ['reddit', 'ogbpr', 'cora', 'ogbar']
models = [ 'gcn_4', 'sage_4', 'gat_4'] 
#models = [ 'gcn_2', 'sage_2', 'gat_2', 'gcn_3', 'sage_3', 'gat_3']
#legends = ['refer-1-val', 'refer-1-test', 'refer-2', 'refer-2', 'refer-3', 'refer-3']
#legends = ['no-dyn-1-val', 'no-dyn-1-test', 'no-dyn-2', 'no-dyn-2', 'no-dyn-3', 'no-dyn-3']
#legends = ['dyn05-1-val', 'dyn05-1-test', 'dyn05-2', 'dyn05-2', 'dyn05-3', 'dyn05-3']
legends = ['run-1-val', 'run-1-test', 'val-2', 'test-2', 'val-3', 'test-3']
BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]

fig, axs = plt.subplots(4, 3, figsize=(14, 6))

for i in range(4):
    for j in range(3):
        axs[i, j].set_title(f"{datasets[i]}_16_{models[j]}")
        ds    = datasets[i]
        model = models[j]
        tag = ""
        if True:
        #if ds == 'ogbpr':
        #    tag = "withselfloop_xno_part_bias"
        #else:
            tag = "metis_xno_part_bias"

        run_50_dft = f"../../logs/train_{ds}_16_16_metis_{model}_dft.log"
        run_50_bdr = f"../../logs/trainctrl_{ds}_16_16_{tag}_050x_{model}_bdr.log"
 
        file_s = run_50_bdr
        test_acc_res = f"{ds} {model} "
        acc_l = []
        try:
            res_val, res_test = handle_epoch_val_test_acc(file_s)

            for rep in range(1,3):
                refer_v, refer_t = res_val[rep], res_test[rep]
                try:
                    x, y = list(zip(*refer_v.items()))
                    axs[i, j].plot(x,y,color[rep-1], linewidth=0.5, marker='o')
                except:
                    print("except-1:", file_s)
                try:
                    x, y = list(zip(*refer_t.items()))
                    axs[i, j].plot(x,y,color[rep-1], linewidth=0.5, marker='^')
                    acc_l.append(np.max(y))
                except:
                    print("except-2:", file_s)
            test_acc_res += f"{np.mean(acc_l):.6f} {np.std(acc_l):.6f} " + " ".join( [ f"{acc:.6f}" for acc in acc_l] )
            print(test_acc_res)
        except Exception as e:
            print("parse error: ", file_s, " exception: ", e)

        axs[i, j].set(xlabel='Epoch Number', ylabel='Accuracy')
        if i + j == 0:
            axs[i, j].legend(legends, loc='best', shadow=False, fontsize="10", markerscale=1.0)

fig.tight_layout()
#plt.savefig("epoch_validation_test_accuracy_dyn50_metis_more_datasets.png")

#df_dump(f"{model}-{layer}", df)
