#!/usr/bin/python3

import subprocess # for run command
import glob       # for enumetating files
import os         # create directory

import re           # for regex
import numpy as np
import pandas as pd # for data processing
import tabulate     # convert to markdown

import multiprocessing as mp

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

####################################
def clean_row_break_duplicate(container, in_str:str, tag: str):
    if in_str.count(tag) < 1:
        raise Exception('no valid data found')
    elif in_str.count(tag) == 1:
        container.append(in_str)
    #elif in_str.count(tag) == 2:
    else:
        t = in_str.split(tag)
        for i in range(1, len(t)):
            container.append(tag + t[i])

def handle_train_time(f_name: str, n_part: int = -1, n_epoch: int  = -1):
    et_cmd = f"cat {f_name} | grep epoch_ | grep epoch | grep 'part|0000|' | sort"
    maybe = shell_or(et_cmd)
    if maybe is None:
        return np.nan

    et_out = maybe.stdout.strip().split('\n')
    cleaned_et_out = []
    for r in et_out:
        clean_row_break_duplicate(cleaned_et_out, r, 'epoch_|') # must be step_|
    cleaned_et_out.sort()

    result = []
    for i in cleaned_et_out:
        if i.find("part|0000|") != -1:
            result.append(i)

    ets = []
    epoch_idx = 2 if f_name.find("trainctrl") !=-1 else 2
    et_idx    = 8 if f_name.find("trainctrl") !=-1 else 6
    for r in result:
        es = r.split('|')
        epoch = float(es[epoch_idx])
        et = float(es[et_idx])
        ets.append(et)

    return np.sum(ets)


def handle_train_acc(f_name: str, n_part: int = -1, n_epoch: int = -1):
    # handle per stop information
    step_cmd = f"cat {f_name} | grep step_ | grep 'part|0000|' | sort | uniq"
    maybe = shell_or(step_cmd)
    if maybe is None:
        return []

    step_out = maybe.stdout.strip().split('\n')
    cleaned_step_out = []
    for r in step_out:
        clean_row_break_duplicate(cleaned_step_out, r, 'step_|') # must be step_|
    cleaned_step_out.sort()

    epoch_idx =  2 if f_name.find("trainctrl") !=-1 else 2
    step_idx  =  6 if f_name.find("trainctrl") !=-1 else 4
    t_acc_idx = 12 if f_name.find("trainctrl") !=-1 else 10  

    def handle_step_row(in_str):
        t = in_str.split('|')
        epoch    = t[epoch_idx]
        step     = t[step_idx]
        t_acc    = float(t[t_acc_idx])
        return epoch, step, t_acc
   
    o_e = "0000" # old epoch number
    o_s = "0000" # old step number

    epoch_acc = []
    cur_acc = []
    for r in cleaned_step_out:
        e, s, a = handle_step_row(r)
        if e != o_e:
            o_e = e
            epoch_acc.append(np.mean(cur_acc))
            cur_acc.clear()
        cur_acc.append(a)

    if len(cur_acc) > 0:
        epoch_acc.append(np.mean(cur_acc))
        cur_acc.clear()

    return epoch_acc

def handle_train_loss(f_name: str, n_part: int = -1, n_epoch: int = -1):
    # handle per stop information
    step_cmd = f"cat {f_name} | grep 'step_|' | grep 'part|' | sort | uniq"
    maybe = shell_or(step_cmd)
    if maybe is None:
        return []

    step_out = maybe.stdout.strip().split('\n')
    cleaned_step_out = []
    for r in step_out:
        try:
            clean_row_break_duplicate(cleaned_step_out, r, 'step_|') # must be step_|
        except:
            pass
    cleaned_step_out.sort()

    epoch_idx =  2 if f_name.find("trainctrl") !=-1 else 2
    step_idx  =  6 if f_name.find("trainctrl") !=-1 else 4
    t_loss_idx = 10 if f_name.find("trainctrl") !=-1 else 8

    def handle_step_row(in_str):
        t = in_str.split('|')
        epoch    = t[epoch_idx]
        step     = t[step_idx]
        try:
            t_loss    = float(t[t_loss_idx])
        except:
            t_loss    = np.nan
        return epoch, step, t_loss
   
    o_e = "0000" # old epoch number
    o_s = "0000" # old step number

    epoch_loss = []
    cur_loss = []
    for r in cleaned_step_out:
        e, s, a = handle_step_row(r)
        if e != o_e:
            o_e = e
            epoch_loss.append(np.mean(cur_loss))
            cur_loss.clear()
        if a is not np.nan:
            cur_loss.append(a)

    if len(cur_loss) > 0:
        epoch_loss.append(np.mean(cur_loss))
        cur_loss.clear()

    return epoch_loss

def handle_val_test_acc(f_name: str, n_part: int = -1, n_epoch: int = -1):
    cmd = f"cat {f_name} | grep 'infer_|' | grep 'part|0000'"
    maybe = shell_or(cmd)
    if maybe is None:
        return np.nan, np.nan

    out = maybe.stdout.strip().split('\n')
    cleaned_out = []
    for r in out:
        clean_row_break_duplicate(cleaned_out, r, 'infer_|') # must be step_|
    cleaned_out.sort()

    def val_test_acc(str_row):
        p = str_row.split('|')
        return float(p[6]), float(p[8])

    vals = []
    tests = []
    for r in cleaned_out:
        v, t = val_test_acc(r)
        vals.append(v)
        tests.append(t)
    
    #return np.mean(vals), np.std(vals), np.mean(tests), np.std(tests)
    return vals[0], tests[0]

def handle_epoch_val_test_acc(f_name: str, n_part: int=-1, n_epoch: int=-1):
    # validition happends for epoch 99, 199, 299, 399, 499
    res_val = {}
    res_test = {}
    for i in range(1, 4):
        res_val[i] = { 99: 0.0, 199: 0.0, 299: 0.0, 399: 0.0, 499: 0.0 }
        res_test[i] = { 99: 0.0, 199: 0.0, 299: 0.0, 399: 0.0, 499: 0.0 }

    cmd = f"cat {f_name} | grep 'infer_|' "
    maybe = shell_or(cmd)
    if maybe is None:
        print("None: ", f_name)
        return res_val, res_test

    out = maybe.stdout.strip().split('\n')
    cleaned_out = []
    try:
        start_flag = False
        for r in out:
            if r.startswith("infer_|epoch|0499") and not start_flag:
                continue
            else:
                start_flag = True
            clean_row_break_duplicate(cleaned_out, r, 'infer_|') # must be step_|
    except e:
        print("dedup: ", e)
        print("dedup: ", f_name)
        return res_val, res_test

    def val_test_acc(str_row):
        p = str_row.split('|')
        epoch = int(p[2])
        v_acc = float(p[6])
        t_acc = float(p[8])
        return epoch, v_acc, t_acc

    idx = 1
    cnt = 0
    for r in cleaned_out:
        e, v, t = val_test_acc(r)
        res_val[idx][e] = v
        res_test[idx][e] = t
        cnt += 1
        if cnt == 5:
            cnt = 0
            idx += 1
        if idx >= 4:
            break
    return res_val, res_test

# ##################
# plots
#

