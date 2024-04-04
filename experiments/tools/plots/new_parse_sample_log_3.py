import numpy as np
import pandas as pd
import random

# convert string "[ xx, xxx, xx]" in into list
def string_to_list(list_string):
    a = list_string.replace('[', '')
    a = a.replace(']', '')
    aa = a.split(',')
    ret = list()
    for ele in aa:
        ret.append(int(ele.strip()))

    return ret

# count #freq of unique numbers into d
def count_freq(d, numbers):
    for num in numbers:
        if num in d:
            d[num] += 1
        else:
            d[num] = 1

#  input is list of #freq and input is sorted
def sorted_freq_to_accumulative(sorted_freq):
    if len(sorted_freq) == 0:
        return [0], [0]
    # cal the freq of the freq
    max_v = sorted_freq[-1]
    d = {int(0):int(0)}
    for i in range(1, max_v+1):
        d[i] = int(0)
    for v in sorted_freq:
        d[v] += int(1)
    
    # question: how many unique nodes are sampled at most X times?
    # (x, y) are dots
    res = sorted(d.items(), key=lambda item: item[0])
    x, y = zip(*res)
    x, y = list(x), list(y)
    for i in range(1, len(y)):
        y[i] = y[i-1] + y[i]

    return x, y

####################
#
# extract data
#
####################
def split_log_to_list_of_nodes(file_path):
    with open(file_path, 'r') as ff:
        file_content = ff.read()
    content = file_content.replace('tensor(', '')
    splits = content.strip().split('--------------------\n--------------------')
    return splits

def per_nodes_to_detail(d1, d2, d3, row, l_2=False, l_3=False):
    cuts = row.split('--------------------')
    l1 = cuts[0]
    l2 = ""
    l3 = ""
    if l_2:
        l2 = cuts[1]
    if l_3:
        l3 = cuts[2]

    l1_src = l1.split(')')[0].strip()
    l1_dst = l1.split(')')[1].strip()
    l1_src = sorted(string_to_list(l1_src))
    l1_dst = sorted(string_to_list(l1_dst))
    #count_freq(d1, l1_dst)
 
    if l_2:
        l2_src = l2.split(')')[0].strip()
        l2_dst = l2.split(')')[1].strip()
        l2_src = sorted(string_to_list(l2_src))
        l2_dst = sorted(string_to_list(l2_dst))
#        count_freq(d2, l2_dst)

    if l_3:
        l3_src = l3.split(')')[0].strip()
        l3_dst = l3.split(')')[1].strip()
        l3_src = sorted(string_to_list(l3_src))
        l3_dst = sorted(string_to_list(l3_dst))
#        count_freq(d3, l3_dst)

    return l1_src[0]

def choose_training_nodes(ds, layer, shuffle):
    selected = set()
    for rank in range(16):
        file_path = f"../sample_log/random_{ds}_{layer}_rank_{rank}_epoch_0_{shuffle}.txt"
        rows = split_log_to_list_of_nodes(file_path)
        d1 = {}
        d2 = {}
        d3 = {}
        while len(selected) < 5*(rank+1):
            r_idx = random.randint(0, len(rows)-1)
            src = per_nodes_to_detail(d1, d2, d3, rows[r_idx])
            selected.add(src)

    tag = "rtest"
    # ogbpr metis
    #prefix_sum = [0, 152094, 307866, 457285, 612970, 766751, 920640, 1076199, 1233313, 1382786, 1532218, 1681479, 1838472, 1987760, 2144387, 2293716, 99999999999999]
    # ogbpr random
    prefix_sum = [0, 153065, 305806, 458800, 611978, 764278, 916936, 1070334, 1223481, 1376612, 1529253, 1682638, 1835925, 1989128, 2141929, 2295613, 9999999999]
    # reddit random
    #prefix_sum = [0, 13624, 27095, 40642, 53943, 67444, 80909, 94278, 108044, 121556, 134820, 148130, 161818, 175327, 188718, 202368, 99999999999999]
    # reddit metis
    #prefix_sum = [0, 14121, 28217, 43203, 58194, 73188, 87786, 101903, 116320, 130736, 145723, 159760, 174126, 189080, 203770, 218491, 99999999999999]

    target_oids = set() 
    lid2oids = {}
    for i_rank in range(16):
        f  = f"/data/ds_pre/{ds}/chunking_16_{tag}/p{i_rank}.oid.bin"
        lid2oid = np.fromfile(f, dtype=np.int64)
        lid2oids[i_rank]=lid2oid

    #print(f"selected: {selected}")
    dyn = {
        # n_gid: 'rank', 'dft_id', 'dft_rank'
    }
    for n_gid in selected:
        rank = -1
        for i in range(16):
            if n_gid >= prefix_sum[i] and n_gid < prefix_sum[i+1]:
                rank = i
                break
        nnid = n_gid - prefix_sum[rank]
        try:
            print(f"dyn_gid: {n_gid}, dyn_rank: {rank}, dyn_lid: {nnid}, oid: {lid2oids[rank][nnid]} ")
            target_oids.add(lid2oids[rank][nnid])
            dyn[n_gid] = { 'rank': rank, 'oid': lid2oids[rank][nnid], 'dft_id':-1 }
        except:
            print(f"dyn_gid: {n_gid}, dyn_rank: {rank}, dyn_lid: {nnid}, oid: -1 ")
            target_oids.add(-1)
            dyn[n_gid] = { 'rank': rank, 'oid': -1, 'dft_id':-1 }

     
#    print(oids)
    id2oid = np.load(f"../preprocessing/{ds}_original_id.npy")
    ret_ids = set()
    for target_oid in target_oids:
        for f_idx, f_oid in enumerate(id2oid):
            if f_oid == target_oid:
                ret_ids.add(f_idx)
                for d_id,d in dyn.items():
                    if d['oid'] == target_oid:
                        dyn[d_id]['dft_id']=f_idx
                        
    return dyn

def readin_all_of(ff_prefix, n_rank, n_epoch, shuffle):
    epoch_data = [None] * n_epoch
    for epoch in range(n_epoch):
        epoch_data[epoch] = [None] * n_rank

        print(f"read in {ff_prefix} @epoch {epoch}")
        for rank in range(n_rank):
            file_path = f"../sample_log/{ff_prefix}_rank_{rank}_epoch_{epoch}_{shuffle}.txt"
            rows = split_log_to_list_of_nodes(file_path)
            for row in rows:
                row=row.strip()
                if row == '':
                    continue
                cuts = row.split('--------------------')
                l1 = cuts[0]
                l2 = cuts[1]

                l3 = None
                l3_dist = None
                if len(cuts) > 2:
                    l3 = cuts[2]
                    l3_dst = l3.split(')')[1].strip()

                l1_src = l1.split(')')[0].strip()
                l1_dst = l1.split(')')[1].strip()
                l1_src = string_to_list(l1_src)
                #l1_dst = string_to_list(l1_dst)

                l2_dst = l2.split(')')[1].strip()
                #l2_dst = string_to_list(l2_dst)

                if epoch_data[epoch][rank] is None:
                    epoch_data[epoch][rank] = {}
                else:
                    epoch_data[epoch][rank][l1_src[0]] = {
                        'l1_dst' : l1_dst,
                        'l2_dst' : l2_dst,
                    }
                    if l3 is not None:
                        epoch_data[epoch][rank][l1_src[0]] = {
                            'l3_dst' : l3_dst,
                        }

    all_of_ds = epoch_data
    return all_of_ds

def get_d1_d2_of_key_all_of(prefix, all_of, rank = 0, key = -1, n_epoch = 10):
    rank = int(rank)
    key  = int(key)

    d1 = {} # freq dict of node(id=key)'s second hop neighbors
    d2 = {}
    d3 = {}
    for epoch in range(n_epoch):
        try:
            rows = all_of[epoch][rank]
            nid = key
            l1_dst = rows[nid]['l1_dst']
            l2_dst = rows[nid]['l2_dst']
        
            l1_dst = string_to_list(l1_dst)
            l2_dst = string_to_list(l2_dst)
        
            count_freq(d1, l1_dst)
            count_freq(d2, l2_dst)
            if 'l3_dst' in rows[nid]:     
                l3_dst = rows[nid]['l3_dst']
                l3_dst = string_to_list(l3_dst)
                count_freq(d3, l3_dst)
        except:
            print(f"error when handle {prefix} epoch={epoch}, rank={rank}, nid={nid}")

    return d1, d2, d3

#def get_d1_d2_of_key(ff_prefix, rank = 0, key = -1, n_epoch = 20):
#    d1 = {} # freq dict of node(id=key)'s second hop neighbors
#    d2 = {}
#    d3 = {}
#    dum1 =  {}
#    dum2 = {}
#    dum3= {}
#    for epoch in range(n_epoch):
#        file_path = f"../sample_log/{ff_prefix}_rank_{rank}_epoch_{epoch}_real_no_shuffle.txt"

#        rows = split_log_to_list_of_nodes(file_path)
    
#        for row in rows:
#            row=row.strip()
#            if row == '':
#                continue
#            src = per_nodes_to_detail(dum1, dum2, dum3, row, l_2=False)
#            if src == key:
#                _ = per_nodes_to_detail(d1, d2, d3, row, l_2=True)
#                break

#    return d1, d2

# choose from dyn-xxx
# get ids for dft
shuffle_dft="no_shuffle"
shuffle_dyn="real_no_shuffle"
#shuffle_dyn="with_shuffle"
ds="ogbpr"
model_layer=3
dataset=f"{ds}_{model_layer}"
ret=choose_training_nodes(ds=ds, layer=model_layer, shuffle=shuffle_dyn)
#print(ret)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]

ROWS = 80
COLS = model_layer
assert COLS >= 2
 
RANKS = []
KEYS  = []

RANKS_2 = []
KEYS_2  = []

for dyn_id, d in ret.items():
    RANKS_2.append(d['rank'])
    KEYS_2.append(dyn_id)
    KEYS.append(d['dft_id'])

RANKS = [-1]*len(KEYS)

for rank in range(16):
    file_path = f"../sample_log/dft_{dataset}_rank_{rank}_epoch_0_no_shuffle.txt"
    rows = split_log_to_list_of_nodes(file_path)
    d1 = {}
    d2 = {}
    d3 = {}
    for row in rows:
        if row.strip() == '':
            continue
        src = per_nodes_to_detail(d1, d2, d3, row)
        for kidx in range(len(KEYS)):
            if src == KEYS[kidx]:
                RANKS[kidx] = rank
                break

print(KEYS)
print(RANKS)

import time
s = time.time()
print("start reading all_of dft...")
all_of_dft=readin_all_of(f"dft_{dataset}", n_rank = 16, n_epoch = 40, shuffle=shuffle_dft)
e = time.time()
print(f"{e-s:.2f} seconds is used.")
print("start reading all_of dyn...")
all_of_dyn=readin_all_of(f"random_{dataset}", n_rank = 16, n_epoch = 40, shuffle=shuffle_dyn)
print(f"{time.time()-e:.2f} seconds is used.")

fig, axs = plt.subplots(ROWS, COLS, figsize=(COLS*5, ROWS*5))
for r in range(ROWS):
    print(f"plotting for ROW {r}")
    # get data
    node_oid = int(ret[KEYS_2[r]]['oid'])
    #ff_prefix =  f"dft_{dataset}"
    rank      =  RANKS[r]
    key       =  KEYS[r]
    d1, d2, d3 = get_d1_d2_of_key_all_of("dft", all_of_dft, rank, key, n_epoch=20)
    freq = sorted(list(d1.values()), reverse=False)
    l1_x, l1_y = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d2.values()), reverse=False)
    l2_x, l2_y = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d3.values()), reverse=False)
    l3_x, l3_y = sorted_freq_to_accumulative(freq)

    #ff_prefix =  f"dft_{dataset}"
    rank      =  RANKS[r]
    key       =  KEYS[r]
    d1, d2, d3 = get_d1_d2_of_key_all_of("dft", all_of_dft, rank, key, n_epoch=40)
    freq = sorted(list(d1.values()), reverse=False)
    l1_xx, l1_yy = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d2.values()), reverse=False)
    l2_xx, l2_yy = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d3.values()), reverse=False)
    l3_xx, l3_yy = sorted_freq_to_accumulative(freq)

    #ff_prefix =  f"{dataset}"
    rank      =  RANKS_2[r]
    key       =  KEYS_2[r]
    d1, d2, d3 = get_d1_d2_of_key_all_of("dyn", all_of_dyn, rank, key, n_epoch=20)
    freq = sorted(list(d1.values()), reverse=False)
    l1_3x, l1_3y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d2.values()), reverse=False)
    l2_3x, l2_3y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d3.values()), reverse=False)
    l3_3x, l3_3y = sorted_freq_to_accumulative(freq)

    #ff_prefix =  f"{dataset}"
    rank      =  RANKS_2[r]
    key       =  KEYS_2[r]
    d1, d2, d3 = get_d1_d2_of_key_all_of("dyn", all_of_dyn, rank, key, n_epoch=40)
    freq = sorted(list(d1.values()), reverse=False)
    l1_4x, l1_4y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d2.values()), reverse=False)
    l2_4x, l2_4y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d3.values()), reverse=False)
    l3_4x, l3_4y = sorted_freq_to_accumulative(freq)

    # plot results
    axs[r, 0].set_title(f"{node_oid}-L1: how many nodes are sampled <=X times?")
    axs[r, 0].plot(l1_x,l1_y,color[0], linewidth=0.5, marker='o')
    axs[r, 0].plot(l1_xx,l1_yy,color[0], linewidth=0.5, marker='x')
    axs[r, 0].plot(l1_3x,l1_3y,color[2], linewidth=0.5, marker='o')
    axs[r, 0].plot(l1_4x,l1_4y,color[2], linewidth=0.5, marker='x')
    axs[r, 0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[r, 0].set(xlabel='sampled #times', ylabel='#unique nodes')
    for i, j in zip(l1_x, l1_y):
        if i % 1 == 0:
            axs[r, 0].annotate(str(j), xy=(i, j), fontsize=6)

    axs[r, 1].set_title(f"{node_oid}-L2: how many nodes are sampled <=X times?")
    axs[r, 1].plot(l2_x,l2_y,color[0], linewidth=0.5, marker='o')
    axs[r, 1].plot(l2_xx,l2_yy,color[0], linewidth=0.5, marker='x')
    axs[r, 1].plot(l2_3x,l2_3y,color[2], linewidth=0.5, marker='o')
    axs[r, 1].plot(l2_4x,l2_4y,color[2], linewidth=0.5, marker='x')
    axs[r, 1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[r, 1].set(xlabel='sampled #times', ylabel='#unique nodes')
    for i, j in zip(l2_x, l2_y):
        if i % 5 == 1:
            axs[r, 1].annotate(str(j), xy=(i, j), fontsize=6)
    if model_layer == 3:
        axs[r, 2].set_title(f"{node_oid}-L3: how many nodes are sampled <=X times?")
        axs[r, 2].plot(l3_x,l3_y,color[0], linewidth=0.5, marker='o')
        axs[r, 2].plot(l3_xx,l3_yy,color[0], linewidth=0.5, marker='x')
        axs[r, 2].plot(l3_3x,l3_3y,color[2], linewidth=0.5, marker='o')
        axs[r, 2].plot(l3_4x,l3_4y,color[2], linewidth=0.5, marker='x')
        axs[r, 2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axs[r, 2].set(xlabel='sampled #times', ylabel='#unique nodes')
        for i, j in zip(l3_x, l3_y):
            if i % 5 == 1:
                axs[r, 2].annotate(str(j), xy=(i, j), fontsize=6)
 
fig.tight_layout()
plt.savefig(f"{dataset}_{ROWS}_random_{shuffle_dyn}.svg")

#for i in range(ROWS):
#    for j in range(COLS):
