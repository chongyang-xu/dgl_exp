import numpy as np
import pandas as pd

def string_to_list(list_string):
    a = list_string.replace('[', '')
    a = a.replace(']', '')
    aa = a.split(',')
    ret = list()
    for ele in aa:
        ret.append(int(ele))

    return ret


def count_freq(d, numbers):
    for num in numbers:
        if num in d:
            d[num] += 1
        else:
            d[num] = 1

def sorted_freq_to_accumulative(sorted_freq):
    max_v = sorted_freq[-1]
    d = {int(0):int(0)}
    for i in range(1, max_v+1):
        d[i] = int(0)
    for v in sorted_freq:
        d[v] += int(1)

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


def get_d1_d2_of_key(ff_prefix, rank = 0, key = 153337, n_epoch = 20):
    d1 = {} # freq dict of node(id=key)'s second hop neighbors
    d2 = {}
    for epoch in range(n_epoch):
        file_path = f"../sample_log/{ff_prefix}_rank_{rank}_epoch_{epoch}_no_shuffle.txt"

        with open(file_path, 'r') as ff:
            file_content = ff.read()
        content = file_content.replace('tensor(', '')
        splits = content.strip().split('--------------------\n--------------------')
    
        count = 0
        for row in splits:
            row=row.strip()
            if row == '':
                continue
            l1 = row.split('--------------------')[0]
            l2 = row.split('--------------------')[1]

            l1_src = l1.split(')')[0].strip()
            l1_dst = l1.split(')')[1].strip()
            l1_src = sorted(string_to_list(l1_src))
            l1_dst = sorted(string_to_list(l1_dst))
            
            #print(f"{l1_src[0]}")
            #count += 1
            #if count > 100:
            #    exit(0)
            if key in l1_src:
                pass
            else:
                continue
        
            l2_src = l2.split(')')[0].strip()
            l2_dst = l2.split(')')[1].strip()
            l2_src = sorted(string_to_list(l2_src))
            l2_dst = sorted(string_to_list(l2_dst))

            count_freq(d1, l1_dst)
            count_freq(d2, l2_dst)
            break
    return d1, d2


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'
color = [ ORANGE_1, GREEN_2, BLUE_2]

ROWS = 20
COLS = 2
fig, axs = plt.subplots(ROWS, COLS, figsize=(COLS*5, ROWS*5))
 
#RANKS = [0     , 0     , 0     , 0     , 0     , 0     , 0     , 0     , 0     , 0     ]
#KEYS  = [153337, 153312, 153468, 215951, 231053, 240889, 250456, 250472, 285649, 285653] # ogbpr metis ids
RANKS = [0]*100
#KEYS = [153312, 153324, 153337, 153468, 153507, 153526, 153608, 153624, 153656, 153678, 153683, 153686, 153716, 153718, 153725, 153729, 153735, 153740, 153777, 153795, 153798, 153819, 153832, 153903, 153942, 153946, 153951, 153993, 154036, 154069, 154081, 154085, 154123, 154125, 154135, 154153, 154157, 154170, 154180, 154181, 154203, 154250, 154288, 154303, 154328, 154347, 154358, 154386, 154409, 154463, 154464, 154484, 154487, 154550, 154575, 154576, 154593, 154635, 154658, 154688, 154691, 154700, 154725, 154743, 154748, 154752, 154791, 154813, 154816, 154851, 154907, 154911, 154919, 154945, 154954, 155074, 155257, 155301, 155311, 155324, 155375, 155391, 155396, 155399, 155406, 155408, 155437, 155454, 155457, 155462, 155503, 155520, 155558, 155585, 155606, 155636, 155645, 155662, 155663, 155690 ] # default_
KEYS = [500613, 500628, 500631, 500633, 500636, 500642, 500647, 500671, 500676, 500685, 500688, 500696, 500698, 500701, 500703, 500711, 500717, 500742, 500749, 500761, 500780, 500782, 500798, 500805, 500811, 500824, 500829, 500836, 500837, 500846, 500850, 500858, 500868, 500878, 500884, 500888, 500892, 500895, 500897, 500899, 500900, 500908, 500913, 500922, 500925, 500932, 500936, 500942, 500951, 500952, 500961, 500969, 500971, 500975, 500977, 500980, 500990, 500991, 500998, 500999, 501006, 501039, 501052, 501058, 501060, 501069, 501072, 501074, 501090, 501095, 501102, 501108, 501109, 501114, 501117, 501121, 501134, 501136, 501149, 501150, 501159, 501160, 501161, 501162, 501173, 501195, 501202, 501203, 501205, 501214, 501232, 501247, 501258, 501260, 501261, 501263, 501264, 501265, 501273, 501280, ] # default2_
#original_ids = [2314678,   57935,  858599, 1695554,  839219,  802029,  914457, 85320,  557590,  758791]

# seems wrong
#RANKS_2 = [ 9      , 6     , 8      , 10     , 7      , 10     , 10     , 7      , 7      , 6      ]
#KEYS_2  = [ 1527598, 934769, 1331016, 1613664, 1137102, 1648053, 1560024, 1091701, 1178602, 1043726]
#RANKS_2 = [ 0      , 0     , 0      , 0     , 0      , 0     , 0     , 0      , 0      , 0      ]
#KEYS_2  = [ 0      ,327    , 721    , 4622  , 11459  , 506   , 9291  , 9595   , 9666   , 9759]
RANKS_2 = [0]*100
KEYS_2  = [0, 5, 14, 35, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 74, 77, 78, 79, 84, 89, 96, 100, 102, 105, 106, 108, 110, 111, 114, 115, 118, 119, 121, 124, 125, 126, 127, 129, 130, 132, 133, 134, 136, 138, 141, 142, 144, 145, 147, 148, 150, 154, 158, 159, 160, 161, 162, 163, 168, 172, 176, 177, 179, 181, 184, 186, 187, 190, 195, 208, 217, 221, 224, 226, 227, 229, 235, 236, 237, 239, 241, 250, 251, 252, 253]

for r in range(ROWS):
    # get data
    ff_prefix =  "default2_ogbpr"
    rank      =  RANKS[r]
    key       =  KEYS[r]
    d1, d2 = get_d1_d2_of_key(ff_prefix, rank, key)
    freq = sorted(list(d1.values()), reverse=False)
    l1_x, l1_y = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d2.values()), reverse=False)
    l2_x, l2_y = sorted_freq_to_accumulative(freq)

    ff_prefix =  "default2_ogbpr"
    rank      =  RANKS[r]
    key       =  KEYS[r]
    d1, d2 = get_d1_d2_of_key(ff_prefix, rank, key, n_epoch=40)
    freq = sorted(list(d1.values()), reverse=False)
    l1_xx, l1_yy = sorted_freq_to_accumulative(freq)

    freq = sorted(list(d2.values()), reverse=False)
    l2_xx, l2_yy = sorted_freq_to_accumulative(freq)

    ff_prefix =  "ogbpr"
    rank      =  RANKS_2[r]
    key       =  KEYS_2[r]
    d1, d2 = get_d1_d2_of_key(ff_prefix, rank, key)
    freq = sorted(list(d1.values()), reverse=False)
    l1_3x, l1_3y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d2.values()), reverse=False)
    l2_3x, l2_3y = sorted_freq_to_accumulative(freq)

    ff_prefix =  "ogbpr"
    rank      =  RANKS_2[r]
    key       =  KEYS_2[r]
    d1, d2 = get_d1_d2_of_key(ff_prefix, rank, key, n_epoch=40)
    freq = sorted(list(d1.values()), reverse=False)
    l1_4x, l1_4y = sorted_freq_to_accumulative(freq)
    freq = sorted(list(d2.values()), reverse=False)
    l2_4x, l2_4y = sorted_freq_to_accumulative(freq)

    # plot results
    axs[r,0].set_title(f"L1: how many nodes are sampled <=X times?")
    axs[r,0].plot(l1_x,l1_y,color[0], linewidth=0.5, marker='o')
    axs[r,0].plot(l1_xx,l1_yy,color[0], linewidth=0.5, marker='x')
    axs[r,0].plot(l1_3x,l1_3y,color[2], linewidth=0.5, marker='o')
    axs[r,0].plot(l1_4x,l1_4y,color[2], linewidth=0.5, marker='x')
    axs[r,0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[r,0].set(xlabel='sampled #times', ylabel='#unique nodes')
    for i, j in zip(l1_x, l1_y):
        if i % 1 == 0:
            axs[r,0].annotate(str(j), xy=(i, j), fontsize=6)

    axs[r,1].set_title(f"L2: how many nodes are sampled <=X times?")
    axs[r,1].plot(l2_x,l2_y,color[0], linewidth=0.5, marker='o')
    axs[r,1].plot(l2_xx,l2_yy,color[0], linewidth=0.5, marker='x')
    axs[r,1].plot(l2_3x,l2_3y,color[2], linewidth=0.5, marker='o')
    axs[r,1].plot(l2_4x,l2_4y,color[2], linewidth=0.5, marker='x')
    axs[r,1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[r,1].set(xlabel='sampled #times', ylabel='#unique nodes')
    for i, j in zip(l2_x, l2_y):
        if i % 5 == 1:
            axs[r,1].annotate(str(j), xy=(i, j), fontsize=6)

    
fig.tight_layout()
plt.savefig("test_20.svg")

#for i in range(ROWS):
#    for j in range(COLS):
