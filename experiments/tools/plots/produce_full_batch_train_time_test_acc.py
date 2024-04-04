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
"""
Neutron
    2   4   8
    sage2 -     -   -
    gcn-2 0.6698   0.6622  0.6629
    gat-2 OOM   OOM OOM

    Neutron
    sage2
    gcn2 9590     9605     12488
    gct-2 OOM     OOM   OOM

    BNS-GCN
    sage2 0.7753  0.7763   0.7740
    gcn-2 0.7342  0.7331   0.7331
    gat-2 0.7438  0.7298   0.7252

    BNS-GCN EPOCH T
    sage2 0.2478  0.1823   0.1604
    gcn-2 0.2314  0.1730   0.1588
    gat-2 0.5963  0.3332   0.2281  

    BNS-GCN EPOCH T*500
    sage2   123.9   91.15   80.2
    gcn2    115.7   86.5    79.4
    gat2    298.15  166.6   114.05
"""

import matplotlib.pyplot as plt


table_time = {
        'model'   : ['sage_2',      'sage_2',     'sage_2',     'gcn_2',        'gcn_2',  'gcn_2'      ,  'gat_2',       'gat_2',          'gat_2',  'sage_3'     ,  'sage_3'     ,      'sage_3',       'gcn_3',   'gcn_3',      'gcn_3' ],
        'systems' : ['NeutronStar', 'BNS-GCN' ,'Triskelion', 'NeutronStar', 'BNS-GCN'  , 'Triskelion'  ,  'NeutronStar', 'BNS-GCN', 'Triskelion'  ,  'NeutronStar', 'BNS-GCN'     ,  'Triskelion', 'NeutronStar', 'BNS-GCN', 'Triskelion' ],
        '2-GPUs'  : [-1           , 124/500       ,   'TBD'    , 9590/500         ,       116/500  ,'TBD'          , 'OOM'         ,  299/500     , 'TBD'         ,      -1       , -1            , -1           , -1           , -1       ,        -1,   ],
        '4-GPUs'  : [-1           ,  92/500       ,    823/500     , 9605/500         ,       87/500   , 834/500           , 'OOM'         ,  167/500     ,  941/500         ,      -1       , - 1           , -1           , -1           , -1       ,        -1,   ],
        '8-GPUs'  : [-1           ,  81/500       ,   'TBD'    , 12488/500        ,       80/500   ,'TBD'          , 'OOM'         ,  115/500     , 'TBD'         ,      -1       , -1            , -1           , -1           , -1       ,        -1,   ],
}


table_test_acc = {
    'model'   : ['sage_2',        'sage_2',    'sage_2',       'gcn_2',     'gcn_2',  'gcn_2'      ,  'gat_2'      , 'gat_2'  ,        'gat_2',  'sage_3'      ,  'sage_3' ,      'sage_3',       'gcn_3',   'gcn_3',      'gcn_3' ],
    'systems' : ['NeutronStar', 'BNS-GCN' ,'Triskelion', 'NeutronStar', 'BNS-GCN'  , 'Triskelion'  ,  'NeutronStar', 'BNS-GCN', 'Triskelion'  ,  'NeutronStar' , 'BNS-GCN' ,  'Triskelion', 'NeutronStar', 'BNS-GCN', 'Triskelion' ],
    '2-GPUs'  : [-1           , 0.7753    ,   'TBD'    , 0.6698       ,   0.7342   ,     'TBD'     ,  'OOM'        ,  0.7438  , 'TBD'         ,       -1        , -1       , -1           , -1           , -1       ,  -1          ],
    '4-GPUs'  : [-1           , 0.7763    ,    0.7503  , 0.6622       ,   0.7331   ,   0.7146      ,  'OOM'        ,  0.7298  ,  0.7718       ,       -1        ,- 1       , -1           , -1           , -1       , -1           ],
    '8-GPUs'  : [-1           , 0.7740    ,   'TBD'    , 0.6629       ,   0.7331   ,    'TBD'      ,  'OOM'        ,  0.7252  , 'TBD'         ,       -1        , -1       , -1           , -1           , -1       , -1           ],
}


df= pd.DataFrame(table_time)
df=df.round(1)
df_dump(f"full_graph_train_time", df)
print(df)
df = pd.DataFrame(table_test_acc)
df=df.round(4)
df_dump(f"full_graph_test_accuracy", df)
print(df)
