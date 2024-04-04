#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from enum import Enum

DictType = Enum('DictType', ['LINE', 'BAR', 'ERR_LU_BAR'])

#########################################
# colors
#########################################
RED_1 = "#C55659"
RED_2 = "#D22027"
BLACK_1 = '#171717'

BLUE_1 = '#7FA5B7'
BLUE_2 = '#385989'
BLUE_3 = '#35547C'

CYAN_1 = '#5BB7CD'

ORANGE_1 = '#F7903D'

YELLOW_1 = '#CBB47B'
YELLOW_2 = '#F7CE7D'

GREEN_1 = '#68BA9F'
GREEN_2 = '#54AC75'

PURPLE_1 = '#7572B5'

PINK_1 = '#DE5D78'

BLUE_2 = '#385989'
ORANGE_1 = '#F7903D'
GREEN_2 = '#54AC75'

PALETTE = {
    1:
    {
        'red': RED_2,
        'blue': BLUE_2
    },
    2:
    {
        'eq': [RED_2, BLUE_2],
        'diff': [RED_2, BLACK_1]
    },
    3:
    {
        'eq': [BLUE_2, ORANGE_1, GREEN_2],
        'diff': [RED_2, BLUE_2, BLUE_1]
    },
    4:
    {
        'eq': [BLUE_2, GREEN_1, PINK_1, YELLOW_2]
    },
    5:
    {
        'eq': [YELLOW_1, CYAN_1, GREEN_2, RED_1, PURPLE_1]
    },
    10: {
        'eq' : sns.color_palette('muted')
    },
    16: {
        'eq' : sns.color_palette("husl", 16)
    }
}

def get_color_map(legends):
    cm = {}
    n_le = len(legends)
    if n_le < 6:
        n_leg = n_le
    elif n_le < 11:
        n_leg = 10
    else:
        n_leg=16
    if n_leg == 1:
        cm[list(legends)[0]] = PALETTE[1]['blue']
        return cm
    for idx, k in enumerate(legends):
        cm[k] = PALETTE[n_leg]['eq'][idx]
    return cm

##############################
# compare
##############################
def plot_lines(lines_dict, color_map=None, repurpose_as_scatter=False):
    """ 
    lines_dict = {
        'title'  ：'title',
        'fig'    : 'fig',
        'xlabel' : 'x' ,
        'ylabel' : 'y',
        'xlim' : [ min, max ],
        'ylim' : [ min, max ],
        'grid' : [ x, y ],
        'data' : {
            key_1 : [ (x0, y0), (x1, y1), ],
            key_2 : [ (x0, y0), (x1, y1), ]
        }
    }
    """
    ld = lines_dict
    cm = color_map

    if cm is None:
        cm = get_color_map(ld['data'].keys())

    legends = tuple()
    fig, ax = plt.subplots()

    if 'xlim' in ld:
        xt = (float(ld['xlim'][1]) - float(ld['xlim'][0])) / float(ld['grid'][0])
        ax.set_xticks([i*xt for i in range(ld['xlim'][0], ld['xlim'][1])])
        plt.xlim([float(ld['xlim'][0]), float(ld['xlim'][1])])

    if 'ylim' in ld:
        yt = (float(ld['ylim'][1]) - float(ld['ylim'][0])) / float(ld['grid'][1])

        ax.set_yticks( [float(i)*yt + float(ld['ylim'][0]) for i in range(ld['grid'][1]) ])
        plt.ylim([float(ld['ylim'][0]), float(ld['ylim'][1])])

    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.15)

    for lgd, line in ld['data'].items():
        legends = legends + (lgd,)
        l = list(zip(*line))
        x, y = list(l[0]), list(l[1])

        if repurpose_as_scatter:
            ax.scatter(x, y, color=cm[lgd], s=20, marker='.')
            #ax.scatter(x, y, color=cm[lgd], s=50, linewidth=1.0, marker='.')
        else:
            ax.plot(x, y, cm[lgd], linewidth=2.0)

    leg = ax.legend(legends, loc='best', shadow=False, fontsize="20", markerscale=4.0)
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)

    ax.grid(True)

    plt.xlabel(ld['xlabel'], fontsize=18)
    plt.ylabel(ld['ylabel'], fontsize=18)
    plt.title(ld['title'], fontsize=18)

    plt.savefig(ld['fig'])

##############################
# compare
##############################
def plot_bars(bars_dict, color_map=None, use_stack_mode=False, data_label=True):
    """ 
    bars_dict = {
        'title'  ：'title',
        'fig'    : 'fig',
        'xlabel' : 'x' ,
        'ylabel' : 'y',
        'xlim' : [ min, max ],
        'ylim' : [ min, max ],
        'grid' : [ x, y ],
        'data' : {
            'group_name' : [ n0, n1, ..., n_k],
            'mean' : {
                'key1' : [ v0, v1, ..., v_k ],
                'key2' : [ v0, v1, ..., v_k ],
                'key3' : [ v0, v1, ..., v_k ]
            },
            'stdev' : {
                'key1' : [ v0, v1, ..., v_k ],
                'key2' : [ v0, v1, ..., v_k ],
                'key3' : [ v0, v1, ..., v_k ]
            }
        }
    }
    """

    bd = bars_dict
    cm = color_map

    if cm is None:
        cm = get_color_map(bd['data']['mean'].keys())

    value_fmt = "{:.1f}"
    err_capsize = 3
    # color='#2a9d8fBB'
    # hatch='///'

    group_names = bd['data']['group_name']
    means = bd['data']['mean']

    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(len(group_names))  # the label locations
    grp_n = len(means.keys()) -1  if len(means.keys()) > 1 else 1
    num_w = grp_n / 2.0
    width = (2.0/num_w) / len(group_names)  # the width of the bars
    xtick_offset = width * num_w
    multiplier = 0

    if use_stack_mode:
        bottom = np.zeros(len(group_names))
        for (k, v) in means.items():
            p = ax.bar(group_names, v, width, label=k, bottom=bottom)
            if data_label:
                ax.bar_label(p, label_type='center')
            bottom += v
    else:
        stdevs = bd['data']['stdev']
        for (k, v), (k2, v2) in zip(means.items(), stdevs.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, v, width, yerr=v2,
                           label=k, capsize=err_capsize, color=cm[k])
            if data_label:
                ax.bar_label(rects, padding=3, label_type='edge', fmt=value_fmt)
            multiplier += 1
        ax.set_xticks(x + xtick_offset, group_names)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(bd['title'])
    ax.set_ylabel(bd['ylabel'])
    ax.set_xlabel(bd['xlabel'])

    if 'xlim' in bd:
        ax.set_xlim(xmin=float(bd['xlim'][0]), amax=float(bd['xlim'][1]))
    if 'ylim' in bd:
        ax.set_ylim(ymin=float(bd['ylim'][0]), ymax=float(bd['ylim'][1]))


    ax.legend(loc='best', ncols=2)
    plt.savefig(bd['fig'])

