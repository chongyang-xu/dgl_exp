"""Functions for partitions. """

import os
import pandas as pd
import numpy as np
import torch as th
from .. import backend as F

def vc_load_full_node_split_mask_from_disk(split_file_path, num_nodes):
    train_idx = F.zerocopy_from_numpy(pd.read_csv(os.path.join(split_file_path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]).to(th.long)
    valid_idx = F.zerocopy_from_numpy(pd.read_csv(os.path.join(split_file_path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]).to(th.long)
    test_idx = F.zerocopy_from_numpy(pd.read_csv(os.path.join(split_file_path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]).to(th.long)
    train_mask = th.zeros((num_nodes,), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((num_nodes,), dtype=th.bool)
    val_mask[valid_idx] = True
    test_mask = th.zeros((num_nodes,), dtype=th.bool)
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def vc_load_full_node_feat_from_disk(full_node_feat_path):
    if full_node_feat_path[-4:] == ".npy":
        full_node_feat = np.load(full_node_feat_path)
    elif full_node_feat_path[-4:] == ".npz":
        data_dict = np.load(full_node_feat_path)
        full_node_feat = data_dict['node_feat']
    elif full_node_feat_path[-4:] == ".bin":
        full_node_feat = np.fromfile(full_node_feat_path, dtype=np.float32)
    else:
        assert False, "{} not supported, only .npy and npz are supported".format(full_node_feat_path[-4:])
    assert full_node_feat is not None
    full_node_feat = F.zerocopy_from_numpy(full_node_feat)
    return full_node_feat

def vc_load_full_node_label_from_disk(full_node_label_path):
    if full_node_label_path[-4:] == ".npz":
        node_label = np.load(full_node_label_path)['node_label']
    elif full_node_label_path[-4:] == ".npy":
        node_label = np.load(full_node_label_path)
    elif full_node_label_path[-4:] == ".csv":
        node_label = np.loadtxt(full_node_label_path, delimiter=',')
    else:
        assert False, "only .npz and .npy formats are implemented"

    if np.isnan(node_label).any():
        node_label = th.from_numpy(node_label).to(th.float32)
    else:
        node_label = th.from_numpy(node_label).to(th.long)
    return node_label

def local_to_part_mask(vc_map, part_id):
    pids = vc_map & 0xFFFF
    mask = pids == part_id
    return mask
