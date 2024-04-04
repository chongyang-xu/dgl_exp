import time
s = time.time()
import ctypes
from numpy.ctypeslib import ndpointer

import dgl
import dgl.backend as F
from dgl.partition import metis_partition_assignment as metis_assignment
from dgl.partition import partition_graph_vertex_cut_with_halo as custom_vc_assignment
from dgl.partition import get_peak_mem
from dgl.random import choice as random_choice

import torch as th

import numpy as np
import os, pathlib

import json

print(f"Importing python packages, time: {time.time()- s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
def get_u_v_nid2pid_vcdeg(vc_json_file, n_n, nid_type, n_parts):
    # parsing binaray edge file
    with open(vc_json_file, 'r') as f:
        vc_json = json.load(f)
    edge_file = vc_json['edge_file_bin']

    s = time.time()
    edges = np.fromfile(edge_file, dtype=nid_type)
    print(f"Loading {edge_file}, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    s = time.time()
    n_e = len(edges) // 2
    u = edges[:n_e].astype(np.int64)
    v = edges[n_e:].astype(np.int64)
    edges = None
    uu = F.zerocopy_from_numpy(u)
    vv = F.zerocopy_from_numpy(v)
    print(f"Converting edges into np.int64, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    # create DGL graph and do metis assignment
    s = time.time()
    if n_parts == 1:
        node_parts = th.zeros((n_n,), dtype=th.int64)
    else:
        assert vc_json is not None, "vc_json is required"
        edge_file_bin = vc_json['edge_file_bin']
        num_nodes = vc_json['num_nodes']
        num_edges = vc_json['num_edges']
        train_mask_bin = vc_json['train_mask_file_bin']
        num_train_nodes = vc_json['num_train_nodes']
        add_rev_edge = vc_json['add_reverse_edge']

        start = time.time()
        vc_maps, parts, _, _ = custom_vc_assignment(
                edge_file_bin, num_nodes, num_edges, n_parts, 'vcdeg', 0, reshuffle=False, # 0 is num_hops
                num_train_nodes=num_train_nodes, train_mask_file=train_mask_bin, add_rev_edge=add_rev_edge)
        print(f"vcdeg partition time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
        VCR_MPID_MASK = 0xFFFF
        vc_maps[0] = vc_maps[0] & VCR_MPID_MASK
    
    nid2pid = F.zerocopy_to_numpy(vc_maps[0])
    assert u.dtype == np.int64
    assert v.dtype == np.int64
    assert nid2pid.dtype == np.int64
    return u, v, nid2pid

def get_u_v_nid2pid(edge_file, n_n, nid_type, n_parts, method='metis'):
    # parsing binaray edge file
    s = time.time()
    edges = np.fromfile(edge_file, dtype=nid_type)
    print(f"Loading {edge_file}, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    s = time.time()
    n_e = len(edges) // 2
    u = edges[:n_e].astype(np.int64)
    v = edges[n_e:].astype(np.int64)
    edges = None
    uu = F.zerocopy_from_numpy(u)
    vv = F.zerocopy_from_numpy(v)
    print(f"Converting edges into np.int64, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    # create DGL graph and do metis assignment
    s = time.time()
    if n_parts == 1:
        node_parts = th.zeros((n_n,), dtype=th.int64)
    else:
        if method == 'metis':
            g = dgl.graph((uu, vv))
            print(f"Creating DGLGraph, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
            uu, vv = None, None
            node_parts = metis_assignment(g, n_parts, mode='k-way')
        elif method == 'random':
            node_parts = random_choice(n_parts, n_n)
            print(f"run random_choice, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
        else:
            assert False, f"not supported method: {method}"
    nid2pid = F.zerocopy_to_numpy(node_parts)
    assert u.dtype == np.int64
    assert v.dtype == np.int64
    assert nid2pid.dtype == np.int64
    return u, v, nid2pid

def get_node_feat(feat_file, feat_type, feat_shape):
    s = time.time()
    node_feat = np.fromfile(feat_file, dtype=feat_type)
    node_feat = node_feat.reshape(feat_shape)
    print(f"Loading {feat_file}, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    return node_feat

def get_node_label_train_valid_test_mask(DS_ORI, label_type, mask_type):
    s = time.time()
    label_file = np.fromfile(f"{DS_ORI}/node_label.bin", dtype=label_type)
    train_mask = np.fromfile(f"{DS_ORI}/train_mask.bin", dtype=mask_type)
    valid_mask = np.fromfile(f"{DS_ORI}/valid_mask.bin", dtype=mask_type)
    test_mask =  np.fromfile(f"{DS_ORI}/test_mask.bin", dtype=mask_type)
    print(f"Loading node labels and masks, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    return label_file, train_mask, valid_mask, test_mask

class Chunking:
    def __init__(self, so_file=None):
        self.so = ctypes.CDLL("./libchunking.so" if so_file is None else so_file)
    def get_build_function(self):
        self.so.build.restype = None
        self.so.build.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ndpointer(ctypes.c_int64, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int64, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int64, flags="C_CONTIGUOUS"),
        ]
        return self.so.build
    def get_split_node_feat_function(self):
        self.so.split_node_feat.restype = None
        self.so.split_node_feat.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ]
        return self.so.split_node_feat

    def get_split_node_data_function(self):
        self.so.split_node_data.restype = None
        self.so.split_node_data.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
        ]
        return self.so.split_node_data

    def build(self, path, n_n, n_e, feat_dim, n_parts, src_nid, dst_nid, nid2pid):
        func = self.get_build_function()
        path_in = path.encode('utf-8')
        s = time.time()
        func(path_in, n_n, n_e, feat_dim, n_parts, src_nid, dst_nid, nid2pid)
        print(f"chunking.build, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    def split_node_feat(self, path, n_n, n_dim, n_parts, node_feat):
        func = self.get_split_node_feat_function()
        path_in = path.encode('utf-8')
        s = time.time()
        func(path_in, n_n, n_dim, n_parts, node_feat)
        print(f"chunking.split_node_feat, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    def split_node_data(self, path, n_n, n_parts, node_label, train_mask, val_mask, test_mask):
        func = self.get_split_node_data_function()
        path_in = path.encode('utf-8')
        s = time.time()
        func(path_in, n_n, n_parts, node_label, train_mask, val_mask, test_mask)
        print(f"chunking.split_node_data, time: {time.time()-s:.2f} seconds, peak_mem: {get_peak_mem():.3f} GB")
    def print_string(self, str_in):
        b_str_in = str_in.encode('utf-8')
        self.so.print_string.argtypes = [ctypes.c_char_p, ]
        self.so.print_string(b_str_in)

# input
n_parts = 16
ds="reddit"
part_method='random' # assign nodes to part

#method2tag = {'metis' : '', 'vcdeg': '_vcdeg'}
method2tag = {'metis' : '_mtest', 'vcdeg': '_vtest', 'random' : '_rtest'}
chunk_path_tag=method2tag[part_method]

if ds == "ogbpr":
#DS_ORI = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_ori/ogbn_products/raw/"
    DS_ORI = "/data/ds_ori/ogbn_products/raw/"
    n_n   = 2449029
    n_e   = 126167053 # 61859140 is the number before preprocessing
    n_dim = 100
    #path = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_pre/ogbn_products/chunking_{n_parts}"
    path = f"/data/ds_pre/ogbpr/chunking_{n_parts}{chunk_path_tag}"
    add_reverse = False # preprocessing phase has added
elif ds == "ogbar":
#DS_ORI = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_ori/ogbn_products/raw/"
    DS_ORI = "/data/ds_ori/ogbn_arxiv/raw/"
    n_n   = 169343
    n_e   = 1335586 # 1166243 w/o self-loop
    n_dim = 128
    #path = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_pre/ogbn_products/chunking_{n_parts}"
    path = f"/data/ds_pre/ogbar/chunking_{n_parts}{chunk_path_tag}"
    add_reverse = False
elif ds == "ogbpa":
#DS_ORI = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_ori/ogbn_products/raw/"
    DS_ORI = "/data/ds_ori/ogbn_papers100M/raw/"
    n_n   = 111059956
    n_e   = 1726745828 # 1615685872 w/o self-loop
    n_dim = 128
    #path = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_pre/ogbn_products/chunking_{n_parts}"
    path = f"/data/ds_pre/ogbpa/chunking_{n_parts}{chunk_path_tag}"
    add_reverse = False
elif ds == "cora":
    DS_ORI = "/data/ds_ori/cora_v2/"
    n_n   = 2708
    n_e   = 13264 # 10556 w/o self-loop
    n_dim = 1433
    #path = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_pre/ogbn_products/chunking_{n_parts}"
    path = f"/data/ds_pre/cora/chunking_{n_parts}{chunk_path_tag}"
    add_reverse = False
elif ds == "reddit":
    DS_ORI = "/data/ds_ori/reddit/"
    n_n   = 232965
    n_e   = 114848857 #114615892 w/o loop
    n_dim = 602
    #path = f"/DS/dsg-graphs/nobackup/ds4gnn/DATA/ds_pre/ogbn_products/chunking_{n_parts}"
    path = f"/data/ds_pre/reddit/chunking_{n_parts}{chunk_path_tag}"
    add_reverse = False

nid_type  = np.uint32
edge_file = f"{DS_ORI}/edge_index.bin"
feat_file = f"{DS_ORI}/node_feat.bin"

# /data/ds_ori/cora_v2/

# build libchunking
os.system("make clean")
os.system("make")

# /data/ds_ori/cora_v2/

# load graph
if part_method == 'metis' or part_method == 'random':
    o_u, o_v, nid2pid = get_u_v_nid2pid(edge_file, n_n, np.uint32, n_parts, part_method)

if part_method == 'vcdeg':
    vc_config = f"{DS_ORI}/vc_{ds}.json" 
    o_u, o_v, nid2pid = get_u_v_nid2pid_vcdeg(vc_config, n_n, np.uint32, n_parts)

#if add_reverse:
if False: # preprocessing decide if add reverse
    u = np.concatenate((o_u, o_v))
    v = np.concatenate((o_v, o_u))
    n_e = n_e * 2
else:
    u = o_u
    v = o_v

# build chunk
chunking = Chunking()

pathlib.Path(path).mkdir(parents=True, exist_ok=True)
chunking.build(path, n_n, n_e, n_dim, n_parts, u, v, nid2pid)

node_feat = get_node_feat(feat_file, np.float32, (n_n, n_dim))
print(node_feat.shape)
print(node_feat)
chunking.split_node_feat(path, n_n, n_dim, n_parts, node_feat)

node_label, train_mask, val_mask, test_mask = get_node_label_train_valid_test_mask(DS_ORI, label_type=np.float32, mask_type=np.int8)
chunking.split_node_data(path, n_n, n_parts, node_label, train_mask, val_mask, test_mask)
print(node_label.shape)
print(node_label)
print(train_mask.shape)
print(train_mask)
