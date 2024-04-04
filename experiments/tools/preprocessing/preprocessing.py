import os
import argparse

import dgl
from dgl.data import CoraGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
#from igb.dataloader import IGBHeteroDGLDataset

import torch as th
import json
import numpy as np
import pandas as pd

def load_ogb(name, root_path):
    print("start loading", name)
    data = DglNodePropPredDataset(name=name, root=root_path)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

#   graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["label"] = labels
#    in_feats = graph.ndata["features"].shape[1]

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels

def load_dataset_into_memory(args, ori_ds_path):
    if args.dataset == "ogbpr":
        ds_full_name = 'ogbn-products'
        dgl_g, _ = load_ogb(ds_full_name, ori_ds_path)
    elif args.dataset == "ogbar":
        ds_full_name = 'ogbn-arxiv'
        dgl_g, _ = load_ogb(ds_full_name, ori_ds_path)
    elif args.dataset == "ogbpa":
        ds_full_name = 'ogbn-papers100M'
        dgl_g, _ = load_ogb(ds_full_name, ori_ds_path)
    elif args.dataset == "cora":
        # num_class = 7
        # in_feats = 1433
        ds_full_name = 'cora'
        dataset = CoraGraphDataset(raw_dir=ori_ds_path)
        dgl_g = dataset[0]
        print(dgl_g)
    elif args.dataset == "reddit":
        ds_full_name = "reddit"
        dataset = RedditDataset(raw_dir=ori_ds_path)
        dgl_g = dataset[0]
        print(dgl_g)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.self_loop and dgl_g is not None:
        dgl_g = dgl.remove_self_loop(dgl_g)
        dgl_g = dgl.add_self_loop(dgl_g)

    return dgl_g

def prepare_dataset_for_vc(args, ori_ds_path, dgl_g):
    if args.dataset == "cora":
        ds_full_name = "cora"
        ds_full_name_dir = "cora_v2"
        raw_path = "{}/{}".format(ori_ds_path, ds_full_name_dir)
        vc_json_file = "{}/vc_cora.json".format(raw_path)
        if os.path.exists(vc_json_file):
            print("found {}, return prepare_dataset_for_vc".format(vc_json_file))
            return vc_json_file
        vc_json = {}
        ##########################
        node_label_file_bin = f"{raw_path}/node_label.bin"
        node_feat_file_bin = f"{raw_path}/node_feat.bin"
        train_mask_file_bin=f"{raw_path}/train_mask.bin"
        val_mask_file_bin=f"{raw_path}/valid_mask.bin"
        test_mask_file_bin=f"{raw_path}/test_mask.bin"
        edge_file_bin = f"{raw_path}/edge_index.bin"

        src, dst = dgl_g.edges()
        edge = th.concat((src, dst)).numpy()
        edge=edge.astype(np.uint32)
        edge.tofile(edge_file_bin)
        #print(edge)

        nl = dgl_g.ndata['label'].numpy()
        nl = nl.astype(np.float32).flatten()
        nl.tofile(node_label_file_bin)

        node_feat = dgl_g.ndata['feat'].numpy()
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
        node_feat.tofile(node_feat_file_bin)
 
        print(dgl_g.ndata)

        dgl_g.ndata['train_mask'].numpy().tofile(train_mask_file_bin)
        dgl_g.ndata['val_mask'].numpy().tofile(val_mask_file_bin)
        dgl_g.ndata['test_mask'].numpy().tofile(test_mask_file_bin)
        ############################

        vc_json['node_label_file'] = node_label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feat_file_bin
        vc_json['split_file_path'] = "none"
        vc_json['num_nodes'] = 2708
        vc_json['num_edges'] = 13264 # 10556 is the number before add self loop
        vc_json['feat_dim'] = 1433
        vc_json['train_mask_file_bin'] = train_mask_file_bin
        vc_json['num_train_nodes'] = 140
        vc_json['add_reverse_edge'] = False

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file
    
    if args.dataset == "reddit":
        ds_full_name = "reddit"
        ds_full_name_dir = "reddit"
        raw_path = "{}/{}".format(ori_ds_path, ds_full_name_dir)
        vc_json_file = "{}/vc_reddit.json".format(raw_path)
        if os.path.exists(vc_json_file):
            print("found {}, return prepare_dataset_for_vc".format(vc_json_file))
            return vc_json_file
        vc_json = {}
        ##########################
        node_label_file_bin = f"{raw_path}/node_label.bin"
        node_feat_file_bin = f"{raw_path}/node_feat.bin"
        train_mask_file_bin=f"{raw_path}/train_mask.bin"
        val_mask_file_bin=f"{raw_path}/valid_mask.bin"
        test_mask_file_bin=f"{raw_path}/test_mask.bin"
        edge_file_bin = f"{raw_path}/edge_index.bin"

        src, dst = dgl_g.edges()
        edge = th.concat((src, dst)).numpy()
        edge=edge.astype(np.uint32)
        edge.tofile(edge_file_bin)
        #print(edge)

        nl = dgl_g.ndata['label'].numpy()
        nl = nl.astype(np.float32).flatten()
        nl.tofile(node_label_file_bin)

        node_feat = dgl_g.ndata['feat'].numpy()
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
        node_feat.tofile(node_feat_file_bin)
 
        dgl_g.ndata['train_mask'].numpy().tofile(train_mask_file_bin)
        dgl_g.ndata['val_mask'].numpy().tofile(val_mask_file_bin)
        dgl_g.ndata['test_mask'].numpy().tofile(test_mask_file_bin)
        ############################

        vc_json['node_label_file'] = node_label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feat_file_bin
        vc_json['split_file_path'] = "none"
        vc_json['num_nodes'] = 232965
        vc_json['num_edges'] = 114848857 # 114615892 is number of edges before adding self loop
        vc_json['feat_dim'] = 602
        vc_json['train_mask_file_bin'] = train_mask_file_bin
        vc_json['num_train_nodes'] = 153431
        vc_json['add_reverse_edge'] = False

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file

    if args.dataset == "igb":
        pass
        ds_full_name = "IGBH"
        ds_full_name_prefix = "igb/IGB-Datasets/igb/igb-public.s3.us-east-2.amazonaws.com/"
        raw_path = "{}/{}/{}/processed/".format(ori_ds_path, ds_full_name_prefix, ds_full_name)

        num_nodes = 269346174
        label_file_bin = os.path.join(raw_path, "paper/node_label_19.bin") # float32
        node_feat_file_bin = os.path.join(raw_path, "paper/node_feat.bin") # float32

        edge_file =  os.path.join(raw_path, "paper__cites__paper/edge_index.npy")
        if not os.path.exists(edge_file_bin):
            edge = np.load(edge_file)
            edge_uint32 = edge.astype(np.uint32)
            src = edge_uint32[:,0].flatten()
            dst = edge_uint32[:,1].flatten()
            edge_file_bin =  os.path.join(raw_path, "paper/edge_index.bin")
            np.concatenate(src, dst).tofile(edge_file_bin) # uint32

        vc_json_file = "{}/vc_igb.json".format(raw_path)
        if os.path.exists(vc_json_file):
            print("found {}, return prepare_dataset_for_vc".format(vc_json_file))
            return vc_json_file

        vc_json = {}
        vc_json['node_label_file'] = label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feat_file_bin
        vc_json['split_file_path'] = "none"
        vc_json['num_nodes'] = 536870912
        vc_json['num_edges'] = 8589934592
        vc_json['feat_dim'] = int(128)
        vc_json['train_mask_file_bin'] = "nonde"
        vc_json['num_train_nodes'] = "none"
        vc_json['add_reverse_edge'] = False

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file

    if args.dataset == "ogbpa":
        ds_full_name = 'ogbn_papers100M'

        raw_path = "{}/{}/raw".format(ori_ds_path, ds_full_name)
        split_path = "{}/{}/split/time".format(ori_ds_path, ds_full_name)

        label_file = os.path.join(raw_path, "node-label.npz")
        train_mask_file = os.path.join(split_path, "train.csv.gz")
        valid_mask_file = os.path.join(split_path, "valid.csv.gz")
        test_mask_file = os.path.join(split_path, "test.csv.gz")
        data_file = os.path.join(raw_path, "data.npz")


#        data_dict=np.load(data_file, mmap_mode='r')
#        num_node_list = data_dict['num_nodes_list']
#        num_edge_list = data_dict['num_edges_list']
#        print(f"loading {valid_mask_file}")
#        valid_idx = th.as_tensor(pd.read_csv(valid_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
#        valid_mask = th.zeros((num_node_list[0],), dtype=th.bool)
#        valid_mask[valid_idx] = True
#        num_valid_nodes = valid_idx.shape[0]
#        print(f"valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}, num_valid_nodes={num_valid_nodes}")
#        valid_mask_file_bin=f"{raw_path}/valid_mask.bin"
#        valid_mask.numpy().tofile(valid_mask_file_bin)
        
#        print(f"loading {test_mask_file}")
#        test_idx = th.as_tensor(pd.read_csv(test_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
#        test_mask = th.zeros((num_node_list[0],), dtype=th.bool)
#        test_mask[test_idx] = True
#        num_test_nodes = test_idx.shape[0]
#        print(f"test_mask: shape={test_mask.shape}, dtype={test_mask.dtype}, num_test_nodes={num_test_nodes}")
#        test_mask_file_bin=f"{raw_path}/test_mask.bin"
#        test_mask.numpy().tofile(test_mask_file_bin)
#        exit(0)

        vc_json_file = "{}/vc_ogbpa.json".format(raw_path)
        if os.path.exists(vc_json_file):
            print("found {}, return prepare_dataset_for_vc".format(vc_json_file))
            return vc_json_file

        node_label=np.load(label_file, mmap_mode='r')
        lst = node_label.files
        node_label_file_bin = "{}/node_label.bin".format(raw_path)
        print("loading {}".format(label_file))
        for item in lst:
            print("{}\n".format(item))
            print(node_label[item])
            print("{}: shape:{}, dtype:{}".format(item, node_label[item].flatten().shape, node_label[item].dtype))
            node_label[item].flatten().tofile(node_label_file_bin)

        data_dict=np.load(data_file, mmap_mode='r')
        num_nodes_list = data_dict['num_nodes_list']
        num_edges_list = data_dict['num_edges_list']

        src, dst = dgl_g.edges()
        edge = th.concat((src, dst)).numpy()
        edge=edge.astype(np.uint32)
        edge_file_bin = "{}/edge_index.bin".format(raw_path)
        edge.tofile(edge_file_bin)

        assert len(num_edges_list) == 1, "ogbpa is homo"
        print("loading {}".format(data_file))
        for key in list(data_dict.keys()):
            if key == 'edge_index':
                continue # skipped edge index
                dt = data_dict[key].astype(np.uint32)
                print("{}: shape:{}, dtype:{}".format(key, dt.shape, dt.dtype))
                edge_file_bin = "{}/{}.bin".format(raw_path, key)
                dt.tofile(edge_file_bin)
            else:
                print("{}: shape:{}, dtype:{}".format(key, data_dict[key].shape, data_dict[key].dtype))
                if key == 'node_feat':
                    node_feat_dim=data_dict[key].shape[1]
                    if 'int' in str(data_dict[key].dtype):
                        node_feat = data_dict[key].astype(np.int64)
                    else:# float
                        node_feat = data_dict[key].astype(np.float32)
                    node_feats_file_bin="{}/{}.bin".format(raw_path, key)
                    node_feat.tofile(node_feats_file_bin)

        print(f"loading {train_mask_file}")
        train_idx = th.as_tensor(pd.read_csv(train_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        train_mask = th.zeros((num_nodes_list[0],), dtype=th.bool)
        train_mask[train_idx] = True
        num_train_nodes = train_idx.shape[0]
        print(f"train_mask: shape={train_mask.shape}, dtype={train_mask.dtype}, num_train_nodes={num_train_nodes}")
        train_mask_file_bin=f"{split_path}/train_mask.bin"
        train_mask.numpy().tofile(train_mask_file_bin)

        print(f"loading {valid_mask_file}")
        valid_idx = th.as_tensor(pd.read_csv(valid_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        valid_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        valid_mask[valid_idx] = True
        num_valid_nodes = valid_idx.shape[0]
        print(f"valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}, num_valid_nodes={num_valid_nodes}")
        valid_mask_file_bin=f"{raw_path}/valid_mask.bin"
        valid_mask.numpy().tofile(valid_mask_file_bin)
        
        print(f"loading {test_mask_file}")
        test_idx = th.as_tensor(pd.read_csv(test_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        test_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        test_mask[test_idx] = True
        num_test_nodes = test_idx.shape[0]
        print(f"test_mask: shape={test_mask.shape}, dtype={test_mask.dtype}, num_test_nodes={num_test_nodes}")
        test_mask_file_bin=f"{raw_path}/test_mask.bin"
        test_mask.numpy().tofile(test_mask_file_bin)

        vc_json = {}
        vc_json['node_label_file'] = node_label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feats_file_bin
        vc_json['split_file_path'] = split_path
        vc_json['num_nodes'] = int(num_nodes_list[0])
        vc_json['num_edges'] = int(num_edges_list[0])
        vc_json['feat_dim'] = int(node_feat_dim)
        vc_json['train_mask_file_bin'] = train_mask_file_bin
        vc_json['train_valid_file_bin'] = valid_mask_file_bin
        vc_json['train_test_file_bin'] = test_mask_file_bin
        vc_json['num_train_nodes'] = int(num_train_nodes)
        vc_json['add_reverse_edge'] = False

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file


    if args.dataset == "ogbpr":
        ds_full_name = 'ogbn_products'

        raw_path = "{}/{}/raw".format(ori_ds_path, ds_full_name)
        split_path = "{}/{}/split/sales_ranking".format(ori_ds_path, ds_full_name)

        label_file = os.path.join(raw_path, "node-label.csv.gz")
        train_mask_file = os.path.join(split_path, "train.csv.gz")
        valid_mask_file = os.path.join(split_path, "valid.csv.gz")
        test_mask_file = os.path.join(split_path, "test.csv.gz")
        edge_file = os.path.join(raw_path, "edge.csv.gz")
        feat_file = os.path.join(raw_path, "node-feat.csv.gz")
        num_node_list_file = os.path.join(raw_path, "num-node-list.csv.gz")
        num_edge_list_file = os.path.join(raw_path, "num-edge-list.csv.gz")

        vc_json_file = "{}/vc_ogbpr.json".format(raw_path)
        if os.path.exists(vc_json_file):
            return vc_json_file

        #edge = pd.read_csv(edge_file, compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        #edge=edge.astype(np.uint32)
        src, dst = dgl_g.edges()
        edge = th.concat((src, dst)).numpy()
        edge=edge.astype(np.uint32)
        edge_file_bin = "{}/edge_index.bin".format(raw_path)
        edge.tofile(edge_file_bin)

        node_feat = pd.read_csv(feat_file, compression='gzip', header = None).values
        num_node_list = pd.read_csv(num_node_list_file, compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(num_edge_list_file, compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
        node_label = pd.read_csv(label_file, compression='gzip', header = None).values

        assert len(num_edge_list) == 1, "ogbpa is homo"

        node_feats_file_bin="{}/node_feat.bin".format(raw_path)
        node_label_file_bin = "{}/node_label.bin".format(raw_path)

        #np.save(node_label_file_bin, node_label.flatten())
        nl = node_label.astype(np.float32).flatten()
        nl.tofile(node_label_file_bin)
        print("node_label.shape:{}, dtype:{}".format(nl.flatten().shape, nl.dtype))
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
        node_feat.tofile(node_feats_file_bin)
        print("node_feat.shape:{}, dtype:{}".format(node_feat.shape, node_feat.dtype))

        print(f"loading {train_mask_file}")
        train_idx = th.as_tensor(pd.read_csv(train_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        train_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        train_mask[train_idx] = True
        num_train_nodes = train_idx.shape[0]
        print(f"train_mask: shape={train_mask.shape}, dtype={train_mask.dtype}, num_train_nodes={num_train_nodes}")
        train_mask_file_bin=f"{raw_path}/train_mask.bin"
        train_mask.numpy().tofile(train_mask_file_bin)
        
        print(f"loading {valid_mask_file}")
        valid_idx = th.as_tensor(pd.read_csv(valid_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        valid_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        valid_mask[valid_idx] = True
        num_valid_nodes = valid_idx.shape[0]
        print(f"valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}, num_valid_nodes={num_valid_nodes}")
        valid_mask_file_bin=f"{raw_path}/valid_mask.bin"
        valid_mask.numpy().tofile(valid_mask_file_bin)
        
        print(f"loading {test_mask_file}")
        test_idx = th.as_tensor(pd.read_csv(test_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        test_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        test_mask[test_idx] = True
        num_test_nodes = test_idx.shape[0]
        print(f"test_mask: shape={test_mask.shape}, dtype={test_mask.dtype}, num_test_nodes={num_test_nodes}")
        test_mask_file_bin=f"{raw_path}/test_mask.bin"
        test_mask.numpy().tofile(test_mask_file_bin)

        vc_json = {}
        vc_json['node_label_file'] = node_label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feats_file_bin
        vc_json['split_file_path'] = split_path
        vc_json['num_nodes'] = int(num_node_list[0])
        vc_json['num_edges'] = int(edge.shape[0]/2) #int(num_edge_list[0])
        vc_json['feat_dim'] = int(node_feat.shape[1])
        vc_json['train_mask_file_bin'] = train_mask_file_bin
        vc_json['train_valid_file_bin'] = valid_mask_file_bin
        vc_json['train_test_file_bin'] = test_mask_file_bin
        vc_json['num_train_nodes'] = int(num_train_nodes)
        vc_json['add_reverse_edge'] = True

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file

    if args.dataset == "ogbar": # self-loop added after preprocessing
        ds_full_name = 'ogbn_arxiv'

        raw_path = "{}/{}/raw".format(ori_ds_path, ds_full_name)
        split_path = "{}/{}/split/time".format(ori_ds_path, ds_full_name)

        label_file = os.path.join(raw_path, "node-label.csv.gz")
        train_mask_file = os.path.join(split_path, "train.csv.gz")
        valid_mask_file = os.path.join(split_path, "valid.csv.gz")
        test_mask_file = os.path.join(split_path, "test.csv.gz")
        edge_file = os.path.join(raw_path, "edge.csv.gz")
        feat_file = os.path.join(raw_path, "node-feat.csv.gz")
        num_node_list_file = os.path.join(raw_path, "num-node-list.csv.gz")
        num_edge_list_file = os.path.join(raw_path, "num-edge-list.csv.gz")

        vc_json_file = "{}/vc_ogbar.json".format(raw_path)
        if os.path.exists(vc_json_file):
            return vc_json_file

        #edge = pd.read_csv(edge_file, compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        #edge=edge.astype(np.uint32)
        #print(edge.shape)
        #print(edge.dtype)


        node_feat = pd.read_csv(feat_file, compression='gzip', header = None).values
        num_node_list = pd.read_csv(num_node_list_file, compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(num_edge_list_file, compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
        node_label = pd.read_csv(label_file, compression='gzip', header = None).values

        assert len(num_edge_list) == 1, "ogbar is homo"

        edge_file_bin = "{}/edge_index.bin".format(raw_path)
        node_feats_file_bin="{}/node_feat.bin".format(raw_path)
        node_label_file_bin = "{}/node_label.bin".format(raw_path)

        #np.save(node_label_file_bin, node_label.flatten())
        nl = node_label.astype(np.float32).flatten()
        nl.tofile(node_label_file_bin)
        print("node_label.shape:{}, dtype:{}".format(nl.flatten().shape, nl.dtype))

        src, dst = dgl_g.edges()
        edge = th.concat((src, dst)).numpy()
        edge=edge.astype(np.uint32)
        edge.tofile(edge_file_bin)
        #print(edge)

        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
        node_feat.tofile(node_feats_file_bin)
        print("node_feat.shape:{}, dtype:{}".format(node_feat.shape, node_feat.dtype))

        print(f"loading {train_mask_file}")
        train_idx = th.as_tensor(pd.read_csv(train_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        train_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        train_mask[train_idx] = True
        num_train_nodes = train_idx.shape[0]
        print(f"train_mask: shape={train_mask.shape}, dtype={train_mask.dtype}, num_train_nodes={num_train_nodes}")
        train_mask_file_bin=f"{raw_path}/train_mask.bin"
        train_mask.numpy().tofile(train_mask_file_bin)
        
        print(f"loading {valid_mask_file}")
        valid_idx = th.as_tensor(pd.read_csv(valid_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        valid_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        valid_mask[valid_idx] = True
        num_valid_nodes = valid_idx.shape[0]
        print(f"valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}, num_valid_nodes={num_valid_nodes}")
        valid_mask_file_bin=f"{raw_path}/valid_mask.bin"
        valid_mask.numpy().tofile(valid_mask_file_bin)
        
        print(f"loading {test_mask_file}")
        test_idx = th.as_tensor(pd.read_csv(test_mask_file, compression='gzip', header = None).values.T[0]).to(th.long) # (num_graph, ) python list
        test_mask = th.zeros((num_node_list[0],), dtype=th.bool)
        test_mask[test_idx] = True
        num_test_nodes = test_idx.shape[0]
        print(f"test_mask: shape={test_mask.shape}, dtype={test_mask.dtype}, num_test_nodes={num_test_nodes}")
        test_mask_file_bin=f"{raw_path}/test_mask.bin"
        test_mask.numpy().tofile(test_mask_file_bin)

        vc_json = {}
        vc_json['node_label_file'] = node_label_file_bin
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feats_file_bin
        vc_json['split_file_path'] = split_path
        vc_json['num_nodes'] = int(num_node_list[0])
        vc_json['num_edges'] = int(edge.shape[0]/2) # edge here is the concated file #int(num_edge_list[0])
        vc_json['feat_dim'] = int(node_feat.shape[1])
        vc_json['train_mask_file_bin'] = train_mask_file_bin
        vc_json['train_valid_file_bin'] = valid_mask_file_bin
        vc_json['train_test_file_bin'] = test_mask_file_bin
        vc_json['num_train_nodes'] = int(num_train_nodes)
        vc_json['add_reverse_edge'] = True

        with open(vc_json_file, 'w+') as f:
            json.dump(vc_json, f, indent=4)
        return vc_json_file

    assert False, f"not supported dataset {args.dataset}"

def main(args):
    data_root_path=args.data_root_path
    ori_ds_path = "{rt}/ds_ori".format(rt=data_root_path)
    print("DATA root is: {0}".format(ori_ds_path))

    if args.part_algo[:2] == 'vc':
        dgl_g = None
        balance_ntypes = None
    else:
        dgl_g = load_dataset_into_memory(args, ori_ds_path)
        balance_ntypes = dgl_g.ndata['train_mask']
    print(dgl_g)

    vc_json_file = prepare_dataset_for_vc(args, ori_ds_path, dgl_g)
    with open(vc_json_file, "r") as f:
        vc_json = json.load(f)

    pre_ds_root = "{0}/ds_pre".format(data_root_path)
    part_out_path = "{root}/{ds}/data_part_n{num}_{algo}{algo_tag}_{hop}_{partial}".format(
            root=pre_ds_root, ds=args.dataset, num=args.n_parts, algo=args.part_algo, algo_tag=args.algo_tag, hop=args.num_hops,
            partial= f"{args.save_first_n_parts}" if args.save_first_n_parts > 0 else f"{args.n_parts}")
    part_config_path = "{out}/{ds}.json".format(out=part_out_path, ds=args.dataset)
    if not os.path.exists(part_out_path):
        os.makedirs(part_out_path)
    print("part_out_path="+part_out_path)
    print("part_config_path="+part_config_path)

    if not args.dist_part:
        if not args.only_read:
            
            materialize_node_feat = False
            if args.dataset == "cora" or args.dataset == "reddit":
                materialize_node_feat = True

            print(dgl_g)
            print('-'*50)
            dgl.distributed.partition_graph(dgl_g, graph_name=args.dataset, num_parts=args.n_parts,
                                            out_path=part_out_path,
                                            # TODO(ds4gnn): to understand
                                            part_method=args.part_algo,
                                            num_hops=args.num_hops,
                                            graph_formats = 'csr',
                                            balance_ntypes=balance_ntypes,
                                            balance_edges=True,  # TODO(ds4gnn): to understand
                                            return_mapping=True, # if args.part_algo[:2]!='vc' else False,
                                            vc_json=vc_json,
                                            save_first_n_parts = args.save_first_n_parts,
                                            materialize_node_feat=materialize_node_feat
                                            )
        # nmap, emap = partition_graph(..., return_mapping=True)
        print("Test load partition 0...")
        part_data = dgl.distributed.load_partition(part_config_path, 0)
        g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data  # unpack

        # dgl.distributed.load_partition_feats()
        # dgl.distributed.load_partition_book()
    else:
        # reference https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline
        print("Test load dist graph...")
        dgl.distributed.initialize("{pre_root}/clust_conf/config_1.txt".format(pre_root=pre_ds_root))
        g = dgl.distributed.DistGraph(
            args.dataset,
            part_config=part_config_path,
        )
        print(g)
    #    print(g.nodes)
    #    print(g.node_attr_schemes())
        print("Test load dist graph...=================")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        required=True,
        help="Dataset name ('reddit', 'cora', 'ogb-pr', 'obg-pa').",
    )
    parser.add_argument(
        "--data-root-path",
        type=str,
        default="DATA",
        required=True,
        help="relative path to root dir of DATA",
    )
    parser.add_argument(
        "--part-algo",
        type=str,
        default="random",
        help="Dataset name ('random', 'metis', 'vcrandom', 'vcoblivious', 'vchdrf').",
    )
    parser.add_argument(
        "--algo-tag",
        type=str,
        default="",
        help="Tag name to identify partitions of same scheme, e.g. 01, 02, 03,..20",
    )
    parser.add_argument(
        "--n-parts", type=int, required=True, help="number of graph partitions"
    )
    parser.add_argument(
        "--save-first-n-parts", type=int, default=-1, help="only save first n  (valid n > 0) partitions"
    )
    parser.add_argument(
        "--only-read",
        action="store_true",
        default=False,
        help="only test to read a partition",
    )
    parser.add_argument(
        "--dist-part",
        action="store_true",
        default=False,
        help="use distributed partition for big graph (default=False)",
    )
    parser.add_argument(
        "--self-loop",
        action="store_false",
        default=True,
        help="add self loop to graph (default=True)",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=1,
        help="number of halo hops",
    )
    parser.add_argument(
        "--one_id",
        type=str,
        default=None,
        required=False,
        help="",
    )
    args = parser.parse_args()
    print(args)

    main(args)
