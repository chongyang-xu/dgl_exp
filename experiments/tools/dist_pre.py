import os
import argparse

import dgl
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

import torch as th
import json
import numpy as np

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
    elif args.dataset == "ogbpa":
        ds_full_name = 'ogbn-papers100M'
        dgl_g, _ = load_ogb(ds_full_name, ori_ds_path)
    elif args.dataset == "cora":
        # num_class = 7
        # in_feats = 1433
        ds_full_name = 'cora'
        dataset = CoraGraphDataset()
        dgl_g = dataset[0]
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.self_loop and dgl_g is not None:
        dgl_g = dgl.remove_self_loop(dgl_g)
        dgl_g = dgl.add_self_loop(dgl_g)

    return dgl_g

def prepare_dataset_for_vc(args, ori_ds_path):
    if args.dataset == "ogbpa":
        ds_full_name = 'ogbn_papers100M'

        raw_path = "{}/{}/raw".format(ori_ds_path, ds_full_name)
        label_file = os.path.join(raw_path, "node-label.npz")
        data_file = os.path.join(raw_path, "data.npz")

        vc_json_file = "{}/vc_ogbpa.json".format(raw_path)
        if os.path.exists(vc_json_file):
            print("found {}, return prepare_dataset_for_vc".format(vc_json_file))
            return vc_json_file

        node_label=np.load(label_file, mmap_mode='r')
        lst = node_label.files
        print("loading {}".format(label_file))
        for item in lst:
            print("{}\n".format(item))
            print(node_label[item])
            print("{}: shape:{}, dtype:{}".format(item, node_label[item].flatten().shape, node_label[item].dtype))
            node_label[item].flatten().tofile("{}.bin".format(item))


        data_dict=np.load(data_file, mmap_mode='r')
        num_nodes_list = data_dict['num_nodes_list']
        num_edges_list = data_dict['num_edges_list']

        assert len(num_edges_list) == 1, "ogbpa is homo"
        print("loading {}".format(data_file))
        for key in list(data_dict.keys()):
            if key == 'edge_index':
                dt = data_dict[key].astype(np.uint32)
                print("{}: shape:{}, dtype:{}".format(key, dt.shape, dt.dtype))
                edge_file_bin = "{}/{}.bin".format(raw_path, key)
                dt.tofile(edge_file_bin)
            else:
                print("{}: shape:{}, dtype:{}".format(key, data_dict[key].shape, data_dict[key].dtype))
                if key == 'node_feat':
                    node_feat_dim=data_dict[key].shape[1]
                    node_feats_file_bin="{}/{}.bin".format(raw_path, key)
                    data_dict[key].tofile(node_feats_file_bin)
        vc_json = {}
        vc_json['edge_file_bin'] = edge_file_bin
        vc_json['node_feats_file_bin'] = node_feats_file_bin
        split_file_path = "{}/{}/split/time".format(ori_ds_path, ds_full_name)
        vc_json['split_file_path'] = split_file_path
        vc_json['num_nodes'] = num_nodes_list[0]
        vc_json['num_edges'] = num_edges_list[0]
        vc_json['feat_dim'] = node_feat_dim

        with open(vc_json_file, 'w') as f:
            json.dump(vc_json, f)
    if args.dataset == "ogbpr":
        ds_full_name = 'ogbn_products'

        raw_path = "{}/ds_ori/{}/raw".format(ori_ds_path, ds_full_name)
        vc_json_file = "{}/vc_ogbpr.json".format(raw_path)
        if os.path.exists(vc_json_file):
            return vc_json_file
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

def main(args):
    data_root_path=args.data_root_path
    ori_ds_path = "{rt}/ds_ori".format(rt=data_root_path)
    print("DATA root is: {0}".format(ori_ds_path))

    if args.part_algo[:2] == 'vc' and args.n_parts > 1:
        dgl_g = None
        balance_ntypes = None
        vc_json_file = prepare_dataset_for_vc(args, ori_ds_path)
        with open(vc_json_file) as f:
            vc_json = json.load(f)
    else:
        dgl_g = load_dataset_into_memory(args, ori_ds_path)
        balance_ntypes = dgl_g.ndata['train_mask']
        vc_json = None
    print(dgl_g)

    pre_ds_root = "{0}/ds_pre".format(data_root_path)
    part_out_path = "{root}/{ds}/data_part_n{num}_{algo}".format(
            root=pre_ds_root, ds=args.dataset, num=args.n_parts, algo=args.part_algo)
    part_config_path = "{out}/{ds}.json".format(out=part_out_path, ds=args.dataset)
    if not os.path.exists(part_out_path):
        os.makedirs(part_out_path)
    print("part_out_path="+part_out_path)
    print("part_config_path="+part_config_path)

    if not args.dist_part:
        if not args.only_read:
            dgl.distributed.partition_graph(dgl_g, graph_name=args.dataset, num_parts=args.n_parts,
                                            out_path=part_out_path,
                                            # TODO(ds4gnn): to understand
                                            part_method=args.part_algo,
                                            num_hops=args.num_hops,
                                            graph_formats = 'csr',
                                            balance_ntypes=balance_ntypes,
                                            balance_edges=True,  # TODO(ds4gnn): to understand
                                            vc_json=vc_json)
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
            part_config=part_config_path
        )
        print(g)
    #    print(g.nodes)
    #    print(g.node_attr_schemes())
        print("Test load dist graph...=================")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dis_pre")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        required=True,
        help="Dataset name ('cora', 'ogb-pr', 'obg-pa').",
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
        "--n-parts", type=int, required=True, help="number of graph partitions"
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
    args = parser.parse_args()
    print(args)

    main(args)
