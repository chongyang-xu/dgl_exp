import os
import argparse

import dgl
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

import torch as th
import json

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

def main(args):
    data_root_path=args.data_root_path
    ori_ds_path = "{rt}/ds_ori".format(rt=data_root_path)
    print("DATA root is: {0}".format(ori_ds_path))

    if args.part_algo[:2] == 'vc' and args.n_parts > 1:
        dgl_g = None
        balance_ntypes = None
        with open(args.vc_json) as f:
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
    parser.add_argument(
        "--vc-json",
        type=str,
        default="{}",
        help="path of a json file, keep meta info for vertex cut partition",
    )
    args = parser.parse_args()
    print(args)

    main(args)
