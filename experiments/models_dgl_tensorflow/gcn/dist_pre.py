import tensorflow as tf
import os
import argparse
import numpy as np

import dgl
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import NodePropPredDataset

assert os.environ.get("DGLBACKEND") == "tensorflow"


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def from_obg_lib_agnostic_dataset(ogb_la_ds):
    # reference https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#

    obg_lag, labels = ogb_la_ds[0]  # obg_lag: library-agnostic graph object

    with tf.device("/cpu:0"):
        dgl_g = dgl.graph((obg_lag['edge_index'][0], obg_lag['edge_index'][1]), num_nodes=obg_lag['num_nodes'])

        if obg_lag['node_feat'] is not None:
            dgl_g.ndata['feat'] = tf.convert_to_tensor(obg_lag['node_feat'])

        # dgl_g.edata["feat"] = tf.convert_to_tensor( obg_lag['edge_feat']))
        dgl_g.ndata['label'] = tf.convert_to_tensor(
            labels[:, 0], dtype=tf.int32)  # TODO(ds4gnn): should be nodel labels
        assert dgl_g.ndata['label'].shape[0] == obg_lag['num_nodes']

        split_idx = ogb_la_ds.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        train_mask = tf.convert_to_tensor(_sample_mask(
            train_idx, (obg_lag['num_nodes'],)), dtype=tf.bool)
        val_mask = tf.convert_to_tensor(_sample_mask(
            valid_idx, (obg_lag['num_nodes'],)), dtype=tf.bool)
        test_mask = tf.convert_to_tensor(_sample_mask(
            test_idx, (obg_lag['num_nodes'],)), dtype=tf.bool)

        dgl_g.ndata['train_mask'] = train_mask
        dgl_g.ndata['val_mask'] = val_mask
        dgl_g.ndata['test_mask'] = test_mask
        print(test_mask)
    return dgl_g

def main(args):
    if args.dataset == "ogb-pr":
        ds_full_name = 'ogbn-products'
        ogb_la_ds = NodePropPredDataset(ds_full_name)
        dgl_g = from_obg_lib_agnostic_dataset(ogb_la_ds)
    elif args.dataset == "ogb-pa":
        ds_full_name = 'ogbn-papers100M'
        ogb_la_ds = NodePropPredDataset(ds_full_name)
        dgl_g = from_obg_lib_agnostic_dataset(ogb_la_ds)
    elif args.dataset == "cora":
        # num_class = 7
        # in_feats = 1433
        ds_full_name = 'cora'
        dataset = CoraGraphDataset()
        dgl_g = dataset[0]
        dgl_g = dgl_g.to("/cpu:0")
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    print(dgl_g)

    if not args.dist_part:
        part_root_path = "/workspace/compiling/chongyang_tmp"
        part_out_path = part_root_path + "/" + ds_full_name + \
            "/" + str(str(args.n_parts)) + "_part_data"
        part_config_path = part_out_path + "/" + ds_full_name + ".json"
        if not os.path.exists(part_out_path):
            os.makedirs(part_out_path)
        print("part_out_path="+part_out_path)
        # nmap, emap = partition_graph(..., return_mapping=True)
        with tf.device("/cpu:0"):
            dgl.distributed.partition_graph(dgl_g, graph_name=ds_full_name, num_parts=args.n_parts,
                                            out_path=part_out_path,
                                            # TODO(ds4gnn): to understand
                                            balance_ntypes=dgl_g.ndata['train_mask'],
                                            balance_edges=True)  # TODO(ds4gnn): to understand
            print("Test load partition 0...")
            part_data = dgl.distributed.load_partition(part_config_path, 0)
            g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data  # unpack
            # dgl.distributed.load_partition_feats()
            # dgl.distributed.load_partition_book()
            print(g)
    else:
        # reference https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dis_pre")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        required=True
        help="Dataset name ('cora', 'ogb-pr', 'obg-pa').",
    )
    parser.add_argument(
        "--part-algo",
        type=str,
        default="random",
        help="Dataset name ('random', 'metis').",
    )
    parser.add_argument(
        "--n-parts", type=int, required=True, help="number of graph partitions"
    )
    parser.add_argument(
        "--dist-part",
        action="store_true",
        help="use distributed partition for big graph (default=False)",
    )
    parser.set_defaults(dist_part=False)

    args = parser.parse_args()
    print(args)

    main(args)
