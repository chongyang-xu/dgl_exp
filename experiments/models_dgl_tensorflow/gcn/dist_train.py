from ogb.nodeproppred import NodePropPredDataset
import argparse
import time
import json
import os
from multiprocessing import util

import numpy as np
import tensorflow as tf
from gcn import GCN
from graphsage import GraphSAGE

import dgl
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.dataloading import NeighborSampler
from dgl.dataloading import DistNodeDataLoader
# from dgl.dataloading import DataLoader #DataLoader only support pytorch

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()  # for enumerate on DistNodeDataLoader


tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


def evaluate(model, g, features, labels, mask):
    logits = model(g, features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc.numpy().item()

# Checkpoint saving and restoring


def _is_chief(task_type, task_id, cluster_spec):
    return (task_type is None
            or task_type == 'chief'
            or (task_type == 'worker'
                and task_id == 0
                and 'chief' not in cluster_spec.as_dict()))


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


def get_train_strategy(args):
    if args.n_worker == 1:
        # reference https://www.tensorflow.org/tutorials/distribute/custom_training
        train_strategy = tf.distribute.MirroredStrategy(
            devices=["/gpu:0", "/gpu:1"],
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
    elif args.n_workser > 1:
        # reference https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "worker": ["host1:port", "host2:port"],
            },
            "task": {"type": "worker", "index": args.worker_idx}
        })
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.RING)  # RING(rpc), NCCL, AUTO
        train_strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=communication_options)
    else:
        raise ValueError("Invalid worker num: {}".format(args.n_workser))
    return train_strategy


def load_g(args):
    # load and preprocess dataset
    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = dataset[0]

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    #labels = g.ndata["label"]
    features = g.ndata["feat"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    in_feats = features.shape[1]
    n_classes = dataset.num_classes

    print(
        """----Data statistics------'
    #Classes %d
    #Feature %d
    #Train samples %d
    #Val samples %d
    #Test samples %d"""
        % (
            n_classes,
            in_feats,
            train_mask.numpy().sum(),
            val_mask.numpy().sum(),
            test_mask.numpy().sum(),
        )
    )
    return g


def load_dist_g(args):
    # load preprocessed distGraph
    if args.dataset == "ogb-pr":
        p_config = "/workspace/compiling/chongyang_tmp/ogbn-products/1_part_data/ogbn-products.json"
        g = dgl.distributed.DistGraph('ogbn-products', part_config=p_config)
    elif args.dataset == "ogb-pa":
        g = dgl.distributed.DistGraph('ogbn-papers100M')
    elif args.dataset == "cora":
        p_config = "/workspace/compiling/chongyang_tmp/cora/1_part_data/cora.json"
        g = dgl.distributed.DistGraph('cora', part_config=p_config)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    print("dataset={ds}, {field}:{detail}".format(
        ds=args.dataset, field="label", detail=str(g.ndata["label"])))
    print("dataset={ds}, {field}:{detail}".format(
        ds=args.dataset, field="train_mask", detail=str(g.ndata["train_mask"])))
    print("dataset={ds}, {field}:{detail}".format(
        ds=args.dataset, field="val_mask", detail=str(g.ndata["val_mask"])))
    print("dataset={ds}, {field}:{detail}".format(
        ds=args.dataset, field="test_mask", detail=str(g.ndata["test_mask"])))
    return g


def get_dataset_fn_for_dist(g):
    def fn(input_context):
        def input_gen():
            yield g.ndata["feat"], g.ndata["label"], g.ndata["train_mask"]
            yield g.ndata["feat"], g.ndata["label"], g.ndata["train_mask"]

        ds = tf.data.Dataset.from_generator(
            input_gen, output_signature=(
                tf.TensorSpec(shape=g.ndata["feat"].shape,
                              dtype=g.ndata["feat"].dtype),
                tf.TensorSpec(shape=g.ndata["label"].shape,
                              dtype=g.ndata["label"].dtype),
                tf.TensorSpec(shape=g.ndata["train_mask"].shape,
                              dtype=g.ndata["train_mask"].dtype))
        )
        #ds = ds.prefetch(2)
        return ds
    return fn


def main(args):

    dgl.distributed.initialize(ip_config='ip_config.txt')

    # set up parallel train strategy
    train_strategy = get_train_strategy(args)

    checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

    dgl_dist_g = load_dist_g(args)
    # prepare data for dist trainer
    # train_dist_dataset = train_strategy.experimental_distribute_dataset(dataset)
    # test_dist_dataset = train_strategy.experimental_distribute_dataset(dataset)

    with tf.device("/cpu:0"):
        train_nid = dgl.distributed.node_split(
                dgl_dist_g.ndata['train_mask'], force_even=True)
        sampler = dgl.dataloading.NeighborSampler([25, 10])
        dist_train_dataloader = dgl.dataloading.DistNodeDataLoader(
            dgl_dist_g, train_nid, sampler, batch_size=1024,shuffle=True, drop_last=False)

    #valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])
    # valid_dataloader = dgl.dataloading.DistNodeDataLoader(
    #                            g, valid_nid, sampler, batch_size=1024,
    #                            shuffle=False, drop_last=False)

    #dataset_fn = get_dataset_fn_for_dist(g)
    # multi_worker_dataset = train_strategy.distribute_datasets_from_function(
    #    dataset_fn)

    with train_strategy.scope():
        # create GCN model
        if args.model == 'gcn':
            model = GCN(
                args.in_feats, args.n_hidden, args.n_classes, args.n_layers, tf.nn.relu, args.dropout,)
        else:
            assert (args.model == 'sage')
            model = GraphSAGE(
                args.in_feats, args.n_hidden, args.n_classes, args.n_layers, tf.nn.relu, args.dropout,)

        # use optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr, epsilon=1e-8)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        g = dgl_dist_g

    @tf.function
    def val_step(iterator):
        # https://www.tensorflow.org/tutorials/distribute/custom_training
        def step_fn(inputs):
            test_loss = []
            test_accuracy = []
        return train_strategy.run(step_fn, args=(next(iterator),))

    @tf.function
    def train_step(iterator):
        """Training step function."""
        def step_fn(inputs):
            (input_nodes, seeds, blocks) = inputs
            batch_features = g.ndata['feat'][input_nodes]
            batch_labels = g.ndata['label'][seeds]

            with tf.GradientTape() as tape:
                logits = model(blocks, batch_features)
                loss_value = tf.keras.losses.SparseCategoricalCrossentropy(
                    reduction=tf.keras.losses.Reduction.SUM, from_logits=True)(batch_labels, logits)

                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in model.trainable_weights:
                    loss_value = loss_value + args.weight_decay * tf.nn.l2_loss(
                        weight
                    )
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_accuracy.update_state(
                    labels[train_mask], logits[train_mask])
                return loss_value
        per_replica_losses = train_strategy.run(
            step_fn, args=(next(iterator),))
        return train_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
    step_in_epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.dtypes.int64),
        name='step_in_epoch')

    # checkpoint
    if args.use_checkpoint:
        task_type, task_id, cluster_spec = (train_strategy.cluster_resolver.task_type,
                                            train_strategy.cluster_resolver.task_id,
                                            train_strategy.cluster_resolver.cluster_spec())

        checkpoint = tf.train.Checkpoint(
            model=model, epoch=epoch, step_in_epoch=step_in_epoch)
        write_checkpoint_dir = write_filepath(
            checkpoint_dir, task_type, task_id, cluster_spec)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
        # Restoring the checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)

    # train loop
    n_epoch = args.n_epochs
    step_per_epoch = args.n_steps_per_epoch
    while epoch.numpy() < n_epoch:
        iterator = iter(dist_train_dataloader)
        total_loss = 0.0
        num_batches = 0

        while step_in_epoch.numpy() < step_per_epoch:
            total_loss += train_step(iterator)
            num_batches += 1
            step_in_epoch.assign_add(1)
        train_loss = total_loss / num_batches

        # val_step(iterator)
        print('Epoch: %d, accuracy: %f, train_loss: %f.'
              % (epoch.numpy(), train_accuracy.result(), train_loss))

        train_accuracy.reset_states()
        # checkpoint
        if args.use_checkpoint:
            checkpoint_manager.save()
            if not _is_chief(task_type, task_id, cluster_spec):
                tf.io.gfile.rmtree(write_checkpoint_dir)

        epoch.assign_add(1)
        step_in_epoch.assign(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'ogb-pr', 'obg-pa').",
    )
    parser.add_argument(
        "--n-classes", type=int, default=-1, required=True, help="number of classes in dataset"
    )
    parser.add_argument(
        "--in-feats", type=int, default=-1, required=True, help="feature dimension in dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        help="Dataset name ('gcn', 'sage').",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument(
        "--n-steps-per-epoch", type=int, default=1, help="number of steps in one epoch"
    )
    parser.add_argument(
        "--batch-size-per-gpu", type=int, default=200, help="batch_size for 1 gpu, global batchsize will be calculated for dist training"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.set_defaults(self_loop=False)
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="use checkpoint (default=False)",
    )
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument(
        "--n-worker", type=int, default=1, help="number of total workers"
    )
    parser.add_argument(
        "--worker-idx", type=int, default=0, help="index of worker self"
    )
    args = parser.parse_args()
    print(args)

    main(args)
