import argparse
import time
import json
import os
from multiprocessing import util

import numpy as np
import tensorflow as tf
from gcn import GCN

import dgl
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

# data preparation
# val set, test set
# graphsage

def evaluate(model, features, labels, mask):
    logits = model(features, training=False)
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


def load_dataset(args):
    # load and preprocess dataset
    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    train_mask = dataset[0].ndata["train_mask"]
    val_mask = dataset[0].ndata["val_mask"]
    test_mask = dataset[0].ndata["test_mask"]
    n_classes = dataset.num_classes
    print(
        """----Data statistics------'
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d"""
        % (
            n_classes,
            train_mask.numpy().sum(),
            val_mask.numpy().sum(),
            test_mask.numpy().sum(),
        )
    )
    return dataset


def main(args):
    # set up parallel train strategy
    train_strategy = get_train_strategy(args)

    checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

#    train_dist_dataset = train_strategy.experimental_distribute_dataset(dataset)
#    test_dist_dataset = train_strategy.experimental_distribute_dataset(dataset)

    dataset = load_dataset(args)
    g = dataset[0]
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    in_feats = g.ndata["feat"].shape[1]
    n_classes = dataset.num_classes

    def dataset_fn(input_context):
        def input_gen():
            count = 0
            while count < 3:
                count = count + 1
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
        return ds

    with train_strategy.scope():
        # create GCN model
        model = GCN(
            g, in_feats, args.n_hidden, n_classes, args.n_layers, tf.nn.relu, args.dropout,)
        # use optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr, epsilon=1e-8)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        multi_worker_dataset = train_strategy.distribute_datasets_from_function(
            dataset_fn)

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
            features, labels, train_mask = inputs

            with tf.GradientTape() as tape:
                logits = model(features)
                loss_value = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)(labels[train_mask], logits[train_mask])

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
        iterator = iter(multi_worker_dataset)
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
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
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
