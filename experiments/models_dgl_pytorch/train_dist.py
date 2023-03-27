import argparse
import socket
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

from gcn import GCN
from graphsage import SAGE as GraphSAGE
from infer_dist import inference as dist_model_inference


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["feat"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["label"][seeds].to(device)
    return batch_inputs, batch_labels


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device, stop_at_border):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = dist_model_inference(model, g, inputs, batch_size, device, stop_at_border)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    shuffle = True
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],
        stop_at_border=args.stop_at_border
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    # Define model and optimizer
    if args.model == 'sage':
        model = GraphSAGE(in_feats, args.num_hidden, n_classes,
                          args.num_layers, F.relu, args.dropout,)
    else:
        assert args.model == 'gcn'
        model = GCN(in_feats, args.num_hidden, n_classes,
                    args.num_layers, F.relu, args.dropout,)

    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    iter_tput = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        g_copy_time = 0  # graph struct copy time
        f_copy_time = 0  # feature copy time
        forward_time = 0
        backward_time = 0
        update_time = 0
        account_time = 0  # calculating time for log
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()
                sample_time += tic_step - start
                # fetch features/labels
                batch_inputs, batch_labels = load_subtensor(
                    g, seeds, input_nodes, "cpu"
                )
                g_copy_end = time.time()
                g_copy_time += g_copy_end - tic_step
                batch_labels = batch_labels.long()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                # move to target device
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                f_copy_end = time.time()
                f_copy_time += f_copy_end - g_copy_end
                # Compute loss and prediction
                #start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - f_copy_end
                backward_time += compute_end - forward_end

                optimizer.step()
                update_end = time.time()
                update_time += update_end - compute_end

                step_t = update_end - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s\n".format(
                            g.rank(),
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                        )
                    )
                account_end = time.time()
                account_time += account_end - update_end
                start = account_end
        toc = time.time()
        print(
            "Part {}, Epoch Time(s): {:.4f}, sampling: {:.4f}, g_copy: {:.4f}, f_copy: {:.4f}, "
            "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, account: {:.4f}, #seeds: {}, "
            "#inputs: {}\n".format(
                g.rank(),
                toc - tic,
                sample_time,
                g_copy_time,
                f_copy_time,
                forward_time,
                backward_time,
                update_time,
                account_time,
                num_seeds,
                num_inputs,
            )
        )
        epoch += 1

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(
                model if args.standalone else model.module,
                g,
                g.ndata["feat"],
                g.ndata["label"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
                args.stop_at_border,
            )
            print(
                "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format
                (
                    g.rank(), val_acc, test_acc, time.time() - start
                )
            )


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    t_b = time.time()
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type)
    print("local_rank={}, initialize TIME={:.4f} sec".format(
        args.local_rank, time.time()-t_b))
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        t_b = time.time()
        th.distributed.init_process_group(backend=args.backend)
        print("local_rank={}, init_process_group TIME={:.4f} sec".format(
            args.local_rank, time.time()-t_b))
    print(socket.gethostname(), "Initializing DistGraph")
    t_b = time.time()
    g = dgl.distributed.DistGraph(
        args.graph_name,
        part_config=args.part_config
    )
    if args.stop_at_border:
        print("=========================warning!===================")
        print("===self loops are required on preprocess dataset====")
        print("=========================warning!===================")

    print("local_rank={}, g.dev={}, DistGraph TIME={:.4f} sec".format(
        args.local_rank, g.device, time.time()-t_b))
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    force_even_flg = False if args.stop_at_border else True
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=force_even_flg,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=force_even_flg,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=force_even_flg,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        # TODO(ds4gnn):is train_nid local to current partition?
        # TODO(ds4gnn):
        # force enven will divide all train nodes evenly across partitions,
        # thus, a trainer might be asigned remote train node
        # the idea of "sampling stop at border" don't want to handle remote sampling
        # thus force_even is set to False
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=force_even_flg
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=force_even_flg
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=force_even_flg
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    #in_feats = g.ndata["features"].shape[1]
    in_feats = g.ndata["feat"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
#    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        required=True,
        help="gnn model in(sage, gcn)",
    )
    parser.add_argument(
        "--stop_at_border", action="store_true", default=False, help="sampler will stop at border"
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="15,10,5")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local-rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
             "of batches to be the same.",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="socket",
        help="backend net type, 'socket' or 'tensorpipe'",
    )
    args = parser.parse_args()

    print(args)
    main(args)
