import argparse
import socket
import time
import yaml

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

from gcn import GCN
from graphsage import SAGE as GraphSAGE
from gat import GAT
from infer_dist import inference as dist_model_inference

import os
import datetime

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
    n_correct = (th.argmax(pred, dim=1) == labels).float().sum()
    n_correct_total = th.Tensor([n_correct, float(len(pred))])
    th.distributed.all_reduce(n_correct_total, async_op=False)
    return n_correct_total


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device, stop_at_border, grouping_hack=False):
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
        pred = dist_model_inference(model, g, inputs, batch_size, device, stop_at_border, grouping_hack)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, train_controller):

    ##################################
    # Define sampler and dataloader
    ##################################
    train_controller.init_sampler(args.fan_out, args.stop_at_border)
    shuffle = True
    train_controller.init_dataloader(args.batch_size, shuffle)

    ##################################
    # Define model and optimizer
    ##################################
    n_classes = train_controller.get_n_classes()
    in_feats  = train_controller.get_in_feats()
    if args.model == 'sage':
        model = GraphSAGE(in_feats, args.num_hidden, n_classes,
                          args.num_layers, F.relu, args.dropout,)
        model.name = 'sage'
    elif args.model == 'gat':
        n_heads = 4
        model = GAT(in_feats, args.num_hidden, n_classes, n_heads,
                          args.num_layers, F.relu, args.dropout,)
        model.name = 'gat'
    else:
        assert args.model == 'gcn'
        model = GCN(in_feats, args.num_hidden, n_classes,
                    args.num_layers, F.relu, args.dropout,)
        model.name = 'gcn'

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

    ##################################
    # Define checkpoint path
    ##################################

    # Training loop
    iter_tput = []
    t_epoch = -1
    while train_controller.next_epoch(model, optimizer):
        CKPT_PATH="/workspace/dgl_dsg/experiments"
        CKPT_NAME="gcn2_ckpt_epoch_500_mode3"
        CKPT_TAG=500
        if False:
            start = time.time()
            ckpt = th.load(CKPT_PATH + "/" + CKPT_NAME)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            g = train_controller.get_infer_g()
            print("use checkpoint for inference....")
            model_infer = model
            #train_controller.load_best_model(model_infer)
            val_acc_t, test_acc_t = evaluate(
                model_infer if args.standalone else model_infer.module,
                g,
                g.ndata["feat"],
                g.ndata["label"],
                train_controller.get_infer_val_nid(),
                train_controller.get_infer_test_nid(),
                args.batch_size_eval,
                device,
                args.stop_at_border,
            )
            print("infer_|epoch|{:04d}|part|{:04d}|val_acc|{:.4f}|test_acc|{:.4f}|time_sec|{:.4f}|val:{:.1f},{:.1f}|test:{:.1f},{:.1f}".format(
                        train_controller.get_epoch(),
                        train_controller.get_rank(),
                        val_acc_t[0].item()  / val_acc_t[1].item(),
                        test_acc_t[0].item() / test_acc_t[1].item(),
                        time.time() - start,
                        val_acc_t[0], val_acc_t[1],
                        test_acc_t[0], test_acc_t[1],
                    )
                )
            exit(0)


        t_epoch = t_epoch + 1
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

        epoch_step_loss = []

        acc = th.tensor([0.0])

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate( train_controller.get_dataloader() ):
                tic_step = time.time()
                sample_time += tic_step - start
                # fetch features/labels
                batch_inputs, batch_labels = load_subtensor(
                    train_controller.get_g(), seeds, input_nodes, "cpu"
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

                epoch_step_loss.append(loss.item())

                if step % args.log_every == 0:
                    acc_t = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )

                    print("step_|t_epoch|{:04d}|epoch|{:04d}|step|{:04d}|part|{:04d}|loss|{:.4f}|"
                            "train_acc|{:.4f}|sample_p_s|{:.2f}|gpu_mb|{:.1f}|step_time|{:.2f}|train:{:.1f},{:.1f}".format(
                                t_epoch,
                                train_controller.get_epoch(),
                                step,
                                train_controller.get_rank(),
                                loss.item(),
                                acc_t[0].item()/acc_t[1].item(),
                                np.mean(iter_tput[3:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every:]),
                                acc_t[0].item(), acc_t[1].item()
                               )
                         )

                account_end = time.time()
                account_time += account_end - update_end
                start = account_end

        ####################################
        # compute validation accuracy
        ####################################
        model.eval()
        with model.join():
        # validation
            vt_start = time.time()
            acc_vs = []
        
            for step, (input_nodes, seeds, blocks) in enumerate( train_controller.get_valid_dataloader() ):
                batch_inputs, batch_labels = load_subtensor(
                    train_controller.get_g(), seeds, input_nodes, "cpu"
                )
                batch_labels = batch_labels.long()
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_pred = model(blocks, batch_inputs)
                acc_v = compute_acc(batch_pred, batch_labels)
                acc_vs.append(acc_v[0].item()/acc_v[1].item())
            vt_end = time.time()
        model.train()
        toc = time.time()

        epoch_step_loss = np.mean(epoch_step_loss)
        train_controller.epoch_end(model, optimizer, acc.item(), np.mean(acc_vs),  epoch_step_loss) # acc is last acc in cur epoch

        print("epoch_|t_epoch|{:04d}|epoch|{:04d}|part|{:04d}|epoch_seconds|{:.4f}|sampling|{:.4f}|g_copy|{:.4f}|f_copy|{:.4f}|"
                "forward|{:.4f}|backward|{:.4f}|update|{:.4f}|account|{:.4f}|vt|{:.4f}|n_seed|{:012d}|n_input|{:012d}".format(
                    t_epoch,
                    train_controller.get_epoch(),
                    train_controller.get_rank(),
                    toc - tic,
                    sample_time,
                    g_copy_time,
                    f_copy_time,
                    forward_time,
                    backward_time,
                    update_time,
                    account_time,
                    vt_end - vt_start,
                    num_seeds,
                    num_inputs,
                  )
              )

        if False and train_controller.get_epoch() + 1 == CKPT_TAG:
            if th.distributed.get_rank() == 0:
                th.save({
                    'epoch': CKPT_TAG,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': 0,
                }, CKPT_NAME)


        if (train_controller.get_epoch() + 1) % args.eval_every == 0 and train_controller.get_epoch() != 0:
            start = time.time()
            g = train_controller.get_infer_g()
            print("use best model so far for inference....")
            model_infer = model
            train_controller.load_best_model(model_infer)
            val_acc_t, test_acc_t = evaluate(
                model_infer if args.standalone else model_infer.module,
                g,
                g.ndata["feat"],
                g.ndata["label"],
                train_controller.get_infer_val_nid(),
                train_controller.get_infer_test_nid(),
                args.batch_size_eval,
                device,
                False if g.get_partition_book().num_partitions() != 256 else True, # do full graph inference, dont use stop-at-the-border for inference #args.stop_at_border
                grouping_hack=True, # flag for grouping mode, add infer policy
            )
            print("infer_|epoch|{:04d}|part|{:04d}|val_acc|{:.4f}|test_acc|{:.4f}|time_sec|{:.4f}|val:{:.1f},{:.1f}|test:{:.1f},{:.1f}".format(
                        train_controller.get_epoch(),
                        train_controller.get_rank(),
                        val_acc_t[0].item()  / val_acc_t[1].item(),
                        test_acc_t[0].item() / test_acc_t[1].item(),
                        time.time() - start,
                        val_acc_t[0], val_acc_t[1],
                        test_acc_t[0], test_acc_t[1],
                    )
                )

def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    t_b = time.time()
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type,
                               disable_backup_server=args.disable_backup_server)
    print("local_rank={}, initialize TIME={:.4f} sec".format(
        args.local_rank, time.time()-t_b))

    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        t_b = time.time()

        if args.backend == "gloo":
            th.distributed.init_process_group(backend=args.backend)
        elif args.backend == "nccl":
            master_ip   = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']
            w_size      = int(os.environ['ROLE_WORLD_SIZE'])
            w_rank      = int(os.environ['ROLE_RANK'])
            os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            os.environ['NCCL_DEBUG']='WARN'
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip=master_ip, master_port='1234')
            th.distributed.init_process_group(backend=args.backend,
                                              init_method=dist_init_method,
                                              world_size=w_size,
                                              rank=w_rank,
                                              timeout=datetime.timedelta(seconds=10)
                                             )
        print("local_rank={}, init_process_group TIME={:.4f} sec".format(
            args.local_rank, time.time()-t_b))


    train_controller = dgl.distributed.TrainController(args.local_rank, args.num_epochs, mode=args.train_ctrl_mode)
    with open(args.graph_data_config, 'r') as f:
        graph_data_config = yaml.safe_load(f)
    train_controller.init_dist_graph_set(graph_data_config,
                                        args.stop_at_border,
                                        args.num_gpus)

    device = train_controller.get_assigned_device()

    run(args, device, train_controller)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument(
        "--graph_data_config", type=str, help="The path to the partition config yaml file"
    )
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
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
        help="gnn model in(sage, gcn, gat)",
    )
    parser.add_argument(
        "--stop_at_border", action="store_true", default=False, help="sampler will stop at border"
    )
    parser.add_argument(
        "--disable_backup_server", action="store_true", default=False, help="when enabled, 1 graph server will serve 1 partition, no backup servers"
    )
    parser.add_argument("--train_ctrl_mode", type=int, default=4)
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
        "--local_rank", type=int, help="get rank of the process"
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
