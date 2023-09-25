from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
import socket
import time
import torch as th

import dgl

CONTROL_STATE_WINDOW_LENGTH = 3
AVERAGE_LOCAL_ACC_WINDOW_LENGTH = 4
AVG_DELTA_THRESHOLD = 0.005

class ControlState(Enum):
    PLATEAU = 0
    INCREASING = 1
    DECREASING = 2

class StateWindow:
    def __init__(self, length):
        self.window = []
        self.WINDOW_LENGTH = length

    def push(self, state):
        while len(self.window) >= self.WINDOW_LENGTH:
            self.window.pop(0)
        self.window.append(state)

    def get_list(self):
        return self.window

    def count_control_state(self):
        plateau_cnt, increasing_cnt, decresing_cnt = 0, 0, 0
        for e in self.window:
            if e == ControlState.PLATEAU:
                plateau_cnt = plateau_cnt + 1
            if e == ControlState.INCREASING:
                increasing_cnt = increasing_cnt + 1
            if e == ControlState.DECREASING:
                decresing_cnt = decresing_cnt + 1
        return plateau_cnt, increasing_cnt, decresing_cnt

class PartitionSwitcher(ABC):
    # continue to next epoch
    @abstractmethod
    def keep(self):
        pass

    # switch a dataloader either eagerly or lazyly
    @abstractmethod
    def switch(self):
        pass

    # restore to stored ckpt
    @abstractmethod
    def fall_back(self, model, opt):
        pass

    # add one ckpt
    @abstractmethod
    def seal(self, model, opt, acc):
        pass

    # set stop train flag
    @abstractmethod
    def stop(self):
        pass

# train controller must be initialized after 
# dgl.distributed.initialize()
# th.distributed.init_process_group(backend=args.backend)
class TrainController(PartitionSwitcher):
    def __init__(self, local_rank, train_epoch_n):
        self.dist_graph_set  = [] # the list of all disgraph
        self.dist_graph_pbs  = [] # the coresponding partition books of dist graph
        self.train_nid_sets  = [] # the coresponding train_nid of the dist graph
        self.val_nid_sets    = [] # the coresponding val_nid of the dist graph
        self.test_nid_sets   = [] # the coresponding test_nid of the dist graph
        self.n_classes       = 0
        self.in_feats        = 0

        self.local_rank       = 0
        self.num_gpus         = 0

        self.sampler          = None
        self.dataloader       = None
        self.dataloader_set   = []
        self.dataloader_idx   = 0

        self.resume_path      = None
        self.checkpoint_path  = None
        self.checkpoint_every = -1

        self.epoch_idx        = -1
        self.train_epoch_n    = train_epoch_n


        self.control_state = ControlState.PLATEAU
        self.control_state_window = StateWindow(CONTROL_STATE_WINDOW_LENGTH)
        self.local_train_acc_window = StateWindow(AVERAGE_LOCAL_ACC_WINDOW_LENGTH)
        self.AVG_DELTA_THRESHOLD = AVG_DELTA_THRESHOLD
        for _ in range(CONTROL_STATE_WINDOW_LENGTH):
            self.control_state_window.push(ControlState.INCREASING)
        for _ in range(AVERAGE_LOCAL_ACC_WINDOW_LENGTH):
            self.local_train_acc_window.push(0)

        self.sealed_ckpt = []


    # mode: 'eager': load all graph into memory, 'lazy': load the graph when needed 
    def init_dist_graph_set(self, data_config_yaml, stop_at_the_border, num_gpus, mode='eager'):

        print(socket.gethostname(), "Initializing DistGraph Set")
        t_b = time.time()

        if stop_at_the_border:
            print("=========================warning!===================")
            print("===self loops are required on preprocess dataset====")
            print("=========================warning!===================")

        for part_config in data_config_yaml["partitions"]:
            graph_name = list(part_config.keys())[0]
            cfg_file = list(part_config.values())[0]

            g = dgl.distributed.DistGraph(
                graph_name,
                part_config=cfg_file,
                partition_grouping_mode=True,
            )
            pb = g.get_partition_book()

            self.dist_graph_set.append(g)
            self.dist_graph_pbs.append(pb)

            force_even_flg = False if stop_at_the_border else True

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

            self.train_nid_sets.append(train_nid)
            self.val_nid_sets.append(val_nid)
            self.test_nid_sets.append(test_nid)

            local_nid = pb.partid2nids(pb.partid).detach().numpy()
            # train_nid is lid when use VCMapPartition
            print(
                "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
                "(local: {}) [#local invalid for vc* partitions]".format(
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

            print("local_rank={}, g.dev={}, DistGraph TIME={:.4f} sec".format(
                self.local_rank, g.device, time.time()-t_b))
            print(socket.gethostname(), "rank:", g.rank())
            print("=" * 50)

        self.local_rank = self.dist_graph_set[0].rank()

        if self.n_classes == 0:
            self.n_classes = self.get_n_classes()

        if self.local_rank == 0:
            print("#labels:", self.n_classes)

        self.in_feats = self.dist_graph_set[0].ndata["feat"].shape[1]

        self.num_gpus = num_gpus

    def get_n_classes(self):
        g = self.dist_graph_set[0]
        labels = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
        return n_classes

    def get_in_feats(self):
        return self.dist_graph_set[0].ndata["feat"].shape[1]

    def get_g(self):
        return self.dist_graph_set[self.dataloader_idx]

    def get_epoch(self):
        return self.epoch_idx

    def get_rank(self):
        return self.local_rank

    def get_assigned_device(self):
        if self.num_gpus == -1:
            device = th.device("cpu")
        else:
            dev_id = self.dist_graph_set[0].rank() % self.num_gpus
            device = th.device("cuda:" + str(dev_id))
        return device

    def get_val_nid(self):
        return self.val_nid_sets[self.dataloader_idx]

    def get_test_nid(self):
        return self.test_nid_sets[self.dataloader_idx]

    def init_sampler(self, fanout_string, stop_at_border):
        self.sampler = dgl.dataloading.NeighborSampler(
            [int(fout) for fout in fanout_string.split(",")],
            stop_at_border=stop_at_border
        )

    def init_dataloader(self, batch_size, shuffle = True):
        # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.

        # a collator will be created from sampler
        # #### self.collator = NodeCollator(g, nids, graph_sampler, **collator_kwargs)
        # the work is done at self.collator
        # #### self.graph_sampler.sample_blocks(self.g, items)
        for idx, dg in enumerate(self.dist_graph_set):
            dataloader = dgl.dataloading.DistNodeDataLoader(
                dg,
                self.train_nid_sets[idx],
                self.sampler,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
            )
            self.dataloader_set.append(dataloader)
        self.dataloader = self.dataloader_set[0]

    def get_dataloader(self):
        return self.dataloader

    def init_checkpoint_config(self, resume_path, checkpoint_path, checkpoint_every):
        self.resume_path = resume_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every

    def next_epoch(self, model, opt):

        self.epoch_idx = self.epoch_idx + 1

        return self.epoch_idx < self.train_epoch_n

    def epoch_end(self, model, opt, local_train_acc):
        self.local_train_acc_window.push(local_train_acc)
        acc_s = self.local_train_acc_window.get_list()
        old_avg = np.mean(acc_s[:-1])
        cur_avg = np.mean(acc_s[1:])

        vote_pla, vote_inc, vote_dec = 0, 0, 0

        if abs(cur_avg - old_avg) < self.AVG_DELTA_THRESHOLD:
            self.control_state = ControlState.PLATEAU
            vote_pla = 1
        elif cur_avg > old_avg:
            self.control_state = ControlState.INCREASING
            vote_inc = 1
        else:
            self.control_state = ControlState.DECREASING
            vote_dec = 1

        # sync across multiple partitions
        vote = th.Tensor([vote_pla, vote_inc, vote_dec ]) # make sure the order
        th.distributed.all_reduce(vote, async_op=False)

        # gather all local control_state to decide current global_control_state
        global_state = th.argmax(vote).item()
        global_state = ControlState(global_state)
        self.control_state_window.push(global_state)

        # control_state_window is identical across all replica
        plateau_cnt, increasing_cnt, decresing_cnt = self.control_state_window.count_control_state()

        #if self.local_rank == 0:
        #    print(f"epoch: {self.epoch_idx}, global_state: {global_state}, ({plateau_cnt}, {increasing_cnt}, {decresing_cnt})")

        if global_state == ControlState.PLATEAU:
            if plateau_cnt == self.control_state_window.WINDOW_LENGTH:
                self.seal(model, opt, local_train_acc)
                self.switch()
            elif (increasing_cnt + plateau_cnt ) > decresing_cnt:
                self.keep()
            else:
                self.fall_back(model, opt)
                self.switch()

        if global_state == ControlState.INCREASING:
            self.keep()

        if global_state == ControlState.DECREASING:
            if decresing_cnt >= (increasing_cnt + plateau_cnt ):
                self.fall_back(model, opt)
                self.switch()
            else:
                self.keep()

    def keep(self):
        if self.local_rank == 0:
            print(f"epoch: {self.epoch_idx}, keep")

    def switch(self):
        if th.distributed.get_rank() == 0:
            new_idx = np.random.randint(0, len(self.dataloader_set))
            idx_tensor = th.tensor([new_idx, 0], dtype=th.int64)
        else:
            idx_tensor = th.zeros(2, dtype=th.int64)
        # broadcast need at least 8 bytes
        th.distributed.broadcast(idx_tensor, 0)
        self.dataloader_idx = idx_tensor.numpy()[0]
        self.dataloader = self.dataloader_set[self.dataloader_idx]

        # reset state
        for _ in range(CONTROL_STATE_WINDOW_LENGTH):
            self.control_state_window.push(ControlState.INCREASING)

        if self.local_rank == 0:
            print(f"epoch: {self.epoch_idx}, switch to {self.dataloader_idx}")

    def fall_back(self, model, opt):
        if len(self.sealed_ckpt) < 1:
            return
        # restore to stored ckpt
        old_epoch  =  self.epoch_idx
        ckpt_dict = self.sealed_ckpt[-1]
        self.epoch_idx = ckpt_dict['epoch']
        model.load_state_dict(ckpt_dict['model_state_dict'])
        opt.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # reset state
        for _ in range(CONTROL_STATE_WINDOW_LENGTH):
            self.control_state_window.push(ControlState.INCREASING)
        # reset acc
        for _ in range(AVERAGE_LOCAL_ACC_WINDOW_LENGTH):
            self.local_train_acc_window.push(ckpt_dict['acc'])

        if self.local_rank == 0:
            print(f"epoch: {old_epoch}, fall_back to {self.epoch_idx}")

    def seal(self, model, opt, acc):
        # add one ckpt
        ckpt_dict = {}
        ckpt_dict['epoch'] = self.epoch_idx
        ckpt_dict['model_state_dict'] = model.state_dict()
        ckpt_dict['optimizer_state_dict'] = opt.state_dict()
        ckpt_dict['acc'] = acc

        self.sealed_ckpt.append(ckpt_dict)

        if self.local_rank == 0:
            print(f"epoch: {self.epoch_idx}, seal")

    def stop(self):
        # set stop train flag
        pass
