import argparse
import socket
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.backend as FF

def pprint(loader, string):
    if loader.get_rank() == 1:
        print(string)

class IsolatedDataloader:
    def __init__(self, args) -> None:
        tic = time.time()
        if args.graph_name == "ogbpr":
            self.in_feats = 100
            self.n_classes = 47
        elif args.graph_name == "ogbar":
            self.in_feats = 128
            self.n_classes = 40
        elif args.graph_name == "ogbpa":
            self.in_feats = 128
            self.n_classes = 172
        elif args.graph_name == "cora":
            self.in_feats = 1433
            self.n_classes = 7
        elif args.graph_name == "reddit":
            self.in_feats = 602
            self.n_classes = 41
        else:
            assert False

        self.num_part = args.n_parts
        self.feat_n = [None] * self.num_part
        self.gid_lid_n = [None] * self.num_part
        self.part_src = None
        self.part_dst = None

        self.data_path = f"/data/ds_pre/{args.graph_name}/chunking_{self.num_part}_rtest"
        # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.

        self.sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(",")],
            stop_at_border=args.stop_at_border
        )

        rank = th.distributed.get_rank()
        world = th.distributed.get_world_size()
        self.rank = rank
        self.world = world
        self.nbr_idx = self.rank
        p0_u = f"{self.data_path}/p{rank}.u.bin"
        p0_v = f"{self.data_path}/p{rank}.v.bin"

        feat_0 = f"{self.data_path}/p{rank}.nfeat.bin"
        label_0 = f"{self.data_path}/p{rank}.label.bin"

        train_mask_0 = f"{self.data_path}/p{rank}.train.bin"
        valid_mask_0 = f"{self.data_path}/p{rank}.valid.bin"
        test_mask_0 = f"{self.data_path}/p{rank}.test.bin"

        # train_mask_1 = f"{data_path}/p{(rank+1)%world}.train.bin"
        # valid_mask_1 = f"{data_path}/p{(rank+1)%world}.valid.bin"
        # test_mask_1 = f"{data_path}/p{(rank+1)%world}.test.bin"

        p0_u = np.fromfile(p0_u, dtype=np.int64)
        if rank == 0:
            print(p0_u.shape)
            print(p0_u)
        p0_v = np.fromfile(p0_v, dtype=np.int64)
        if rank == 0:
            print(p0_v.shape)
            print(p0_v)

        feat_0 = np.fromfile(feat_0, dtype=np.float32)
        feat_0 = np.reshape(feat_0, (-1, self.in_feats))
        if rank == 0:
            print(feat_0.shape)
            print(feat_0)
        label_0 = np.fromfile(label_0, dtype=np.float32)
        if rank == 0:
            print(label_0.shape)
            print(label_0)
        train_mask_0 = np.fromfile(train_mask_0, dtype=np.int8)
        valid_mask_0 = np.fromfile(valid_mask_0, dtype=np.int8)
        test_mask_0 = np.fromfile(test_mask_0, dtype=np.int8)
        if rank == 0:
            print(train_mask_0.shape)
            print(train_mask_0)
        #train_mask_1 = np.fromfile(train_mask_1, dtype=np.int8)
        #valid_mask_1 = np.fromfile(valid_mask_1, dtype=np.int8)
        #test_mask_1 = np.fromfile(test_mask_1, dtype=np.int8)

        train_mask_0 = FF.zerocopy_from_numpy(train_mask_0)
        if rank == 0:
            print(train_mask_0.shape)
            print(train_mask_0)

        valid_mask_0 = FF.zerocopy_from_numpy(valid_mask_0)
        test_mask_0 = FF.zerocopy_from_numpy(test_mask_0)
        #train_mask_1 = FF.zerocopy_from_numpy(train_mask_1)
        ##valid_mask_1 = FF.zerocopy_from_numpy(valid_mask_1)
        ##test_mask_1 = FF.zerocopy_from_numpy(test_mask_1)
        #train_mask_1 = th.zeros(train_mask_1.shape[0])
        #valid_mask_1 = th.zeros(train_mask_1.shape[0])
        #test_mask_1  = th.zeros(train_mask_1.shape[0])

        self.p0_u = FF.zerocopy_from_numpy(p0_u)
        if rank == 0:
            print(self.p0_u.shape)
            print(self.p0_u)
        #exit(0)

        self.p0_v = FF.zerocopy_from_numpy(p0_v)
        self.label_0 = FF.zerocopy_from_numpy(label_0)
        self.feat_0 = FF.zerocopy_from_numpy(feat_0)

        train_nid0 = th.flatten(th.nonzero(train_mask_0))
        #train_nid1 = th.flatten(th.nonzero(train_mask_1)) + label_0.size(dim=0)
        self.train_nid = train_nid0 # th.concat((train_nid0, train_nid1))

        valid_nid0 = th.flatten(th.nonzero(valid_mask_0))
        #valid_nid1 = th.flatten(th.nonzero(valid_mask_1)) + label_0.size(dim=0)
        self.valid_nid = valid_nid0 #th.concat((valid_nid0, valid_nid1))

        test_nid0 = th.flatten(th.nonzero(test_mask_0))
        # test_nid1 = th.flatten(th.nonzero(test_mask_1)) + label_0.size(dim=0)
        self.test_nid = test_nid0 # th.concat((test_nid0, test_nid1))

        self.node_label = self.label_0

        pprint(self, f"rank: {rank}: loading pivot chunk from disk: {time.time() - tic:.2f}")
        local_batch = (train_nid0.size()[0]) // int(args.batch_size)
        print(f" rank: {rank}: batch number: {local_batch}: sample: {train_nid0.size()[0]}")
        local_batch = th.tensor([local_batch, local_batch])
        th.distributed.all_reduce(local_batch, op=th.distributed.ReduceOp.MIN)
        self.local_batch = local_batch[0].item()
        pprint(self, f" rank: {rank}: new batch number: {local_batch}")

        self.explore_mode = 1

        ############################################
        # a new mode, in this mode,
        # p0 and sum(b0i) are collected as p0',
        # p0' and p1' are the new partition
        ############################################
        # get number of nodes in each primary chunk
        # this is used to calcula
        self.explore_mode = 2 
        self.part_num_node = []
        for pid in range(self.num_part):
            # used to count node number
            train_mask_f  = f"{self.data_path}/p{pid}.train.bin"
            blob = np.memmap(train_mask_f, dtype=np.int8, mode='r')
            self.part_num_node.append(blob.size)
            print(f"xxxxxxxxx pid = {pid}")

        self.prefix_sum = [0] * len(self.part_num_node)
        for i in range(1, len(self.part_num_node)):
            self.prefix_sum[i] = ( self.part_num_node[i-1] + self.prefix_sum[i-1] )

        print("prefix_sum:")
        print(self.prefix_sum)
        #exit(0)

        self.train_nid = self.train_nid + self.prefix_sum[self.rank]

        self.manual_idx = 0 # index used for scheduling

    def load_nbr_chunk_v2(self, nbr_idx):
        tic = time.time()


        # load second primary
        p1_u = f"{self.data_path}/p{nbr_idx}.u.bin"
        p1_v = f"{self.data_path}/p{nbr_idx}.v.bin"
        p1_u = np.fromfile(p1_u, dtype=np.int64)
        p1_v = np.fromfile(p1_v, dtype=np.int64)
        # update to the global node id
        p1_u = FF.zerocopy_from_numpy(p1_u) + self.prefix_sum[nbr_idx]
        p1_v = FF.zerocopy_from_numpy(p1_v) + self.prefix_sum[nbr_idx]

        feat_1 = f"{self.data_path}/p{nbr_idx}.nfeat.bin"
        label_1 = f"{self.data_path}/p{nbr_idx}.label.bin"

        feat_1 = np.fromfile(feat_1, dtype=np.float32)
        feat_1 = np.reshape(feat_1, (-1, self.in_feats))
        label_1 = np.fromfile(label_1, dtype=np.float32)

        self.feat_1 = FF.zerocopy_from_numpy(feat_1)
        self.label_1 = FF.zerocopy_from_numpy(label_1)

        self.feat_n[self.rank] = self.feat_0
        self.feat_n[nbr_idx] = self.feat_1

        #  It does the following things
        #  0) work in a new global node id space: new_id in pi + sum_0_(i-1){num(pj)}
        #  1) it take bi_j.u or bi_j.v (local id)
        #  2) create a map from bi_j.u -> reid or -1 into bi_j.u.reid.bin
        #  3) coupled with 2), create a feature file from pj.feat.bin used by bi_j.u and rerrange into bij.u.reid.feat.bin,
        #  the workflow is batch_ids -> gids in range p -> look up id from bij_u.reid.bin -> look up feature from bij.u.reid.feat.bin

        # idx-relabel by nbr_idx

        # b01_u = f"{data_path}/b{nbr_idx}_{self.rank}.u.bin"
        # b01_v = f"{data_path}/b{nbr_idx}_{self.rank}.v.bin"
        tensor_src  = th.concat((p1_u, ))
        tensor_dest = th.concat((p1_v, ))

        for bidx in range(self.num_part):
            tensor_ids = None
            for prim_idx in [ self.rank, nbr_idx]:
                if bidx == prim_idx:
                    continue

                b01_u = f"{self.data_path}/b{prim_idx}_{bidx}.u.bin"
                b01_v = f"{self.data_path}/b{prim_idx}_{bidx}.v.bin"

                b01_u = np.fromfile(b01_u, dtype=np.int64)
                b01_v = np.fromfile(b01_v, dtype=np.int64)

                b01_u = FF.zerocopy_from_numpy(b01_u) + self.prefix_sum[prim_idx]
                b01_v = FF.zerocopy_from_numpy(b01_v) + self.prefix_sum[bidx]

                tensor_src  = th.concat((tensor_src, b01_u))
                tensor_dest  = th.concat((tensor_dest, b01_v))
                if tensor_ids is None:
                    tensor_ids = b01_v
                else:
                    tensor_ids = th.concat((tensor_ids, b01_v))

            if bidx in [ self.rank, nbr_idx]:
                pass
            else:
               uniq_ids =  th.unique(tensor_ids) # mapping 0... to global nid

               feat = f"{self.data_path}/p{bidx}.nfeat.bin"

               feat = np.fromfile(feat, dtype=np.float32)
               feat = np.reshape(feat, (-1, self.in_feats))

               feat = FF.zerocopy_from_numpy(feat)
               feat = feat[uniq_ids - self.prefix_sum[bidx]]

               self.feat_n[bidx] = feat

               gid_lid = {}
               for idx, e in enumerate(uniq_ids):
                   gid_lid[e.item()] = idx
               self.gid_lid_n[bidx] = gid_lid

        p0_u = self.p0_u + self.prefix_sum[self.rank]
        p0_v = self.p0_v + self.prefix_sum[self.rank]
        self.part_src = th.concat((tensor_src, p0_u))
        self.part_dst = th.concat((tensor_dest, p0_v))
        tensor_src = None
        tensor_dest = None

        pprint(self, f"rank: {self.rank}: loading bridge chunks from disk: {time.time() - tic:.2f}")

    def load_nbr_chunk(self, nbr_idx):
        tic = time.time()
        p1_u = f"{self.data_path}/p{nbr_idx}.u.bin"
        p1_v = f"{self.data_path}/p{nbr_idx}.v.bin"
        b01_u = f"{self.data_path}/b{self.rank}_{nbr_idx}.u.bin"
        b01_v = f"{self.data_path}/b{self.rank}_{nbr_idx}.v.bin"

        # b01_u = f"{data_path}/b{nbr_idx}_{self.rank}.u.bin"
        # b01_v = f"{data_path}/b{nbr_idx}_{self.rank}.v.bin"

        feat_1 = f"{self.data_path}/p{nbr_idx}.nfeat.bin"
        label_1 = f"{self.data_path}/p{nbr_idx}.label.bin"

        p1_u = np.fromfile(p1_u, dtype=np.int64)
        p1_v = np.fromfile(p1_v, dtype=np.int64)
        b01_u = np.fromfile(b01_u, dtype=np.int64)
        b01_v = np.fromfile(b01_v, dtype=np.int64)

        feat_1 = np.fromfile(feat_1, dtype=np.float32)
        feat_1 = np.reshape(feat_1, (-1, self.in_feats))
        label_1 = np.fromfile(label_1, dtype=np.float32)

        self.p1_u = FF.zerocopy_from_numpy(p1_u) + self.label_0.size(dim=0)
        self.p1_v = FF.zerocopy_from_numpy(p1_v) + self.label_0.size(dim=0)
        self.label_1 = FF.zerocopy_from_numpy(label_1)
        self.feat_1 = FF.zerocopy_from_numpy(feat_1)

        self.b01_u = FF.zerocopy_from_numpy(b01_u)
        self.b01_v = FF.zerocopy_from_numpy(b01_v)+ self.label_0.size(dim=0)

        pprint(self, f"rank: {self.rank}: loading bridge chunk {nbr_idx} from disk: {time.time() - tic:.2f}")

    def build_partition(self):
        tic = time.time()
        self.cur_g = dgl.graph(
            ( th.concat( (self.p0_u, self.b01_u, self.p1_u) ),
              th.concat( (self.p0_v, self.b01_v, self.p1_v) )
            )
        )
        self.cur_node_feat  = th.concat((self.feat_0, self.feat_1), 0)
        #self.cur_node_label = th.concat((self.label_0, self.label_1), 0)
        pprint(self, f"rank: {self.rank}: build partition: {time.time() - tic:.2f}")

    def build_partition_v2(self):
        tic = time.time()
        self.cur_g = dgl.graph( (self.part_src, self.part_dst) )
        #print("xxxxx: ", self.cur_g)
        #self.cur_node_label = th.concat((self.label_0, self.label_1), 0)
        pprint(self, f"rank: {self.rank}: build partition: {time.time() - tic:.2f}")

    def get_cur_dataloader(self, args, epoch):
        # for ogbpr metis 16 only
        # ogbpr
        """
        manual_sched_1 = {
            0:  [1, 2, 3],
            1:  [0],
            2:  [0, 3 ],
            3:  [0, 2, 4],
            4:  [3, 5],
            5:  [4, 7, 12, 15],
            6:  [8],
            7:  [5],
            8:  [6, 9, 11],
            9:  [8, 10, 11, 12],
            10: [9, 11, 12],
            11: [8, 9, 10, 12],
            12: [5, 9, 10, 11, 13, 14, 15],
            13: [12, 14],
            14: [12, 13, 15],
            15: [5, 12, 14],
        }

        manual_sched_2 = {
            0:  [4, 5, 6, 7, 9, 15],
            1:  [2, 5, 7, ],
            2:  [1, 4, 6, 7, 14, 15],
            3:  [6, 11, 12, 14, 15],
            4:  [0, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            5:  [0, 1, 6, 8, 9, 10, 11, 13, 14],
            6:  [0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15],
            7:  [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 14, 15],
            8:  [4, 5, 7, 10, 12, 13, 14, 15],
            9:  [0, 4, 5,  6, 7, 13, 14, 15],
            10: [4, 5, 6,  7, 8, 13, 14, 15],
            11: [3, 4, 5,  6, 7, 13, 14, 15],
            12: [3, 4, 6,  7, 8],
            13: [4, 5, 6, 8, 9, 10, 11, 15 ],
            14: [2, 3, 4, 5, 6, 7,   8, 9, 10, 11 ],
            15: [0, 2, 3, 4, 6, 7,   8, 9, 10, 11, 13],
        }
        """
        manual_sched_1 = {
            0 : [1],
            1 : [2],
            2 : [3],
            3 : [4],
            4 : [5],
            5 : [6],
            6 : [7],
            7 : [8],
            8 : [9],
            9 : [10],
            10 : [11],
            11 : [12],
            12 : [13],
            13 : [14],
            14 : [15],
            15 : [0],
        }
        """
        # reddit
        manual_sched_1 = {
            0:[1], #
            1:[2], #
            2:[3], #
            3:[4], #
            4:[5], #
            5:[6], #
            6:[7], #
            7:[8], #
            8:[9], #
            9:[10], #
            10: [11],
            11: [10],
            12:[13], #
            13:[14], #
            14: [15],
            15: [14],
        }
        
        manual_sched_2 = {
            0:  [1], #
            1:  [2], #
            2:  [4, 10],
            3:  [4], #
            4:  [2, 5, 8, 9, 10, 11],
            5:  [4, 10],
            6:  [7],
            7:  [6],
            8:  [4, 9, 10, 11],
            9:  [4, 8, 10],
            10: [2, 4, 5, 8, 9, 11],
            11: [4, 8, 10],
            12: [13], #
            13: [14], #
            14: [15],
            15: [14],
        }
        """

        if epoch % 50 == 0:
            ## the selective sched 
            #len_1   = len(manual_sched_1[self.rank])
            #len_2   = len(manual_sched_2[self.rank])
            ##len_sum = len_1 + len_2 

            #self.manual_idx = (self.manual_idx + 1) % len_1
            #self.nbr_idx = manual_sched_1[self.rank][self.manual_idx]

            #self.manual_idx = (self.manual_idx + 1) % len_sum
            #if self.manual_idx >= len_1:
            #    self.nbr_idx = manual_sched_2[self.rank][self.manual_idx - len_1]
            #else:
            #    self.nbr_idx = manual_sched_1[self.rank][self.manual_idx]

            # the simple rr sched
            self.nbr_idx = (self.nbr_idx + 1) % self.world

            if self.nbr_idx == self.rank:
                self.nbr_idx = (self.nbr_idx + 1) % self.world

            if self.explore_mode == 1:
                self.load_nbr_chunk(self.nbr_idx)
                self.build_partition()
            if self.explore_mode == 2:
                self.load_nbr_chunk_v2(self.nbr_idx)
                self.build_partition_v2()

            # self.cur_g = dgl.graph((self.p0_u, self.p0_v))
            # self.cur_node_feat  = self.feat_0

            # a collator will be created from sampler
            # #### self.collator = NodeCollator(g, nids, graph_sampler, **collator_kwargs)
            # the work is done at self.collator
            # #### self.graph_sampler.sample_blocks(self.g, items)
            self.dataloader = dgl.dataloading.DataLoader(
                self.cur_g,
                self.train_nid,
                self.sampler,
                batch_size=args.batch_size,
                shuffle=True, # False for debug
                drop_last=False,
            )
        return self.dataloader

    def get_batch_inputs(self, input_nodes):
        th.set_printoptions(profile="full")
        #if self.rank == 1:
        #    print("="*100 + "\n")
        #    print(input_nodes)
        #    print("="*100 + "\n")

        tic = time.time()

        if self.explore_mode == 1:
            return self.cur_node_feat[input_nodes].to("cpu")

        batch_inputs = th.zeros([input_nodes.size(dim=0), self.in_feats], dtype=th.float32)

        # pprint(self, f"self.prefix_sum : {self.prefix_sum}\n")

        # continous part
        for bidx in [ self.rank, self.nbr_idx]:
            cond = input_nodes >= self.prefix_sum[bidx]
            if bidx + 1 < self.num_part:
                cond = th.logical_and(cond, input_nodes < self.prefix_sum[bidx+1])
            idx = cond.nonzero()
            ids = input_nodes[idx] - self.prefix_sum[bidx]
            batch_inputs[idx] = self.feat_n[bidx][ids]
            # nbr

        #pprint(self, f"input_nodes={input_nodes}\n")
        for bidx in range(self.num_part):
            if bidx in [ self.rank, self.nbr_idx]:
                continue
            #pprint(self, f"bbbidx={bidx}, nbr_idx={self.nbr_idx}\n")
            cond = input_nodes >= self.prefix_sum[bidx]
            if bidx + 1 < self.num_part:
                cond = th.logical_and(cond, input_nodes < self.prefix_sum[bidx+1])
            #pprint(self, f"cond = {cond}")
            idx = cond.nonzero()
            #pprint(self, f"idx = {idx}")

            gid2lid = self.gid_lid_n[bidx]

            for i in idx:
                gid = input_nodes[i].item()
                if gid in gid2lid:
                    lid = gid2lid[gid]
                    batch_inputs[i] = self.feat_n[bidx][lid]
                else:
                    pprint(self, f"[{i}] gid: {gid} is not in gid2lid")
                    pprint(self, f"!"*50)
                    pprint(self, f"{gid2lid}\n")
                    pprint(self, f"!"*50)

        #pprint(self, f"rank: {self.rank}: get_batch_inputs: {time.time() - tic:.2f}")
        return batch_inputs.to("cpu")

    def get_batch_labels(self, seed):
        if self.explore_mode == 1:
            return self.node_label[seed].to("cpu")
        # otherwise
        index = seed - self.prefix_sum[self.rank]
        return self.node_label[index].to("cpu")

    def get_node_label(self):
        return self.node_label
    def get_rank(self):
        return self.rank
    def get_max_step(self):
        return self.local_batch
