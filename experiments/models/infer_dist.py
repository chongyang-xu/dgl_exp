import numpy as np

import dgl
from dgl.distributed.graph_partition_book import VCMapPartitionBook
import torch as th
import tqdm


def inference_for_default_sampling(model, g, x, batch_size, device, grouping_hack=False):
    """
    Inference with the GraphSAGE model on full neighbors (i.e. without
    neighbor sampling).

    g : the entire graph.
    x : the input of entire node set.

    Distributed layer-wise inference.
    """
    # During inference with sampling, multi-layer blocks are very
    # inefficient because lots of computations in the first few layers
    # are repeated. Therefore, we compute the representation of all nodes
    # layer by layer.  The nodes on each layer are of course splitted in
    # batches.
    # TODO: can we standardize this?
    stop_at_border = False
    force_even_flg = False #if stop_at_border else True
    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()),
        g.get_partition_book(),
        force_even=force_even_flg,
    )
    if model.name == 'gat':
        infer_hidden_dim = model.n_hidden * model.n_heads
    else:
        infer_hidden_dim = model.n_hidden
 
    policy = None
    if grouping_hack:
        policy = g.get_node_partition_policy('_N~infer') # node~_N~infer
    y = dgl.distributed.DistTensor(
        (g.num_nodes(), infer_hidden_dim),
        th.float32,
        "h",
        persistent=True,
        part_policy=policy,
    )
    for i, layer in enumerate(model.layers):
        if i == len(model.layers) - 1:
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), model.n_classes),
                th.float32,
                "h_last",
                persistent=True,
                part_policy=policy,
            )
        print(
                f"g.rank()={g.rank()}, |V|={g.num_nodes()}, eval batch size: {batch_size}"
        )

        sampler = dgl.dataloading.NeighborSampler(
            [-1],
            stop_at_border=stop_at_border
        )
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        if model.name == 'gat':
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = h.flatten(1)
                    h = model.activation(h)
                    h = model.dropout(h)
                else:
                    h = h.mean(1)

                y[output_nodes] = h.cpu()
                h = None
                h_dst = None
        else:
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = model.activation(h)
                    h = model.dropout(h)

                y[output_nodes] = h.cpu()
                h = None


        x = y
        g.barrier()
    x = None
    return y

def inference_for_stop_at_the_border(model, g, x, batch_size, device):
    stop_at_border = True
    force_even_flg = False if stop_at_border else True

    pb = g.get_partition_book()
    is_vcmap_pb = isinstance(pb, VCMapPartitionBook)

    num_node_this_part = pb.get_part_size_node(pb.partid, type_name='_N')
    
    if model.name == 'gat':
        infer_hidden_dim = model.n_hidden * model.n_heads
    else:
        infer_hidden_dim = model.n_hidden

    if is_vcmap_pb:
        nodes = th.arange(start=0, end=num_node_this_part, dtype=th.int64)
        y = th.zeros((num_node_this_part, infer_hidden_dim), dtype=th.float32)
    else: # RangePartitionBook
        offset = 0
        for i in range(pb.partid):
            offset = offset + pb.get_part_size_node(i, type_name='_N')
        nodes = th.arange(start=offset, end=offset+num_node_this_part, dtype=th.int64)
        y = th.zeros((pb._num_nodes(), infer_hidden_dim), dtype=th.float32)

    for i, layer in enumerate(model.layers):
        if i == len(model.layers) - 1:
            if is_vcmap_pb:
                y = th.zeros((num_node_this_part, model.n_classes), dtype=th.float32)
            else: # RangePartitionBook
                y = th.zeros((pb._num_nodes(), model.n_classes), dtype=th.float32)
        print(
                f"g.rank()={g.rank()}, |V|={num_node_this_part}, eval batch size: {batch_size}"
        )

        sampler = dgl.dataloading.NeighborSampler(
            [-1],
            stop_at_border=stop_at_border
        )

        dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        if model.name == 'gat':
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = h.flatten(1)
                    h = model.activation(h)
                    h = model.dropout(h)
                else:
                    h = h.mean(1)
                y[output_nodes] = h.cpu()
                h = None
                h_dst = None

        else:
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = model.activation(h)
                    h = model.dropout(h)

                y[output_nodes] = h.cpu()
                h = None
                h_dst = None

        x = y
        g.barrier()
    x = None
    return y

# x feature of g
def inference(model, g, x, batch_size, device, stop_at_border=False, grouping_hack=False):
    if stop_at_border == False:
        return inference_for_default_sampling(model, g, x, batch_size, device, grouping_hack)
    else:
        return inference_for_stop_at_the_border(model, g, x, batch_size, device)
