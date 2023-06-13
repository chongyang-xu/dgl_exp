/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/metis_partition.cc
 * @brief Call Metis partitioning
 */

#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/parallel_for.h>

#include "../heterograph.h"
#include "../unit_graph.h"
#include "../serialize/mmap_file.h"
#include "../serialize/mmap_file.h"

#include "vc.h"

#include <climits>

#if !defined(_WIN32)
#include <GKlib.h>
#endif  // !defined(_WIN32)

using namespace dgl::runtime;

namespace dgl {

#if !defined(_WIN32)
gk_csr_t *Convert2GKCsr(const aten::CSRMatrix mat, bool is_row);
aten::CSRMatrix Convert2DGLCsr(gk_csr_t *gk_csr, bool is_row);
#endif  // !defined(_WIN32)

namespace transform {

class HaloHeteroSubgraph : public HeteroSubgraph {
 public:
  std::vector<IdArray> inner_nodes;
};

HeteroGraphPtr ReorderUnitGraph(UnitGraphPtr ug, IdArray new_order) {
  auto format = ug->GetCreatedFormats();
  // We only need to reorder one of the graph structure.
  if (format & CSC_CODE) {
    auto cscmat = ug->GetCSCMatrix(0);
    auto new_cscmat = aten::CSRReorder(cscmat, new_order, new_order);
    return UnitGraph::CreateFromCSC(
        ug->NumVertexTypes(), new_cscmat, ug->GetAllowedFormats());
  } else if (format & CSR_CODE) {
    auto csrmat = ug->GetCSRMatrix(0);
    auto new_csrmat = aten::CSRReorder(csrmat, new_order, new_order);
    return UnitGraph::CreateFromCSR(
        ug->NumVertexTypes(), new_csrmat, ug->GetAllowedFormats());
  } else {
    auto coomat = ug->GetCOOMatrix(0);
    auto new_coomat = aten::COOReorder(coomat, new_order, new_order);
    return UnitGraph::CreateFromCOO(
        ug->NumVertexTypes(), new_coomat, ug->GetAllowedFormats());
  }
}

HaloHeteroSubgraph GetSubgraphWithHalo(
    std::shared_ptr<HeteroGraph> hg, IdArray nodes, int num_hops) {
  CHECK_EQ(hg->NumBits(), 64) << "halo subgraph only supports 64bits graph";
  CHECK_EQ(hg->relation_graphs().size(), 1)
      << "halo subgraph only supports homogeneous graph";
  CHECK_EQ(nodes->dtype.bits, 64)
      << "halo subgraph only supports 64bits nodes tensor";
  const dgl_id_t *nid = static_cast<dgl_id_t *>(nodes->data);
  const auto id_len = nodes->shape[0];
  // A map contains all nodes in the subgraph.
  // The key is the old node Ids, the value indicates whether a node is a inner
  // node.
  std::unordered_map<dgl_id_t, bool> all_nodes;
  // The old Ids of all nodes. We want to preserve the order of the nodes in the
  // vector. The first few nodes are the inner nodes in the subgraph.
  std::vector<dgl_id_t> old_node_ids(nid, nid + id_len);
  std::vector<std::vector<dgl_id_t>> outer_nodes(num_hops);
  for (int64_t i = 0; i < id_len; i++) all_nodes[nid[i]] = true;
  auto orig_nodes = all_nodes;

  std::vector<dgl_id_t> edge_src, edge_dst, edge_eid;

  // When we deal with in-edges, we need to do two things:
  // * find the edges inside the partition and the edges between partitions.
  // * find the nodes outside the partition that connect the partition.
  EdgeArray in_edges = hg->InEdges(0, nodes);
  auto src = in_edges.src;
  auto dst = in_edges.dst;
  auto eid = in_edges.id;
  auto num_edges = eid->shape[0];
  const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
  const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
  const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
  for (int64_t i = 0; i < num_edges; i++) {
    // We check if the source node is in the original node.
    auto it1 = orig_nodes.find(src_data[i]);
    if (it1 != orig_nodes.end() || num_hops > 0) {
      edge_src.push_back(src_data[i]);
      edge_dst.push_back(dst_data[i]);
      edge_eid.push_back(eid_data[i]);
    }
    // We need to expand only if the node hasn't been seen before.
    auto it = all_nodes.find(src_data[i]);
    if (it == all_nodes.end() && num_hops > 0) {
      all_nodes[src_data[i]] = false;
      old_node_ids.push_back(src_data[i]);
      outer_nodes[0].push_back(src_data[i]);
    }
  }

  // Now we need to traverse the graph with the in-edges to access nodes
  // and edges more hops away.
  for (int k = 1; k < num_hops; k++) {
    const std::vector<dgl_id_t> &nodes = outer_nodes[k - 1];
    EdgeArray in_edges = hg->InEdges(0, aten::VecToIdArray(nodes));
    auto src = in_edges.src;
    auto dst = in_edges.dst;
    auto eid = in_edges.id;
    auto num_edges = eid->shape[0];
    const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
    const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
    const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
    for (int64_t i = 0; i < num_edges; i++) {
      auto it1 = orig_nodes.find(src_data[i]);
      // If the source node is in the partition, we have got this edge when we
      // iterate over the out-edges above.
      if (it1 == orig_nodes.end()) {
        edge_src.push_back(src_data[i]);
        edge_dst.push_back(dst_data[i]);
        edge_eid.push_back(eid_data[i]);
      }
      // If we haven't seen this node.
      auto it = all_nodes.find(src_data[i]);
      if (it == all_nodes.end()) {
        all_nodes[src_data[i]] = false;
        old_node_ids.push_back(src_data[i]);
        outer_nodes[k].push_back(src_data[i]);
      }
    }
  }

  if (num_hops > 0) {
    EdgeArray out_edges = hg->OutEdges(0, nodes);
    auto src = out_edges.src;
    auto dst = out_edges.dst;
    auto eid = out_edges.id;
    auto num_edges = eid->shape[0];
    const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
    const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
    const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
    for (int64_t i = 0; i < num_edges; i++) {
      // If the outer edge isn't in the partition.
      auto it1 = orig_nodes.find(dst_data[i]);
      if (it1 == orig_nodes.end()) {
        edge_src.push_back(src_data[i]);
        edge_dst.push_back(dst_data[i]);
        edge_eid.push_back(eid_data[i]);
      }
      // We don't expand along the out-edges.
      auto it = all_nodes.find(dst_data[i]);
      if (it == all_nodes.end()) {
        all_nodes[dst_data[i]] = false;
        old_node_ids.push_back(dst_data[i]);
      }
    }
  }

  // We assign new Ids to the nodes in the subgraph. We ensure that the HALO
  // nodes are behind the input nodes.
  std::unordered_map<dgl_id_t, dgl_id_t> old2new;
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    old2new[old_node_ids[i]] = i;
  }

  num_edges = edge_src.size();
  IdArray new_src = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  IdArray new_dst = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  dgl_id_t *new_src_data = static_cast<dgl_id_t *>(new_src->data);
  dgl_id_t *new_dst_data = static_cast<dgl_id_t *>(new_dst->data);
  for (size_t i = 0; i < edge_src.size(); i++) {
    new_src_data[i] = old2new[edge_src[i]];
    new_dst_data[i] = old2new[edge_dst[i]];
  }

  std::vector<int> inner_nodes(old_node_ids.size());
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    dgl_id_t old_nid = old_node_ids[i];
    inner_nodes[i] = all_nodes[old_nid];
  }
  aten::COOMatrix coo(
      old_node_ids.size(), old_node_ids.size(), new_src, new_dst);
  HeteroGraphPtr ugptr = UnitGraph::CreateFromCOO(1, coo);
  HeteroGraphPtr subg = CreateHeteroGraph(hg->meta_graph(), {ugptr});
  HaloHeteroSubgraph halo_subg;
  halo_subg.graph = subg;
  halo_subg.induced_vertices = {aten::VecToIdArray(old_node_ids)};
  halo_subg.induced_edges = {aten::VecToIdArray(edge_eid)};
  // TODO(zhengda) we need to switch to 8 bytes afterwards.
  halo_subg.inner_nodes = {aten::VecToIdArray<int>(inner_nodes, 32)};
  return halo_subg;
}

DGL_REGISTER_GLOBAL("partition._CAPI_DGLReorderGraph_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Reorder only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];
      const IdArray new_order = args[1];
      auto reorder_ugptr = ReorderUnitGraph(ugptr, new_order);
      std::vector<HeteroGraphPtr> rel_graphs = {reorder_ugptr};
      *rv = HeteroGraphRef(std::make_shared<HeteroGraph>(
          hgptr->meta_graph(), rel_graphs, hgptr->NumVerticesPerType()));
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLPartitionWithHalo_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Metis partition only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];

      IdArray node_parts = args[1];
      int num_hops = args[2];

      CHECK_EQ(node_parts->dtype.bits, 64)
          << "Only supports 64bits tensor for now";

      const int64_t *part_data = static_cast<int64_t *>(node_parts->data);
      int64_t num_nodes = node_parts->shape[0];
      std::unordered_map<int, std::vector<int64_t>> part_map;
      for (int64_t i = 0; i < num_nodes; i++) {
        dgl_id_t part_id = part_data[i];
        auto it = part_map.find(part_id);
        if (it == part_map.end()) {
          std::vector<int64_t> vec;
          vec.push_back(i);
          part_map[part_id] = vec;
        } else {
          it->second.push_back(i);
        }
      }
      std::vector<int> part_ids;
      std::vector<std::vector<int64_t>> part_nodes;
      int max_part_id = 0;
      for (auto it = part_map.begin(); it != part_map.end(); it++) {
        max_part_id = std::max(it->first, max_part_id);
        part_ids.push_back(it->first);
        part_nodes.push_back(it->second);
      }
      // When we construct subgraphs, we need to access both in-edges and
      // out-edges. We need to make sure the in-CSR and out-CSR exist.
      // Otherwise, we'll try to construct in-CSR and out-CSR in openmp for
      // loop, which will lead to some unexpected results.
      ugptr->GetInCSR();
      ugptr->GetOutCSR();
      std::vector<std::shared_ptr<HaloHeteroSubgraph>> subgs(max_part_id + 1);
      int num_partitions = part_nodes.size();
      runtime::parallel_for(0, num_partitions, [&](int b, int e) {
        for (auto i = b; i < e; i++) {
          auto nodes = aten::VecToIdArray(part_nodes[i]);
          HaloHeteroSubgraph subg = GetSubgraphWithHalo(hgptr, nodes, num_hops);
          std::shared_ptr<HaloHeteroSubgraph> subg_ptr(
              new HaloHeteroSubgraph(subg));
          int part_id = part_ids[i];
          subgs[part_id] = subg_ptr;
        }
      });
      List<HeteroSubgraphRef> ret_list;
      for (size_t i = 0; i < subgs.size(); i++) {
        ret_list.push_back(HeteroSubgraphRef(subgs[i]));
      }
      *rv = ret_list;
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLPartitionVertexCutWithHalo_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string edge_bin_file_name = args[0];
      uint64_t num_nodes = args[1];
      uint64_t num_edges = args[2];
      uint64_t num_parts = args[3];
      std::string strategy = args[4];
      uint64_t extra_hops = args[5];
      uint64_t num_train_nodes = args[6];
      std::string train_mask_file = args[7];
      uint64_t rev_edge = args[8];

      bool use_1_hop_halo =  extra_hops > 0;
      bool add_reverse_edge = rev_edge > 0 ;

      bool add_self_loop = true;

      LOG(INFO) << edge_bin_file_name << " " << num_edges << " " << num_parts << " " << strategy;

      dgl::serialize::MmapFile mf(edge_bin_file_name);
      //maybe: generalize to more type
      CHECK_EQ(mf.GetLength(), num_edges*2*sizeof(vc_vid_t)) << "file size doesn't match edge numer";

      // a SEQUENTIAL vertex cut implementation, with few optimization
      // construct each partitions edge list in original vid
      ska::flat_hash_map<uint32_t, std::vector<vc_vid_t>> pid2src;
      ska::flat_hash_map<uint32_t, std::vector<vc_vid_t>> pid2dst;
      std::vector<vc_record_t> vc_map(num_nodes, VCR_MPID_MASK);//must initialize part id as invalid
      std::vector<ska::flat_hash_set<uint32_t>> gid2rpids(num_nodes);

      vc_vid_t* src = mf.AsUint32Ptr();
      vc_vid_t* dst = src + num_edges;
      vc_vid_t s_vid, d_vid;

      if(strategy == "vcrandom"){
        for (size_t idx=0; idx < num_edges; idx++){
            s_vid = src[idx];
            d_vid = dst[idx];
            if (add_self_loop){
                if(s_vid == d_vid){
                    continue;
                }
            }
            uint32_t pid = HashEdge(s_vid, d_vid) % num_parts;
            pid2src[pid].push_back(s_vid);
            pid2dst[pid].push_back(d_vid);
            // in this sequential implementation,
            // assign main part_id when a node shows up for the first time
            if( get_mpid(vc_map[s_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[s_vid], pid);
            }else if (get_mpid(vc_map[s_vid]) != pid) {
                gid2rpids[s_vid].insert(pid);
            }
            if( get_mpid(vc_map[d_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[d_vid], pid);
            }else if (get_mpid(vc_map[d_vid]) != pid){
                gid2rpids[d_vid].insert(pid);
            }
        }
      }else if (strategy == "vcoblivious"){
        //d is degree : #machines which a vertex spans
        //diff to graphlab, here just book keeping onece for all partitions
        std::vector<std::bitset<MAX_N_PARTITION>> dht(MAX_N_NODE, 0);//2.4GB degree hash table, record machine ids each node spans
        std::vector<size_t> part_num_edges(num_parts, 0);
        std::vector<double> part_score(num_parts, 0);
        bool usehash = false, userecent = false;

        for (size_t idx=0; idx < num_edges; idx++){
            s_vid = src[idx];
            d_vid = dst[idx];
            if (add_self_loop){
                if(s_vid == d_vid)
                    continue;
            }
            uint32_t pid = AsignEdgeToPartitionGreedy(s_vid, d_vid, dht[s_vid], dht[d_vid], part_num_edges, part_score);
            pid2src[pid].push_back(s_vid);
            pid2dst[pid].push_back(d_vid);
            // in this sequential implementation,
            // assign main part_id when a node shows up for the first time
            if( get_mpid(vc_map[s_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[s_vid], pid);
            }else if (get_mpid(vc_map[s_vid]) != pid) {
                gid2rpids[s_vid].insert(pid);
            }
            if( get_mpid(vc_map[d_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[d_vid], pid);
            }else if (get_mpid(vc_map[d_vid]) != pid){
                gid2rpids[d_vid].insert(pid);
            }
        }
      }else if (strategy == "vchdrf"){
        std::vector<std::bitset<MAX_N_PARTITION>> dht(MAX_N_NODE, 0);
        std::vector<uint32_t> degree_dht(MAX_N_NODE, 0);//1.2GB, record degree number of each node
        std::vector<size_t> part_num_edges(num_parts, 0);
        std::vector<double> part_score(num_parts, 0);

        for (size_t idx=0; idx < num_edges; idx++){
            s_vid = src[idx];
            d_vid = dst[idx];
            if (add_self_loop){
                if(s_vid == d_vid)
                    continue;
            }
            uint32_t pid = AsignEdgeToPartitionHDRF(s_vid, d_vid, dht[s_vid], dht[d_vid], degree_dht[s_vid], degree_dht[d_vid], part_num_edges, part_score);
            pid2src[pid].push_back(s_vid);
            pid2dst[pid].push_back(d_vid);
            // in this sequential implementation,
            // assign main part_id when a node shows up for the first time
            if( get_mpid(vc_map[s_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[s_vid], pid);
            }else if (get_mpid(vc_map[s_vid]) != pid) {
                gid2rpids[s_vid].insert(pid);
            }
            if( get_mpid(vc_map[d_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[d_vid], pid);
            }else if (get_mpid(vc_map[d_vid]) != pid){
                gid2rpids[d_vid].insert(pid);
            }
        }
      } else if (strategy == "vcbfs"){
        const int N_BFS_SRC_NODES      = num_nodes / 1000;
        const int N_BLOCK_NEIGHBOR_HOP = 2;
        const int N_BLOCK_MAX = 100000;
        LOG(INFO) << "bfs #src_cnt:" << N_BFS_SRC_NODES;
        LOG(INFO) << "bfs #blk_max:" << N_BLOCK_MAX;
        // generate BFS source nodes
        IdArray random_source_nodes = dgl::RandomEngine::ThreadLocal()->UniformChoice<int32_t>(
              N_BFS_SRC_NODES, num_nodes, false);
        CHECK_EQ(random_source_nodes->dtype.bits, 32) << "Only supports 32bits tensor for now";
        const int32_t *src_nodes = static_cast<int32_t *>(random_source_nodes->data);
        CHECK_EQ(random_source_nodes->shape[0], N_BFS_SRC_NODES);

        std::vector<uint8_t> is_train;
        {
            dgl::serialize::MmapFile tm(train_mask_file);
            CHECK_EQ(tm.GetLength(), num_nodes) << "file size doesn't match train nodes numer";
            std::vector<uint8_t> tmp(tm.AsUint8Ptr(), tm.AsUint8Ptr() + num_nodes);
            is_train = std::move(tmp);
        }

        /////////////////////////////////////////////////////////
        ////  multi-source BFS, the graph is treated as directed
        /////////////////////////////////////////////////////////
        TIK(vcbfs);
        std::vector<vc_bid_t> gid2bid(num_nodes, VC_BID_MAX); // global node id to block id
        std::vector<bool> vst(num_nodes, false);   // count node number for each block
        std::vector<uint32_t> bid2cnt(N_BFS_SRC_NODES, 0);   // count node number for each block
        std::vector<uint32_t> bid2train_cnt(N_BFS_SRC_NODES, 0);
        ska::flat_hash_set<vc_vid_t> cur;
        ska::flat_hash_set<vc_vid_t> next;
        // initialize
        for(int i=0; i < N_BFS_SRC_NODES; i++){
            bid2cnt[i] = 1;
            vst[src_nodes[i]] = true;
            gid2bid[src_nodes[i]] = i;
            next.insert(src_nodes[i]);
            bid2train_cnt[i] = is_train[src_nodes[i]] ? 1 : 0;
        }
        // #iter: num pass of full edge list
        bool converged = false;
        uint32_t iter_cnt = 0;
        while(!converged){
            converged = true;
            iter_cnt++;
            cur = std::move(next);
            for (size_t idx = 0; idx < num_edges; idx++) {
                s_vid = src[idx];
                auto iter = cur.find(s_vid);
                if (iter == cur.end()) continue;
                d_vid = dst[idx];
                auto bid = gid2bid[s_vid];
                if (vst[d_vid] == false && bid2cnt[bid]  < N_BLOCK_MAX){
                    vst[d_vid] = true;
                    bid2cnt[bid]++;
                    converged = false;
                    gid2bid[d_vid] = bid;
                    bid2train_cnt[bid] += is_train[d_vid] ? 1:0;
                    next.insert(d_vid);
                }
            }
        }
        {
            cur.clear();
            next.clear();
            vst.clear();
        }
        // fix 1)
        // due to N_BLOCK_MAX, an edge has exactly one node visited, in this case,
        // merge unvisited nodes to its neighbor, regardless of direction
        converged = false;
        while (!converged) {
            converged = true;
            iter_cnt++;
            for(size_t idx = 0; idx < num_edges; idx++){
                s_vid = src[idx];
                d_vid = dst[idx];

                vc_vid_t old, nbr;
                if (vst[s_vid] ^ vst[d_vid]){
                    converged = false;
                    if(vst[s_vid]){
                        old = s_vid;
                        nbr = d_vid;
                    }else{
                        old = d_vid;
                        nbr = s_vid;
                    }
                    vst[nbr] = true;
                    bid2cnt[gid2bid[old]]++;
                    bid2train_cnt[gid2bid[old]] += is_train[nbr] ? 1:0;
                    gid2bid[nbr] = gid2bid[old];
                    /*
                    After this merging, it's possible that an edge has both nodes visited but assigned in different block
                    o--o|x|
                        \
                        o
                        \
                        |x|o--o
                    */
                }
            }
        }

        ska::flat_hash_set<vc_vid_t> unlabeled_node;
        std::vector<ska::flat_hash_set<vc_bid_t>> co_adj(N_BFS_SRC_NODES);
        // after handling fix 1), now fix 2) and in-place graph coarsening
        // fix 2): for edges both nodes are not visited, they are not connected by bfs_soure,
        // randomly assign unvisited edge, regardless of direction
        for(size_t idx = 0; idx < num_edges; idx++){
            s_vid = src[idx];
            d_vid = dst[idx];
            auto& u = gid2bid[s_vid]; 
            auto& v = gid2bid[d_vid];
            if( !vst[d_vid] && !vst[s_vid] /* vst and VC_BID_MAX are both needed */) {
                if (u != VC_BID_MAX && v == VC_BID_MAX){ // VC_BID_MAX is used as flag for visit
                    v = u;
                    bid2cnt[u]++;
                    bid2train_cnt[u] += is_train[d_vid] ? 1:0;
                } else if (u == VC_BID_MAX && v != VC_BID_MAX){
                    u = v;
                    bid2cnt[v]++;
                    bid2train_cnt[v] += is_train[s_vid] ? 1:0;
                } else if (gid2bid[s_vid] == VC_BID_MAX && gid2bid[d_vid] == VC_BID_MAX){
                    auto m = HashEdge(s_vid, d_vid) % N_BFS_SRC_NODES;
                    u = v = m;
                    bid2cnt[m]+=2;
                    bid2train_cnt[m] += is_train[s_vid] ? 1:0;
                    bid2train_cnt[m] += is_train[d_vid] ? 1:0;
                } else { }
                // debugging info
                unlabeled_node.insert(s_vid);
                unlabeled_node.insert(d_vid);
            } else if ( vst[d_vid] ^ vst[s_vid] ){
                LOG(FATAL) << "should not reach here";
            }
            ////////////////////////
            // in-place coarsening
            ////////////////////////
            if(u != v){
                co_adj[u].insert(v); //remove duplicated edge
            }
        }
        //debug info
        uint32_t co_edge_n = 0;
        for(uint64_t i=0; i < co_adj.size(); i++)
            co_edge_n += co_adj[i].size();
        // merging small blk to bigger
        // pass
        LOG(INFO) << "bfs takes   :" << iter_cnt+1 << " iter(s) to converge";
        LOG(INFO) << "bfs #unlabel:" << unlabeled_node.size() << " (" << num_nodes << ") randomly assigned";
        LOG(INFO) << "bfs #co_edge:" << co_edge_n << " (deg_avg = " << co_edge_n/N_BFS_SRC_NODES << ")";
        TOK(vcbfs);

        ///////////////////////////////////////
        ////   assign block to partition
        ///////////////////////////////////////
        TIK(assign_blk);
        std::vector<double> pid2cnt(num_parts, 0.0); 
        std::vector<double> pid2train_cnt(num_parts, 0.0); 
        std::vector<uint32_t> bid2pid(N_BFS_SRC_NODES);
        double part_c  = (num_nodes + num_parts ) / num_parts; //  partition nodes capacity
        double part_ct = (num_train_nodes + num_parts ) / num_parts; // partition train nodes capacity

        std::vector<double> pid2score(num_parts, 0.0); 
        std::vector<ska::flat_hash_set<vc_bid_t>> pid2bids(num_parts);

        if (N_BLOCK_NEIGHBOR_HOP != 2) LOG(FATAL) << "only support N_BLOCK_NEIGHBOR_HOP = 2";
        for (size_t bidx=0; bidx < co_adj.size(); bidx++){
            double max_score = 0.0;    
            uint32_t pid = bidx % num_parts;
            for(uint64_t pidx = 0; pidx < num_parts; pidx++){
                uint32_t k = 0;
                {// find num of intersect
                    auto p_i = pid2bids[pidx];
                    ska::flat_hash_set<vc_bid_t> vst;
                    for(auto t: co_adj[bidx]){
                        if(vst.find(t) == vst.end()){
                            vst.insert(t);
                            k += p_i.find(t) != p_i.end() ? 1 : 0; 
                        }
                        for(auto q: co_adj[t]){
                            if(vst.find(q) == vst.end()){
                                vst.insert(q);
                                k += p_i.find(q) != p_i.end() ? 1 : 0; 
                            }
                        }
                    }
                }
                double score = k * ( 1 - pid2train_cnt[pidx]/part_ct) * ( 1 - pid2cnt[pidx]/part_c);
                if (score > max_score){
                    max_score = score;
                    pid = pidx;
                }
            }
            pid2cnt[pid] += bid2cnt[bidx];
            pid2train_cnt[pid] += bid2train_cnt[bidx];
            pid2bids[pid].insert(bidx);
            bid2pid[bidx] = pid;
            // LOG(INFO) << "bid=" << bidx << " => " << "pid=" << pid;
        }
        TOK(assign_blk);
        ////////////////////////////////
        // uncoarsening
        ////////////////////////////////
        TIK(uncoarsening);
        for (size_t idx=0; idx < num_edges; idx++){
            s_vid = src[idx];
            d_vid = dst[idx];
            if (add_self_loop){
                if(s_vid == d_vid){
                    continue;
                }
            }
            // 1 node => 1 block => 1 partition
            auto a = gid2bid[s_vid];
            auto b = gid2bid[d_vid];
            uint32_t pid_u, pid_v;
            if (a == b){
                pid_u = pid_v = bid2pid[a];
            }else{
                pid_u = bid2pid[a];
                pid_v = bid2pid[b];
            }

            if( get_mpid(vc_map[s_vid]) == VCR_MPID_MASK){
               set_mpid(vc_map[s_vid], pid_u);
            }
            if( get_mpid(vc_map[d_vid]) == VCR_MPID_MASK){
                set_mpid(vc_map[d_vid], pid_v);
            }

            if (pid_u == pid_v ){
                pid2src[pid_u].push_back(s_vid);
                pid2dst[pid_u].push_back(d_vid);
            }else{
                // if this edge is cutted, drop it
                // if use_1_hop_halo, replicate edge to destination nodes
                if(use_1_hop_halo){
                    pid2src[pid_u].push_back(s_vid);
                    pid2dst[pid_u].push_back(d_vid);
                    //pid2src[pid_v].push_back(s_vid);
                    //pid2dst[pid_v].push_back(d_vid);
                }
            }
        }
        TOK(uncoarsening);
        use_1_hop_halo = false;
        LOG(INFO) << "vcbs: use_1_hop_halo === "<< use_1_hop_halo << " when constructing subgraph";
      } else if (strategy == "randwalk"){
        LOG(FATAL) << "not supported edge assign strategy: "<< strategy;
      } else {
        LOG(FATAL) << "not supported edge assign strategy: "<< strategy;
      }

    // Construct DGL's subgraphs
    List<HeteroSubgraphRef> ret_list;
    std::vector<std::shared_ptr<HeteroSubgraph>> subgs(num_parts);
    if (strategy == "vcrandom" || strategy == "vcoblivious" || strategy == "vchdrf"){
        ConstructVCSubGraph(pid2src, pid2dst, vc_map, gid2rpids, subgs, num_parts, use_1_hop_halo, add_self_loop, add_reverse_edge);
    }else if (strategy == "vcbfs"){
        ConstructVCSubGraph(pid2src, pid2dst, vc_map, gid2rpids, subgs, num_parts, use_1_hop_halo, add_self_loop, add_reverse_edge);
    }else if (strategy == "randwalk"){
        LOG(FATAL) << "not supported edge assign strategy: "<< strategy;
    }else{
        LOG(FATAL) << "not supported edge assign strategy: "<< strategy;
    }

    for (size_t i = 0; i < subgs.size(); i++) {
        ret_list.push_back(HeteroSubgraphRef(subgs[i]));
    }
    *rv = ret_list;
});

template <class IdType>
struct EdgeProperty {
  IdType eid;
  int64_t idx;
  int part_id;
};

// Reassign edge IDs so that all edges in a partition have contiguous edge IDs.
// The original edge IDs are returned.
DGL_REGISTER_GLOBAL("partition._CAPI_DGLReassignEdges_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Reorder only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];
      IdArray etype = args[1];
      IdArray part_id = args[2];
      bool is_incsr = args[3];
      auto csrmat = is_incsr ? ugptr->GetCSCMatrix(0) : ugptr->GetCSRMatrix(0);
      int64_t num_edges = csrmat.data->shape[0];
      int64_t num_rows = csrmat.indptr->shape[0] - 1;
      IdArray new_data =
          IdArray::Empty({num_edges}, csrmat.data->dtype, csrmat.data->ctx);
      // Return the original edge Ids.
      *rv = new_data;

      // Generate new edge Ids.
      ATEN_ID_TYPE_SWITCH(new_data->dtype, IdType, {
        CHECK(etype->dtype.bits == sizeof(IdType) * 8);
        CHECK(part_id->dtype.bits == sizeof(IdType) * 8);
        const IdType *part_id_data = static_cast<IdType *>(part_id->data);
        const IdType *etype_data = static_cast<IdType *>(etype->data);
        const IdType *indptr_data = static_cast<IdType *>(csrmat.indptr->data);
        IdType *typed_data = static_cast<IdType *>(csrmat.data->data);
        IdType *typed_new_data = static_cast<IdType *>(new_data->data);
        std::vector<EdgeProperty<IdType>> indexed_eids(num_edges);
        for (int64_t i = 0; i < num_rows; i++) {
          for (int64_t j = indptr_data[i]; j < indptr_data[i + 1]; j++) {
            indexed_eids[j].eid = typed_data[j];
            indexed_eids[j].idx = j;
            indexed_eids[j].part_id = part_id_data[i];
          }
        }
        auto comp = [etype_data](
                        const EdgeProperty<IdType> &a,
                        const EdgeProperty<IdType> &b) {
          if (a.part_id == b.part_id) {
            return etype_data[a.eid] < etype_data[b.eid];
          } else {
            return a.part_id < b.part_id;
          }
        };
        // We only need to sort the edges if the input graph has multiple
        // relations. If it's a homogeneous grap, we'll just assign edge Ids
        // based on its previous order.
        if (etype->shape[0] > 0) {
          std::sort(indexed_eids.begin(), indexed_eids.end(), comp);
        }
        for (int64_t new_eid = 0; new_eid < num_edges; new_eid++) {
          int64_t orig_idx = indexed_eids[new_eid].idx;
          typed_new_data[new_eid] = typed_data[orig_idx];
          typed_data[orig_idx] = new_eid;
        }
      });
      ugptr->InvalidateCSR();
      ugptr->InvalidateCOO();
    });

DGL_REGISTER_GLOBAL("partition._CAPI_GetHaloSubgraphInnerNodes_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroSubgraphRef g = args[0];
      auto gptr = std::dynamic_pointer_cast<HaloHeteroSubgraph>(g.sptr());
      CHECK(gptr) << "The input graph has to be HaloHeteroSubgraph";
      *rv = gptr->inner_nodes[0];
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLMakeSymmetric_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Metis partition only supports homogeneous graph";
      auto ugptr = hgptr->relation_graphs()[0];

#if !defined(_WIN32)
      // TODO(zhengda) should we get whatever CSR exists in the graph.
      gk_csr_t *gk_csr = Convert2GKCsr(ugptr->GetCSCMatrix(0), true);
      gk_csr_t *sym_gk_csr = gk_csr_MakeSymmetric(gk_csr, GK_CSR_SYM_SUM);
      auto mat = Convert2DGLCsr(sym_gk_csr, true);
      gk_csr_Free(&gk_csr);
      gk_csr_Free(&sym_gk_csr);

      auto new_ugptr = UnitGraph::CreateFromCSC(
          ugptr->NumVertexTypes(), mat, ugptr->GetAllowedFormats());
      std::vector<HeteroGraphPtr> rel_graphs = {new_ugptr};
      *rv = HeteroGraphRef(std::make_shared<HeteroGraph>(
          hgptr->meta_graph(), rel_graphs, hgptr->NumVerticesPerType()));
#else
      LOG(FATAL) << "The fast version of making symmetric graph is not "
                    "supported in Windows.";
#endif  // !defined(_WIN32)
    });

}  // namespace transform
}  // namespace dgl
