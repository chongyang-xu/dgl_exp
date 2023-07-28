/**
 * Copyright (c) 2023
 * @file graph/transform/vc.h
 * @brief vertex cut
 */

#ifndef DGL_GRAPH_TRANSFORM_VC_H_
#define DGL_GRAPH_TRANSFORM_VC_H_

#define MAX_N_PARTITION 64
#define MAX_N_NODE 300000000

#include <bitset>
#include <cassert>
#include <ctime>  // for clock_gettime

#include "flat_hash_map.hpp"


inline int cur_ns(uint64_t* ns) {
	  struct timespec cur;
	    int ret = clock_gettime(CLOCK_REALTIME, &cur);
	      *ns = cur.tv_sec * 1000000000 + cur.tv_nsec;
	        return ret;
}

#define TIK(n)                  \
	  uint64_t n##start, n##end;    \
	    int n##start_ret, n##end_ret; \
	      double __FILE__##n##_us;      \
	        n##start_ret = cur_ns(&n##start)

#define TOK(n)                                              \
	  n##end_ret = cur_ns(&n##end);                             \
	    if (n##start_ret != 0 || n##end_ret != 0) {               \
		        std::cout << "clock_gettime error" << std::endl;        \
		        throw 0;                                                \
		      }                                                         \
		        __FILE__##n##_us = (n##end - n##start) / 1000000.0;          \
			  LOG(INFO) << "[Timer][" << #n << "][" \
			              << __FILE__##n##_us << "][ms]"

namespace dgl {
namespace transform {

// NOTE(ds4gnn)
using vc_record_t=uint64_t;
using vc_vid_t=uint32_t;
using vc_bid_t=uint16_t;

#define VC_BID_MAX USHRT_MAX
#define VCR_MPID_MASK (0xFFFF)
inline vc_record_t set_mpid (vc_record_t& r, uint32_t pid){
    r &= ~VCR_MPID_MASK;
    return  r |= (pid & VCR_MPID_MASK);
}
//use uint64_t to store lid, to avoid overflow
inline vc_record_t set_lid (vc_record_t& r, uint64_t lid){
    return  r |= (lid << 16);
}

inline vc_record_t get_mpid(vc_record_t r){
    return (r & VCR_MPID_MASK);
}

inline vc_record_t get_lid (vc_record_t r){
    return r >> 16;
}

//////////////////////////////////////////////////////////////////////////
//  ported from graphlab: Start
//////////////////////////////////////////////////////////////////////////

// Jenkin's 32 bit integer mix from
// http://burtleburtle.net/bob/hash/integer.html
inline uint32_t integer_mix(uint32_t a) {
  a -= (a << 6);
  a ^= (a >> 17);
  a -= (a << 9);
  a ^= (a << 4);
  a -= (a << 3);
  a ^= (a << 10);
  a ^= (a >> 15);
  return a;
}

inline static size_t HashEdge(
    const uint32_t one, const uint32_t another, const uint32_t seed = 5) {
  // a bunch of random numbers
#if (__SIZEOF_PTRDIFF_T__ == 8)
  static const size_t a[8] = {0x6306AA9DFC13C8E7, 0xA8CD7FBCA2A9FFD4,
                              0x40D341EB597ECDDC, 0x99CFA1168AF8DA7E,
                              0x7C55BCC3AF531D42, 0x1BC49DB0842A21DD,
                              0x2181F03B1DEE299F, 0xD524D92CBFEC63E9};
#else
  static const size_t a[8] = {0xFC13C8E7, 0xA2A9FFD4, 0x597ECDDC, 0x8AF8DA7E,
                              0xAF531D42, 0x842A21DD, 0x1DEE299F, 0xBFEC63E9};
#endif
  // always put smaller id on left
  if (one < another)
    return (integer_mix(one ^ a[seed % 8])) ^
           (integer_mix(another ^ a[(seed + 1) % 8]));
  else
    return (integer_mix(another ^ a[seed % 8])) ^
           (integer_mix(one ^ a[(seed + 1) % 8]));
}

inline static uint32_t AsignEdgeToPartitionGreedy(
    const uint32_t one, const uint32_t another,
    std::bitset<MAX_N_PARTITION>& sd, std::bitset<MAX_N_PARTITION>& dd,
    std::vector<size_t>& part_num_edges, std::vector<double>& part_score,
    const bool usehash = false, const bool userecent = false) {
  // Compute the score of each proc.
  const size_t kPartNum = part_num_edges.size();

  uint32_t best_pid = -1;
  double maxscore = 0.0;
  double epsilon = 1.0;

  auto res = std::minmax_element(part_num_edges.begin(), part_num_edges.end());
  size_t minedges = *res.first;
  size_t maxedges = *res.second;

  for (size_t i = 0; i < kPartNum; ++i) {
    size_t sd_v = sd.test(i) + (usehash && (one % kPartNum == i));
    size_t td_v = dd.test(i) + (usehash && (another % kPartNum == i));
    double bal =
        (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
    part_score[i] = bal + ((sd_v > 0) + (td_v > 0));
  }
  maxscore = *std::max_element(part_score.begin(), part_score.end());

  std::vector<uint32_t> top_procs;
  for (size_t i = 0; i < kPartNum; ++i)
    if (std::fabs(part_score[i] - maxscore) < 1e-5) top_procs.push_back(i);

  // Hash the edge to one of the best procs.
  best_pid = top_procs[HashEdge(one, another) % top_procs.size()];

  assert(best_pid < kPartNum);
  if (userecent) {
    sd.reset();
    dd.reset();
  }
  sd.set(best_pid);
  dd.set(best_pid);
  ++part_num_edges[best_pid];
  return best_pid;
}

inline static uint32_t AsignEdgeToPartitionGreedyDegree(
    const uint32_t one, const uint32_t another,
    const uint32_t deg_one, const uint32_t deg_another,
    const uint32_t deg_low, const uint32_t deg_high,
    std::bitset<MAX_N_PARTITION>& sd, std::bitset<MAX_N_PARTITION>& dd,
    std::vector<size_t>& part_num_edges) {
  // Compute the score of each proc.
  const size_t kPartNum = part_num_edges.size();

  uint32_t best_pid = -1;
  double maxscore = 0.0;
  double epsilon = 1.0;

  auto res = std::minmax_element(part_num_edges.begin(), part_num_edges.end());
  size_t minedges = *res.first;
  size_t maxedges = *res.second;

  std::vector<double> part_score(kPartNum, 0.0);
  for (size_t i = 0; i < kPartNum; ++i) {
 
    part_score[i] = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);

    if(sd.test(i)){
    	part_score[i] += (deg_one < deg_low) ? 2.0 : (deg_one > deg_high) ? 0.0 : 0.4;
    }
    if(dd.test(i)){
    	part_score[i] += (deg_another < deg_low) ? 2.0 : (deg_one > deg_high) ? 0.0 : 0.4;
    }

  }
  maxscore = *std::max_element(part_score.begin(), part_score.end());

  std::vector<uint32_t> top_procs;
  for (size_t i = 0; i < kPartNum; ++i)
    if (std::fabs(part_score[i] - maxscore) < 1e-5) top_procs.push_back(i);

  // Hash the edge to one of the best procs.
  best_pid = top_procs[HashEdge(one, another) % top_procs.size()];

  assert(best_pid < kPartNum);
  sd.set(best_pid);
  dd.set(best_pid);
  ++part_num_edges[best_pid];
  return best_pid;
}

/*
 *  author : Fabio Petroni [www.fabiopetroni.com]
 *           Giorgio Iacoboni [g.iacoboni@gmail.com]
 *
 *  Based on the publication:
 *  F. Petroni, L. Querzoni, K. Daudjee, S. Kamali and G. Iacoboni:
 *  "HDRF: Stream-Based Partitioning for Power-Law Graphs".
 *  CIKM, 2015.
 * */
inline static uint32_t AsignEdgeToPartitionHDRF(
    const uint32_t one, const uint32_t another,
    std::bitset<MAX_N_PARTITION>& sd, std::bitset<MAX_N_PARTITION>& dd,
    uint32_t& src_true_degree, uint32_t& dst_true_degree,
    std::vector<size_t>& part_num_edges, std::vector<double>& part_score,
    const bool usehash = false, const bool userecent = false) {
  // Compute the score of each proc.
  const size_t kPartNum = part_num_edges.size();

  size_t deg_u = src_true_degree;
  deg_u = deg_u +1;
  size_t deg_v = dst_true_degree;
  deg_v = deg_v +1;
  size_t SUM = deg_u + deg_v;
  double fu = deg_u;
  fu /= SUM;
  double fv = deg_v;
  fv /= SUM;

  uint32_t best_pid = -1;
  double maxscore = 0.0;
  double epsilon = 1.0;

  auto res = std::minmax_element(part_num_edges.begin(), part_num_edges.end());
  size_t minedges = *res.first;
  size_t maxedges = *res.second;

  double new_sd = 0;
  double new_td = 0;
  for (size_t i = 0; i < kPartNum; ++i) {
    size_t sd_v = sd.test(i) + (usehash && (one % kPartNum == i));
    size_t td_v = dd.test(i) + (usehash && (another % kPartNum == i));
    new_sd = (sd_v > 0) ? 1 + (1 - fu) : 0;
    new_td = (td_v > 0) ? 1 + (1 - fv) : 0;
    double bal =
        (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
    part_score[i] = bal + new_sd + new_td;
  }

  maxscore = *std::max_element(part_score.begin(), part_score.end());

  std::vector<uint32_t> top_procs;
  for (size_t i = 0; i < kPartNum; ++i)
    if (std::fabs(part_score[i] - maxscore) < 1e-5) top_procs.push_back(i);

  // Hash the edge to one of the best procs.
  best_pid = top_procs[HashEdge(one, another) % top_procs.size()];

  assert(best_pid < kPartNum);
  if (userecent) {
    sd.reset();
    dd.reset();
  }
  sd.set(best_pid);
  dd.set(best_pid);
  ++part_num_edges[best_pid];
  ++src_true_degree;
  ++dst_true_degree;
  return best_pid;
}

//////////////////////////////////////////////////////////////////////////
//  ported from graphlab: End
//////////////////////////////////////////////////////////////////////////

void ConstructVCSubGraph(ska::flat_hash_map<uint32_t, std::vector<vc_vid_t>>& pid2src,
                         ska::flat_hash_map<uint32_t, std::vector<vc_vid_t>>& pid2dst,
                         std::vector<vc_record_t>& vc_map,
                         std::vector<ska::flat_hash_set<uint32_t>>& gid2rpids,
                         std::vector<std::shared_ptr<HeteroSubgraph>>& subgs,
                         uint64_t num_parts,
                         bool use_1_hop_halo,
                         bool add_self_loop,
                         bool add_reverse_edge){
    char* dbg_str;
    uint32_t vc_debug_level = 0;
    if( (dbg_str = getenv("VC_DEBUG_LEVEL")) ){
        if(std::string(dbg_str) == "1"){
            vc_debug_level = 1;
        }else if(std::string(dbg_str) == "2"){
            vc_debug_level = 2;
        }
    }
    if (vc_debug_level == 2) {
        // count #replica of each node
        std::unordered_map<uint64_t, uint64_t> node_freq_cnt;
        for(uint64_t i=0; i <= num_parts; i++){
            node_freq_cnt[i] = 0;
        }
        for(uint64_t i=0; i < gid2rpids.size(); i++){
            uint64_t freq = gid2rpids[i].size() + 1;
            node_freq_cnt[freq]++;
        }
        LOG(INFO) << "node #replica frequency histogram";
        LOG(INFO) << "count \t freq";
        for(auto p : node_freq_cnt){
            LOG(INFO) << p.first <<" \t" << p.second;
        }

        // count union( #replica_u, #replica_v) of each edge
        std::unordered_map<uint64_t, uint64_t> edge_freq_cnt_u_v;
        std::unordered_map<uint64_t, uint64_t> edge_freq_cnt_u;
        std::unordered_map<uint64_t, uint64_t> edge_freq_cnt_v;

        for(uint64_t i=0; i <= num_parts; i++){
            edge_freq_cnt_u_v[i] = 0;
            edge_freq_cnt_u[i] = 0;
            edge_freq_cnt_v[i] = 0;
        }
        for(uint64_t p=0; p < num_parts; p++){
            auto& src = pid2src[p];
            auto& dst = pid2dst[p];
            ska::flat_hash_set<uint64_t> union_pids;

            for(uint64_t i=0; i < src.size(); i++){
                auto u = src[i];
                auto v = dst[i];
                union_pids.clear();
                union_pids.insert(get_mpid(vc_map[u]));
                union_pids.insert(get_mpid(vc_map[v]));
                for(auto pid : gid2rpids[u])    union_pids.insert(pid);
                for(auto pid : gid2rpids[v])    union_pids.insert(pid);
                edge_freq_cnt_u_v[union_pids.size()]++;
                edge_freq_cnt_u[gid2rpids[u].size()+1]++;
                edge_freq_cnt_v[gid2rpids[v].size()+1]++;
            }
        }
        LOG(INFO) << "edge #union(replica_u, replica_v) frequency histogram";
        LOG(INFO) << "count\tfreq";
        for(auto p : edge_freq_cnt_u_v){
            LOG(INFO) << p.first <<"\t" << p.second;
        }
        LOG(INFO) << "#edge replica_u frequency histogram";
        LOG(INFO) << "count\tfreq";
        for(auto p : edge_freq_cnt_u){
            LOG(INFO) << p.first <<"\t" << p.second;
        }
        LOG(INFO) << "#edge replica_v frequency histogram";
        LOG(INFO) << "count\tfreq";
        for(auto p : edge_freq_cnt_v){
            LOG(INFO) << p.first <<"\t" << p.second;
        }
    }
    bool add_self_loop_only_main = true;
    LOG(INFO) << "add_self_loop_only_main: " << add_self_loop_only_main;

    // a vector for all partitions, used for extend 1 hop neighbors
    //    in each partition: store for each remote partition, their 1_hop neighbor edges in this parition
    //     store src and dst seperately
    std::vector<std::vector<std::vector<vc_vid_t>>> remote_1_hop_edges_u;
    std::vector<std::vector<std::vector<vc_vid_t>>> remote_1_hop_edges_v;
    for(uint64_t i = 0; i < num_parts; i++){
        std::vector<std::vector<vc_vid_t>> t;
        t.resize(num_parts);
        remote_1_hop_edges_u.push_back(t);
        remote_1_hop_edges_v.push_back(t);
    }

    runtime::parallel_for(0, num_parts, [&](uint64_t b, uint64_t e) {
       for (auto l_pid = b; l_pid < e; l_pid++) {
            auto& src_nodes = pid2src[l_pid];
            auto& dst_nodes = pid2dst[l_pid];

            {
                if (vc_debug_level == 1 &&  l_pid == 0)
                for(uint64_t i=0; i < num_parts; i++){
                    int max_row = pid2src[i].size() > 10 ? 10: pid2src[i].size();
                    LOG(INFO) << "GID: Partition " << i;
                    for(int j = 0; j < max_row; j++){
                        LOG(INFO) << "(" << pid2src[i][j] << ", " << pid2dst[i][j] << ")";
                    }
                }
            }
            TIK(construct_vc_relabel);
#pragma omp barrier
            std::bitset<MAX_N_PARTITION> replicate2pids;
            for(uint64_t i=0; use_1_hop_halo && i < num_parts; i++) {
                remote_1_hop_edges_u[l_pid][i].reserve(src_nodes.size()/num_parts/8);
                remote_1_hop_edges_v[l_pid][i].reserve(src_nodes.size()/num_parts/8);
            }

            std::vector<vc_vid_t> induced_nodes;
            induced_nodes.reserve(vc_map.size()/num_parts);
            //relabling
            ska::flat_hash_map<vc_vid_t, vc_vid_t> seen_vid_to_nid; //to new id in current partition
            {
                for(uint64_t idx=0; idx < src_nodes.size(); idx++){
                    vc_vid_t u = src_nodes[idx];
                    vc_vid_t v = dst_nodes[idx];

                    auto it = seen_vid_to_nid.find(u);
                    if (it == seen_vid_to_nid.end()){
                        induced_nodes.push_back(u);
                        seen_vid_to_nid[u] = induced_nodes.size()-1;

                        // if u's main part-id is current partition, seen_vid_to_nid[u] is valid for vc_map[u].l_pid
                        // if u's main part-id is not current partition, other thread shall update vc_map[u]
                        // lock should NOT be necessary here
                        // vc_map[u] only exposes main replica of a cutted vertex
                        if (l_pid == get_mpid(vc_map[u])){
                            set_lid(vc_map[u], induced_nodes.size()-1);
                        }
                        // regarding to sampling (assume no halo hops)
                        // 1) stop-at-the-border sampling:
                        //      get local ids of seeds: lid = vc_map[seeds_gid].lid
                        //      sampling via local_id (mirror nodes are visible)
                        //      gid can be get from gid=induced_nodes[local_id],ndata[NID][local_id],ndata['node_feat'][local_id]
                        // 2) sampling across partition: not supported
                        //      edge-cut has inner nodes and outer nodes; vertex cut all nodes are inner
                        src_nodes[idx] = induced_nodes.size()-1; // update local new id in place
                    }else{
                        src_nodes[idx] = it->second;
                    }

                    it = seen_vid_to_nid.find(v);
                    if (it == seen_vid_to_nid.end()){
                        induced_nodes.push_back(v);
                        seen_vid_to_nid[v] = induced_nodes.size()-1;

                        if (l_pid == get_mpid(vc_map[v])){
                            set_lid(vc_map[v], induced_nodes.size()-1);
                        }

                        dst_nodes[idx] = induced_nodes.size()-1; // update local new id in place
                    }else{
                        dst_nodes[idx] = it->second;
                    }

                    //////////////////////////////////////////
                    // store 1 extra hop: BEGIN
                    //////////////////////////////////////////
                    if (use_1_hop_halo) {
                        replicate2pids.reset();
                        replicate2pids.set(get_mpid(vc_map[u]));
                        replicate2pids.set(get_mpid(vc_map[v]));
                        for(auto rpid : gid2rpids[u])
                            replicate2pids.set(rpid);
                        for(auto rpid : gid2rpids[v])
                            replicate2pids.set(rpid);
                        for(uint64_t i = 0; i < num_parts; i++){
                            if (i == l_pid || !replicate2pids[i]) continue;
                            remote_1_hop_edges_u[l_pid][i].push_back(u);
                            remote_1_hop_edges_v[l_pid][i].push_back(v);
                        }
                    }
                    //////////////////////////////////////////
                    // store 1 extra hop: END
                    //////////////////////////////////////////
                }

                if(l_pid == 0) TOK(construct_vc_relabel);

                //assign each sigle node
                for(uint64_t idx=l_pid; idx < vc_map.size(); idx+=num_parts){
                        if(get_mpid(vc_map[idx]) == VCR_MPID_MASK){
                            set_mpid(vc_map[idx], l_pid);
                            induced_nodes.push_back(idx);
                            set_lid(vc_map[idx], induced_nodes.size()-1);
                        }
                }

            }
            {
                if (vc_debug_level == 1 && l_pid == 0){
                    for(uint64_t i=0; i < num_parts; i++){
                        uint64_t max_row = pid2src[i].size() > 10 ? 10: pid2src[i].size();
                        LOG(INFO) << "Relabeled: Partition " << i;
                        for(uint64_t j = 0; j < max_row; j++){
                            LOG(INFO) << "(" << pid2src[i][j] << ", " << pid2dst[i][j] << ")";
                        }
                    }
                    LOG(INFO) << "VCMap:";
                    uint64_t max_row = vc_map.size() > 10 ? 10: vc_map.size();
                    for(uint64_t i=0; i < max_row; i++){
                        LOG(INFO) << "gid: " << i << "\t lid: " << get_lid(vc_map[i]) <<"\t pid: " << get_mpid(vc_map[i]);
                    }
                }
            }

            TIK(construct_vc_extend_1_hop);
#pragma omp barrier
            ////////////////////////////////////////////////////
            //extend 1 hop neighbor: BEGIN
            ////////////////////////////////////////////////////
            bool enter_loop = use_1_hop_halo;
            for(uint64_t i = 0; enter_loop && i < remote_1_hop_edges_u.size(); i++){
                auto& edge_src = remote_1_hop_edges_u[i][l_pid];
                auto& edge_dst = remote_1_hop_edges_v[i][l_pid];

                for(uint64_t idx=0; idx < edge_src.size(); idx++){
                    const vc_vid_t u = edge_src[idx];
                    const vc_vid_t v = edge_dst[idx];

                    auto it = seen_vid_to_nid.find(u);
                    if (it == seen_vid_to_nid.end()){
                        // just include 1 hop neighbor and treat them as inner node
                        induced_nodes.push_back(u);
                        seen_vid_to_nid[u] = induced_nodes.size()-1;
                        // extends 1 hop neighbors into current partition
                        pid2src[l_pid].push_back(induced_nodes.size()-1);
                    }else{
                        pid2src[l_pid].push_back(it->second);
                    }
                    it = seen_vid_to_nid.find(v);
                    if (it == seen_vid_to_nid.end()){
                        induced_nodes.push_back(v);
                        seen_vid_to_nid[v] = induced_nodes.size()-1;
                        pid2dst[l_pid].push_back(induced_nodes.size()-1);
                    }else{
                        pid2dst[l_pid].push_back(it->second);
                    }
                }
            }
            ////////////////////////////////////////////////////
            //extend 1 hop neighbor: END
            ////////////////////////////////////////////////////
#pragma omp barrier
            if(l_pid == 0) TOK(construct_vc_extend_1_hop);

            {
                if (vc_debug_level == 1 && l_pid == 0){
                    int max_row = pid2src[0].size() > 10 ? 10: pid2src[0].size();
                    LOG(INFO) << "Extended Partition " << 0;
                    for(int j = 0; j < max_row; j++){
                        LOG(INFO) << "(" << induced_nodes[pid2src[0][j]] << ", " << induced_nodes[pid2dst[0][j]] << ")";
                    }
                }
            }
            TIK(construct_vc_add_rev_sl_construct);
#pragma omp barrier
            if (l_pid == 0) LOG(INFO) << "Partition: " << l_pid <<", #induced_nodes: " << induced_nodes.size();

            // add reverse edge
            if (add_reverse_edge) {
                int old_size = pid2src[l_pid].size();
                pid2src[l_pid].resize(old_size + old_size);
                pid2dst[l_pid].resize(old_size + old_size);

                std::copy(pid2dst[l_pid].begin(), pid2dst[l_pid].begin() + old_size, pid2src[l_pid].begin() + old_size);
                std::copy(pid2src[l_pid].begin(), pid2src[l_pid].begin() + old_size, pid2dst[l_pid].begin() + old_size);
            }
            // add self loop
            if (add_self_loop) {
                //self-loop was not added
                if(add_self_loop_only_main){
                    for (uint64_t i=0; i < induced_nodes.size(); i++) {
                        if(get_mpid(vc_map[induced_nodes[i]]) == l_pid){
                            pid2src[l_pid].push_back(i);
                            pid2dst[l_pid].push_back(i);
                        }
                    }
                } else {
                    int old_size = pid2src[l_pid].size();
                    pid2src[l_pid].resize( old_size + induced_nodes.size() );
                    pid2dst[l_pid].resize( old_size + induced_nodes.size() );

                    for(uint64_t i=old_size; i < pid2src[l_pid].size(); i++){
                        pid2src[l_pid][i]=i-old_size;
                        pid2dst[l_pid][i]=i-old_size;
                    }
                }
            }

            //contruct sub graph with new id
            IdArray s_vids = aten::VecToIdArray(pid2src[l_pid]);
            IdArray d_vids = aten::VecToIdArray(pid2dst[l_pid]);

            aten::COOMatrix coo(induced_nodes.size(), induced_nodes.size(), s_vids, d_vids);
            HeteroGraphPtr subg = CreateFromCOO(1, coo, ALL_CODE);
            HeteroSubgraph hsg;
            hsg.graph = subg;
            hsg.induced_vertices = {aten::VecToIdArray(induced_nodes)};
            EdgeArray ea = subg->Edges(0);
            hsg.induced_edges = {ea.id};
            if(l_pid == 0){
                hsg.vc_maps = {aten::VecToIdArray(vc_map, 64)};
            }
            std::shared_ptr<HeteroSubgraph> subg_ptr(
                new HeteroSubgraph(hsg));
            subgs[l_pid] = subg_ptr;
	    
            if(l_pid==0) {
                TOK(construct_vc_add_rev_sl_construct);
            }
        }
    });

}


}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_VC_H_
