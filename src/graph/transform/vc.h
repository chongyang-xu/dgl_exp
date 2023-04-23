/**
 * Copyright (c) 2023
 * ported from graphlab
 * @file graph/transform/vc.h
 * @brief vertex cut
 */

#ifndef DGL_GRAPH_TRANSFORM_VC_H_
#define DGL_GRAPH_TRANSFORM_VC_H_

#define MAX_N_PARTITION 64
#define MAX_N_NODE 300000000

#include <bitset>
#include <cassert>

namespace dgl {
namespace transform {

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

  size_t deg_u = src_true_degree + 1;
  size_t deg_v = dst_true_degree + 1;
  double SUM = deg_u + deg_v;
  double fu = static_cast<double>(deg_u) / SUM;
  double fv = 1.0 - fu;

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

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_VC_H_
