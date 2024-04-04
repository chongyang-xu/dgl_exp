/**
 *  Copyright (c) 2023 by MPI-SWS DSG
 * @brief mmap a file as a data pointer
 */

#include "mmap_file.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <vector>

extern "C"
{
    void build(const char* path_in, const int64_t n_n, const int64_t n_e,
               const int64_t feat_dim, const int64_t n_part, const int64_t* src,
               const int64_t* dst, const int64_t* nid2pid);

    void split_node_feat(const char* path_in, const int64_t n_n,
                         const int64_t feat_dim, const int64_t n_part,
                         const float* node_feat);

    void split_node_data(const char* path_in, const int64_t n_n,
                         const int64_t n_part, const float* node_label,
                         const int8_t* train_mask, const int8_t* valid_mask,
                         const int8_t* test_mask);

    void print_string(const char* str);
}

void print_string(const char* str) { printf("%s", str); }

void build(const char* path_in, const int64_t n_n, const int64_t n_e,
           const int64_t feat_dim, const int64_t n_part, const int64_t* src,
           const int64_t* dst, const int64_t* nid2pid)
{
    printf("n_n: %ld\n", n_n);
    printf("n_e: %ld\n", n_e);
    printf("n_part: %ld\n", n_part);
    printf("n_dim: %ld\n", feat_dim);
    std::string path = std::string(path_in);
    printf("path: %s\n", path.c_str());

    std::vector<int64_t> ele;
    std::vector<std::vector<int64_t>> p_u;
    std::vector<std::vector<int64_t>> p_v;
    for (int i = 0; i < n_part; i++)
    {
        p_u.push_back(ele);
        p_v.push_back(ele);
    }

    std::vector<std::vector<std::vector<int64_t>>> b_u;
    std::vector<std::vector<std::vector<int64_t>>> b_v;
    for (int i = 0; i < n_part; i++)
    {
        b_u.push_back(p_u);
        b_v.push_back(p_u);
    }

    for (int64_t i = 0; i < n_e; i++)
    {
        auto u = src[i];
        auto v = dst[i];
        auto pid_u = nid2pid[u];
        auto pid_v = nid2pid[v];
        if (pid_u == pid_v)
        {
            p_u[pid_u].push_back(u);
            p_v[pid_u].push_back(v);
        }
        else
        {
            b_u[pid_u][pid_v].push_back(u);
            b_v[pid_u][pid_v].push_back(v);
        }
    }

    for (int pid = 0; pid < n_part; pid++)
    {
        std::unordered_map<int64_t, int64_t> seen_nid;
        std::vector<int64_t> induced_nodes;
        // (u, v)
        std::vector<int64_t> vec_u;
        std::vector<int64_t> vec_v;
        for (int i = 0; i < p_u[pid].size(); i++)
        {
            auto nidu = p_u[pid][i];
            auto nidv = p_v[pid][i];
            auto nidu_new = 0;
            auto nidv_new = 0;

            if (seen_nid.find(nidu) == seen_nid.end())
            {
                induced_nodes.push_back(nidu);
                seen_nid[nidu] = induced_nodes.size() - 1;
                nidu_new = induced_nodes.size() - 1;
            }
            else
            {
                nidu_new = seen_nid[nidu];
            }

            if (seen_nid.find(nidv) == seen_nid.end())
            {
                induced_nodes.push_back(nidv);
                seen_nid[nidv] = induced_nodes.size() - 1;
                nidv_new = induced_nodes.size() - 1;
            }
            else
            {
                nidv_new = seen_nid[nidv];
            }

            if (nidu == nidv)
            {
                continue;
            }
            else
            {
                vec_u.push_back(nidu_new);
                vec_v.push_back(nidv_new);
            }
        }
        /*
        { // assign nodes w/o edges into part 0
            if(pid == 0){
                for(size_t i = 0; i < n_n; i++){
                    if(seen_nid.find(i) == seen_nid.end()){
                        induced_nodes.push_back(i);
                        seen_nid[i] = induced_nodes.size() - 1;
                    }
                }
            }
        }*/
        {
            printf("induced_nodes.size()=%lu\n", induced_nodes.size());
            size_t num_edge = (vec_u.size() + induced_nodes.size());
            MmapFile mfu(path + "/p" + std::to_string(pid) + ".u.bin",
                         sizeof(int64_t) * num_edge);
            int64_t* uptr = mfu.AsInt64Ptr();
            for (size_t i = 0; i < vec_u.size(); i++)
            {
                uptr[i] = vec_u[i];
            }
            for (size_t i = vec_u.size(); i < num_edge; i++)
            {
                uptr[i] = i - vec_u.size();
            }
            printf("vec_u.size()=%lu\n", vec_u.size());
            printf("num_edge=%lu\n", num_edge);
        }
        {
            size_t num_edge = (vec_u.size() + induced_nodes.size());
            MmapFile mfv(path + "/p" + std::to_string(pid) + ".v.bin",
                         sizeof(int64_t) * num_edge);
            int64_t* vptr = mfv.AsInt64Ptr();
            for (size_t i = 0; i < vec_u.size(); i++)
            {
                vptr[i] = vec_v[i];
            }
            for (size_t i = vec_u.size(); i < num_edge; i++)
            {
                vptr[i] = i - vec_u.size();
            }
        }
        // original id
        // original id
        {
            MmapFile mfo(path + "/p" + std::to_string(pid) + ".oid.bin",
                         induced_nodes.size() * sizeof(int64_t));

            int64_t* o = mfo.AsInt64Ptr();
            for (int i = 0; i < induced_nodes.size(); i++)
            {
                o[i] = induced_nodes[i];
            }
        }

        // b_i_[0-k]
        // b_[0-k]_i
        for (int off = 0; off < n_part; off++)
        {
            if (pid == off)
            {
                continue;
            }
            {
                auto& l = b_u[pid][off];
                MmapFile mfbu(path + "/b" + std::to_string(pid) + "_" +
                                  std::to_string(off) + ".u.bin",
                              l.size() * sizeof(int64_t));
                int64_t* bu = mfbu.AsInt64Ptr();
                for (int i = 0; i < l.size(); i++)
                {
                    bu[i] = seen_nid[l[i]];
                }
            }
            {
                auto& l = b_v[off][pid];
                MmapFile mfbv(path + "/b" + std::to_string(off) + "_" +
                                  std::to_string(pid) + ".v.bin",
                              l.size() * sizeof(int64_t));
                int64_t* bv = mfbv.AsInt64Ptr();
                for (int i = 0; i < l.size(); i++)
                {
                    bv[i] = seen_nid[l[i]];
                }
            }
        }
    }

    int64_t pivot_sum = 0;
    for (int i = 0; i < n_part; i++)
    {
        pivot_sum += p_u[i].size();
        printf("p_u[%d] = %ld\n", i, p_u[i].size());
        for (int j = 0; j < n_part; j++)
        {
            printf("b_u[%2d][%2d] = %8ld\n", i, j, b_u[i][j].size());
        }
    }
    printf("pivot_sum = %ld\n", pivot_sum);
    printf("print sched 1/100:\n");
    for (int i = 0; i < n_part; i++)
    {
	printf("%2d:", i);
        for (int j = 0; j < n_part; j++)
        {
	    float r = float(b_u[i][j].size()) / float(n_e);
	    r = r * 100.0;
            if(r >= 1.0) 
	    	printf(" %2d", j);
        }
	printf("\n");
    }
    printf("print sched 2/1000:\n");
    for (int i = 0; i < n_part; i++)
    {
	printf("%2d:", i);
        for (int j = 0; j < n_part; j++)
        {
	    float r = float(b_u[i][j].size()) / float(n_e);
	    r = r * 1000.0;
            if(r >= 2.0) 
	    	printf(" %2d", j);
        }
	printf("\n");
    }


}

void split_node_feat(const char* path_in, const int64_t n_n,
                     const int64_t feat_dim, const int64_t n_part,
                     const float* node_feat)
{
    printf("n_n: %ld\n", n_n);
    printf("n_part: %ld\n", n_part);
    printf("n_dim: %ld\n", feat_dim);
    std::string path = std::string(path_in);
    printf("path: %s\n", path.c_str());
    for (int i = 0; i < n_part; i++)
    {
        int64_t* oid = NULL;
        size_t len = 0;
        {
            MmapFile foid(path + "/p" + std::to_string(i) + ".oid.bin");
            int64_t* ptr = foid.AsInt64Ptr();
            len = foid.GetLength();
            oid = static_cast<int64_t*>(malloc(len));
            if (oid == NULL)
            {
                printf("malloc failed %s:%d\n", __FILE__, __LINE__);
                return;
            }
            memcpy(oid, ptr, len);
            printf("copy to oid(%p) from ptr(%p), len=%lu bytes\n", oid, ptr, len);
        }
        {
            const size_t stride = feat_dim * sizeof(float);
            MmapFile ffeat(path + "/p" + std::to_string(i) + ".nfeat.bin",
                           len / sizeof(int64_t) * stride);
            float* feat = ffeat.AsFloat32Ptr();
            for (size_t id = 0; id < len / sizeof(int64_t); id++)
            {
                memcpy((feat + id * feat_dim), (node_feat + oid[id] * feat_dim),
                       stride);
            }
            free(oid);
        }
    }
}

void split_node_data(const char* path_in, const int64_t n_n,
                     const int64_t n_part, const float* node_label,
                     const int8_t* train_mask, const int8_t* valid_mask,
                     const int8_t* test_mask)
{

    printf("n_n: %ld\n", n_n);
    printf("n_part: %ld\n", n_part);
    std::string path = std::string(path_in);
    printf("path: %s\n", path.c_str());
    for (int i = 0; i < n_part; i++)
    {
        int64_t* oid = NULL;
        size_t len = 0;
        {
            MmapFile foid(path + "/p" + std::to_string(i) + ".oid.bin");
            int64_t* ptr = foid.AsInt64Ptr();
            len = foid.GetLength();
            oid = static_cast<int64_t*>(malloc(len));
            if (oid == NULL)
            {
                printf("malloc failed %s:%d\n", __FILE__, __LINE__);
                return;
            }
            memcpy(oid, ptr, len);
            printf("copy to oid(%p) from ptr(%p), len=%lu bytes\n", oid, ptr, len);
        }
        {
            MmapFile flabel(path + "/p" + std::to_string(i) + ".label.bin",
                            len / sizeof(int64_t) * sizeof(float));
            float* label = flabel.AsFloat32Ptr();
            for (int id = 0; id < len / sizeof(int64_t); id++)
            {
                label[id] = node_label[oid[id]];
            }
        }
        {
            MmapFile ftrain(path + "/p" + std::to_string(i) + ".train.bin",
                            len / sizeof(int64_t));
            int8_t* train = ftrain.AsInt8Ptr();
            for (int id = 0; id < len / sizeof(int64_t); id++)
            {
                train[id] = train_mask[oid[id]];
            }
        }
        {
            MmapFile fvalid(path + "/p" + std::to_string(i) + ".valid.bin",
                            len / sizeof(int64_t));
            int8_t* valid = fvalid.AsInt8Ptr();
            for (int id = 0; id < len / sizeof(int64_t); id++)
            {
                valid[id] = valid_mask[oid[id]];
            }
        }
        {
            MmapFile ftest(path + "/p" + std::to_string(i) + ".test.bin",
                           len / sizeof(int64_t));
            int8_t* test = ftest.AsInt8Ptr();
            for (int id = 0; id < len / sizeof(int64_t); id++)
            {
                test[id] = test_mask[oid[id]];
            }
        }
        free(oid);
    }
}
