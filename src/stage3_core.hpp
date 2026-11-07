#pragma once
// stage3_core.hpp —— stage3 逻辑的"共享可复用"版本（新文件，不改动 stage3_main.cpp）
//
// 目的：让 float 版 / uint8 版 / vector 图 / CSR 图 跑的是**同一份** 图生成、GT、τ 表代码，
// 这样"逐位一致"才有意义（不同代码路径算出来的相等不能证明存储改造无损）。
//
// 社交图有两种存储，接口用 sg_begin/sg_end 统一：
//   anns::graph::SocialGraph  框架自带 vector<vector<int>>（163 B/节点，100M 要 15.2 GiB）
//   SocialGraphCSR            本项目的 CSR（84 B/节点，100M 约 7.8 GiB）
// 两者边序列逐位相同（同一 seed 的 mt19937、同样的两遍扫描），因此 τ 表必须完全相同。

#include <vector>
#include <set>
#include <queue>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <string>

#include "retrieval_framework.hpp"
#include "u8_io.hpp"
#include "social_csr.hpp"

// ------------------------------------------------ 社交图（与 stage3_main.cpp 逐行等价）
static void mkG_social(anns::graph::SocialGraph& sg, size_t n, int e) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> d(0, (int)n - 1);
    int ni = (int)std::max<size_t>(1, n / 100);
    for (int i = 0; i < (int)n; ++i)
        for (int k = 0; k < e; ++k) {
            int t = (k == 0) ? (int)(rng() % ni) : d(rng);
            if (i != t) sg.add_edge(i, t);
        }
}

// ------------------------------------------------ 统一的邻居遍历接口
static inline const int* sg_begin(const anns::graph::SocialGraph& g, int u) {
    return g.get_neighbors(u).data();
}
static inline const int* sg_end(const anns::graph::SocialGraph& g, int u) {
    const auto& v = g.get_neighbors(u);
    return v.data() + v.size();
}
static inline const uint32_t* sg_begin(const SocialGraphCSR& g, int u) { return g.begin(u); }
static inline const uint32_t* sg_end(const SocialGraphCSR& g, int u) { return g.end(u); }

// ------------------------------------------------ 暴力 GT（模板化以便 float / uint8 各算一遍再比对）
template <typename VecT>
static void compute_gt(const VecT* base, size_t N, size_t dim, const VecT* qs, int nq, int K,
                       std::vector<std::vector<int>>& gt, int omp_threads) {
    gt.assign((size_t)nq, std::vector<int>());
#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (int i = 0; i < nq; ++i) {
        std::priority_queue<std::pair<float, int>> pq;
        for (size_t j = 0; j < N; ++j) {
            float dd = vec_L2sqr(qs + (size_t)i * dim, base + j * dim, dim);
            if (pq.size() < (size_t)K) pq.push({dd, (int)j});
            else if (dd < pq.top().first) { pq.pop(); pq.push({dd, (int)j}); }
        }
        gt[(size_t)i].resize((size_t)K);
        for (int k = K - 1; k >= 0; --k) { gt[(size_t)i][(size_t)k] = pq.top().second; pq.pop(); }
    }
}

// ------------------------------------------------ 按度数排出的全局大 V 候选池
template <typename GraphT>
static std::vector<std::pair<int, int>> build_bydeg(const GraphT& sg, size_t N, int NC) {
    std::vector<std::pair<int, int>> bydeg(N);
    for (size_t i = 0; i < N; ++i) bydeg[i] = {(int)sg.get_degree((int)i), (int)i};
    int nc = (int)std::min<size_t>((size_t)NC, N);
    std::partial_sort(bydeg.begin(), bydeg.begin() + nc, bydeg.end(), std::greater<std::pair<int, int>>());
    return bydeg;
}

// ------------------------------------------------ τ 表（与 stage3_main.cpp 的评测循环逐行等价）
template <typename IdxT, typename VecT>
struct TauOut {
    double recall = 0, inf = 0, ms = 0;
    uint64_t search_digest = 1469598103934665603ULL;  // 原始 Search 结果 (外部ID, float 位模式) 的 FNV-1a
    uint64_t sel_digest = 1469598103934665603ULL;     // 最终 K 个 ID 的 FNV-1a
    std::vector<std::vector<int>> S;
};

template <typename IdxT, typename VecT, typename GraphT>
static TauOut<IdxT, VecT> run_tau(IdxT& idx, const VecT* qs, size_t dim, int nq, float tau, int K, size_t ef,
                                  const std::vector<int>& ext, const std::vector<std::vector<int>>& gt,
                                  const GraphT& sg,
                                  const std::vector<std::pair<int, int>>& bydeg, int NC) {
    TauOut<IdxT, VecT> out;
    out.S.resize((size_t)nq);
    double rr = 0, ii = 0, ms = 0;
    int s0 = (int)std::ceil(tau * K);
    uint64_t sd = out.search_digest, qd = out.sel_digest;

    for (int i = 0; i < nq; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto pq = idx.Search(qs + (size_t)i * dim, K, ef);
        std::vector<int> S;
        while (!pq.empty()) {
            int eid = ext[(size_t)pq.top().second];
            sd = fnv1a_u64(sd, (uint64_t)(uint32_t)eid);
            sd = fnv1a_u64(sd, (uint64_t)f32_bits(pq.top().first));
            S.push_back(eid);
            pq.pop();
        }
        std::reverse(S.begin(), S.end());
        S.resize(std::min((size_t)s0, S.size()));

        std::set<int> cov;
        for (int x : S)
            for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) cov.insert((int)*p);
        while ((int)S.size() < K) {
            int best = -1, bg = -1;
            for (int k = 0; k < NC; ++k) {
                int c = bydeg[(size_t)k].second;
                if (std::find(S.begin(), S.end(), c) != S.end()) continue;
                int g = 0;
                for (const auto* p = sg_begin(sg, c); p != sg_end(sg, c); ++p)
                    if (!cov.count((int)*p)) g++;
                if (g > bg) { bg = g; best = c; }
                if (g >= 120) break;
            }
            if (best < 0) break;
            S.push_back(best);
            for (const auto* p = sg_begin(sg, best); p != sg_end(sg, best); ++p) cov.insert((int)*p);
        }
        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)i].begin(), gt[(size_t)i].end());
        int h = 0;
        for (int x : S) if (gs.count(x)) h++;
        rr += (float)h / K;
        ii += (float)cov.size();
        for (int x : S) qd = fnv1a_u64(qd, (uint64_t)(uint32_t)x);
        out.S[(size_t)i] = S;
    }
    out.recall = rr / nq;
    out.inf = ii / nq;
    out.ms = ms / nq;
    out.search_digest = sd;
    out.sel_digest = qd;
    return out;
}

// ------------------------------------------------ GT 缓存（避免每次评测重算暴力 GT）
// 文件格式：'GTQ1' + int32 nq + int32 K + nq*K int32
static inline bool save_gt_cache(const std::string& path, const std::vector<std::vector<int>>& gt, int K) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "[gt] 无法写入 " << path << std::endl; return false; }
    std::fwrite("GTQ1", 1, 4, f);
    int32_t nq = (int32_t)gt.size(), k = (int32_t)K;
    std::fwrite(&nq, 4, 1, f);
    std::fwrite(&k, 4, 1, f);
    for (const auto& v : gt) std::fwrite(v.data(), 4, (size_t)K, f);
    std::fclose(f);
    return true;
}

static inline bool load_gt_cache(const std::string& path, int nq, int K, std::vector<std::vector<int>>& gt) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t a = 0, b = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GTQ1", 4) != 0 ||
        std::fread(&a, 4, 1, f) != 1 || std::fread(&b, 4, 1, f) != 1 || a != nq || b != K) {
        std::fclose(f);
        return false;
    }
    gt.assign((size_t)nq, std::vector<int>((size_t)K));
    for (int i = 0; i < nq; ++i)
        if (std::fread(gt[(size_t)i].data(), 4, (size_t)K, f) != (size_t)K) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

// ------------------------------------------------ 跑完整 τ 表（可一次跑多个 ef）
// do_baseline=true 时每个 ef 先打印一行「纯 HNSW top-K」基线：
//   它就是 tau=1.0 的退化情形（s0 = K，之后不需要任何大 V 填充），
//   与 τ 行**完全同一套代码、同一套评测口径**，因此可以直接并列比较。
// 打印行里带 ef，方便把 ef 扫描的结果并列成表。
template <typename IdxT, typename VecT, typename GraphT>
static void run_all_taus(IdxT& idx, const VecT* qs, size_t dim, int nq, int K,
                         const std::vector<size_t>& efs,
                         const std::vector<int>& ext, const std::vector<std::vector<int>>& gt,
                         const GraphT& sg, size_t N, int NC,
                         bool do_baseline = true, bool do_tau = true) {
    auto bydeg = build_bydeg(sg, N, NC);
    printf("top degrees:");
    for (int k = 0; k < 5; ++k) printf(" %d", bydeg[(size_t)k].first);
    printf("\n");
    fflush(stdout);
    for (size_t ef : efs) {
        if (do_baseline) {
            auto r = run_tau<IdxT, VecT, GraphT>(idx, qs, dim, nq, 1.0f, K, ef, ext, gt, sg, bydeg, NC);
            printf("baseline ef=%zu recall=%.5f inf=%.2f %.4f ms  sd=%016llx sel=%016llx\n",
                   ef, r.recall, r.inf, r.ms,
                   (unsigned long long)r.search_digest, (unsigned long long)r.sel_digest);
            fflush(stdout);
        }
        if (do_tau) {
            for (float tau : {0.7f, 0.8f, 0.9f}) {
                auto r = run_tau<IdxT, VecT, GraphT>(idx, qs, dim, nq, tau, K, ef, ext, gt, sg, bydeg, NC);
                printf("tau=%.1f ef=%zu recall=%.5f inf=%.2f %.4f ms  sd=%016llx sel=%016llx\n",
                       (double)tau, ef, r.recall, r.inf, r.ms,
                       (unsigned long long)r.search_digest, (unsigned long long)r.sel_digest);
                fflush(stdout);
            }
        }
    }
}
