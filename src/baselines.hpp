#pragma once
// baselines.hpp —— 审稿人要的外部对照基线（新文件）
//
// 三个基线，对应审稿意见：
//   1) CELF-Near   : 原实现口径（imp_hnsw.hpp:1656 SearchWithCELF）—— 在 top-2K 近邻里做
//                    CELF 惰性贪心，最大化一跳覆盖。这是"把影响力最大化套到检索候选上"的标准做法。
//   2) CELF-Global : 同一算法，但候选池换成全图按度排序的 top-M（不看 recall 约束）。
//                    用来展示"纯影响力最大化会牺牲召回"。
//   3) Greedy-Replace : R1(cdRU) 在 Q1 里点名的强基线 —— 保持 recall ≥ τ 的前提下，
//                    把最低匹配度的种子换成影响力最大的候选，迭代。
//
// 三者与主方法共用同一套索引、同一套 GT、同一套覆盖率口径。

#include <queue>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>

#include "stage3_core.hpp"

template <typename IdxT>
struct BOut { double recall = 0, inf = 0, ms = 0; };

// 通用 CELF（lazy greedy）最大覆盖：从 cands 里选 K 个，使一跳并集最大
template <typename IdxT, typename GraphT>
static std::vector<int> celf_select(IdxT& idx, const GraphT& sg, const std::vector<int>& cands,
                                    int K, size_t N, std::vector<uint32_t>& mark, uint32_t& ep) {
    struct NG { int gain; int id; bool operator<(const NG& o) const { return gain < o.gain; } };
    std::priority_queue<NG> pq;
    for (int id : cands) pq.push({(int)sg.get_degree(id), id});

    std::vector<int> sel;
    ++ep;
    size_t covered = 0;
    while ((int)sel.size() < K && !pq.empty()) {
        NG top = pq.top(); pq.pop();
        int actual = 0;
        for (const auto* p = sg_begin(sg, top.id); p != sg_end(sg, top.id); ++p)
            if (mark[(size_t)(int)*p] != ep) ++actual;
        if (!pq.empty() && actual < pq.top().gain) { pq.push({actual, top.id}); continue; }
        sel.push_back(top.id);
        for (const auto* p = sg_begin(sg, top.id); p != sg_end(sg, top.id); ++p) {
            int v = (int)*p;
            if (mark[(size_t)v] != ep) { mark[(size_t)v] = ep; ++covered; }
        }
        (void)covered;
    }
    return sel;
}

// ---------------------------------------------------------------- 1) CELF-Near
template <typename IdxT, typename VecT, typename GraphT>
static BOut<IdxT> run_celf_near(IdxT& idx, const VecT* qs, size_t dim, int nq, int K,
                                const std::vector<std::vector<int>>& gt, const GraphT& sg, size_t N) {
    BOut<IdxT> out;
    std::vector<uint32_t> mark(N ? N : 1, 0);
    uint32_t ep = 0;
    double rr = 0, ii = 0, ms = 0;
    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const VecT* q = qs + (size_t)qi * dim;
        std::vector<int> cands;
        auto pq = idx.Search(q, (size_t)(2 * K), (size_t)std::max(100, 4 * K));
        while (!pq.empty()) { cands.push_back((int)pq.top().second); pq.pop(); }
        auto S = celf_select(idx, sg, cands, K, N, mark, ep);
        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
        int h = 0; for (int x : S) if (gs.count(x)) ++h;
        rr += (float)h / K;
        ++ep; size_t cov = 0;
        for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
            int v = (int)*p; if (mark[(size_t)v] != ep) { mark[(size_t)v] = ep; ++cov; }
        }
        ii += (float)cov;
    }
    out.recall = rr / nq; out.inf = ii / nq; out.ms = ms / nq;
    return out;
}

// ---------------------------------------------------------------- 2) CELF-Global（无召回约束）
template <typename IdxT, typename VecT, typename GraphT>
static BOut<IdxT> run_celf_global(IdxT& idx, const VecT* qs, size_t dim, int nq, int K,
                                  const std::vector<std::vector<int>>& gt, const GraphT& sg,
                                  size_t N, const std::vector<std::pair<int, int>>& bydeg, int M) {
    BOut<IdxT> out;
    std::vector<int> pool;
    for (int i = 0; i < M && i < (int)bydeg.size(); ++i) pool.push_back(bydeg[(size_t)i].second);
    std::vector<uint32_t> mark(N ? N : 1, 0);
    uint32_t ep = 0;
    double rr = 0, ii = 0, ms = 0;
    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto S = celf_select(idx, sg, pool, K, N, mark, ep);   // 与查询无关 → 所有查询同一组
        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
        int h = 0; for (int x : S) if (gs.count(x)) ++h;
        rr += (float)h / K;
        ++ep; size_t cov = 0;
        for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
            int v = (int)*p; if (mark[(size_t)v] != ep) { mark[(size_t)v] = ep; ++cov; }
        }
        ii += (float)cov;
    }
    out.recall = rr / nq; out.inf = ii / nq; out.ms = ms / nq;
    return out;
}

// ---------------------------------------------------------------- 3) Greedy-Replace（R1 点名）
template <typename IdxT, typename VecT, typename GraphT>
static BOut<IdxT> run_greedy_replace(IdxT& idx, const VecT* qs, size_t dim, int nq, int K, float tau,
                                     const std::vector<std::vector<int>>& gt, const GraphT& sg,
                                     size_t N, const std::vector<std::pair<int, int>>& bydeg, int M, size_t ef) {
    BOut<IdxT> out;
    const int min_overlap = (int)std::ceil((float)K * tau);
    const int max_replace = std::max(0, K - min_overlap);
    std::vector<uint32_t> mark(N ? N : 1, 0);
    uint32_t ep = 0;
    double rr = 0, ii = 0, ms = 0;
    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const VecT* q = qs + (size_t)qi * dim;
        // 初始：物理 top-K
        std::vector<int> S; std::vector<float> D;
        {
            auto pq = idx.Search(q, (size_t)K, ef);
            while (!pq.empty()) { S.push_back((int)pq.top().second); D.push_back(pq.top().first); pq.pop(); }
        }
        std::set<int> inS(S.begin(), S.end());
        ++ep; size_t cov = 0;
        for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
            int v = (int)*p; if (mark[(size_t)v] != ep) { mark[(size_t)v] = ep; ++cov; }
        }
        // 迭代：把"匹配度最低"的种子换成"影响力增益最大"的候选（R1 的描述）
        for (int r = 0; r < max_replace; ++r) {
            // 最低匹配度 = 距离最大
            int v = 0; for (int i = 1; i < (int)D.size(); ++i) if (D[(size_t)i] > D[(size_t)v]) v = i;
            int vid = S[(size_t)v];
            // 换入：bydeg 池中边际增益最大且不在 S 里的
            int best_c = -1, best_gain = 0;
            for (int m = 0; m < M && m < (int)bydeg.size(); ++m) {
                int cid = bydeg[(size_t)m].second;
                if (inS.count(cid)) continue;
                int g = 0;
                for (const auto* p = sg_begin(sg, cid); p != sg_end(sg, cid); ++p)
                    if (mark[(size_t)(int)*p] != ep) ++g;
                if (g > best_gain) { best_gain = g; best_c = cid; break; }   // bydeg 有序，取第一个正增益
            }
            if (best_c < 0 || best_gain <= 0) break;
            // 执行
            S[(size_t)v] = best_c; D[(size_t)v] = -1.0f;   // 已替换的槽位不再参与"最差"筛选
            inS.erase(vid); inS.insert(best_c);
            ++ep; cov = 0;
            for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
                int yy = (int)*p; if (mark[(size_t)yy] != ep) { mark[(size_t)yy] = ep; ++cov; }
            }
        }
        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
        int h = 0; for (int x : S) if (gs.count(x)) ++h;
        rr += (float)h / K;
        ii += (float)cov;
    }
    out.recall = rr / nq; out.inf = ii / nq; out.ms = ms / nq;
    return out;
}
