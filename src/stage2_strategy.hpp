#pragma once
// stage2_strategy.hpp —— 论文最终算法的移植版（Strategy-3 语义）：proxy 集 + 重叠下限 + 受许可限制的替换
//
// 依据（inputs/repo/imp_hnsw.hpp 的原始实现）：
//   混合分公式   : score = (1-α)·norm_d - α·norm_i                       (:875)
//   自适应 α     : α = min(0.8, ratio·0.8), ratio = 邻域平均距离/(max_d·0.5) (:843-856)
//   proxy 集     : 纯物理 Search(q, K, ef)                                (:3171, :3490)
//   重叠下限     : min_overlap = ⌈K·τ⌉                                     (:3200, :3495)
//   替换预算     : max_replace = K − min_overlap                          (:3202)
//   预熔断       : 替换后重叠 < min_overlap 则否决                          (:3325, :3615)
//
// 工程适配（不改语义）：uint8 距离归一化用 max_d = D·255²；社交图用 CSR；
//   候选池用一次 HNSW 检索 + 混合分重排（原实现是 SearchDynamicAlpha 内置引导）；
//   暂不做 2-to-2 交换，只做 1-to-1。

#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>

#include "stage3_core.hpp"
// ---------------------------------------------------------------- 混合分引导的图搜索
// 对应原始实现里的 SearchDynamicAlpha：扩张时按混合分排序，而不是纯距离，
// 这样候选池里才会出现"向量空间里远、但社交影响力大"的用户。
template <typename IdxT, typename VecT, typename GraphT>
static std::vector<std::pair<float, int>> hybrid_search(IdxT& idx, const VecT* q, size_t dim,
                                                        size_t k, size_t ef, const GraphT& sg,
                                                        float alpha, float max_d) {
    using P = std::pair<float, int>;
    std::vector<P> res;
    if (idx.cur_element_count_ == 0) return res;
    const float max_deg = 300.0f;
    auto sc = [&](float d, int id) {
        return (1.0f - alpha) * (d / max_d) - alpha * ((float)sg.get_degree(id) / max_deg);
    };

    // 上层：纯距离贪心下降到 level 0
    int cur = idx.enterpoint_node_;
    float cd = vec_L2sqr(q, idx.GetDataByInternalID(cur), dim);
    for (int lev = idx.element_levels_[cur]; lev > 0; --lev) {
        bool changed = true;
        while (changed) {
            changed = false;
            const size_t* ll = (const size_t*)idx.GetLinkByInternalID(cur, lev);
            const int* nb = (const int*)(ll + 1);
            for (size_t i = 0; i < ll[0]; ++i) {
                float d = vec_L2sqr(q, idx.GetDataByInternalID(nb[i]), dim);
                if (d < cd) { cd = d; cur = nb[i]; changed = true; }
            }
        }
    }

    // level 0：按混合分做 beam search
    std::vector<char> visited(idx.max_elements_, 0);
    std::priority_queue<P> top;                 // 存 (score, id)，score 越小越好 → 用 -score
    std::priority_queue<P, std::vector<P>, std::greater<P>> cand;  // 小顶堆：(score, id)
    float s0 = sc(vec_L2sqr(q, idx.GetDataByInternalID(cur), dim), cur);
    top.emplace(-s0, cur);
    cand.emplace(s0, cur);
    visited[(size_t)cur] = 1;
    float low = s0;
    while (!cand.empty()) {
        auto el = cand.top();
        if (el.first > low && top.size() == ef) break;
        cand.pop();
        int cid = el.second;
        const size_t* ll = (const size_t*)idx.GetLinkByInternalID(cid, 0);
        const int* nb = (const int*)(ll + 1);
        for (size_t i = 0; i < ll[0]; ++i) {
            int nid = nb[i];
            if (visited[(size_t)nid]) continue;
            visited[(size_t)nid] = 1;
            float d = vec_L2sqr(q, idx.GetDataByInternalID(nid), dim);
            float s = sc(d, nid);
            if (top.top().first > -s || top.size() < ef) {
                cand.emplace(s, nid);
                top.emplace(-s, nid);
                if (top.size() > ef) top.pop();
                low = -top.top().first;
            }
        }
    }
    res.reserve(top.size());
    while (!top.empty()) { res.push_back({-top.top().first, top.top().second}); top.pop(); }
    std::sort(res.begin(), res.end());
    if (res.size() > k) res.resize(k);
    return res;
}

template <typename IdxT, typename VecT>
struct S3Out {
    double recall = 0, inf = 0, ms = 0;
    double swaps = 0;                 // 平均每查询替换次数
    std::vector<std::vector<int>> S;
};

// tau: 召回目标；L_fetch: 候选池大小；ef: 检索预算
template <typename IdxT, typename VecT, typename GraphT>
static S3Out<IdxT, VecT> run_strategy3(IdxT& idx, const VecT* qs, size_t dim, int nq, int K, float tau,
                                       const std::vector<std::vector<int>>& gt,
                                       const GraphT& sg, int L_fetch, size_t ef,
                                       const std::vector<std::pair<int,int>>* bydeg = nullptr, int M_glob = 0,
                                       size_t N = 0, int elig = 1) {
    S3Out<IdxT, VecT> out;
    out.S.resize((size_t)nq);
    const int min_overlap = (int)std::ceil((float)K * tau);   // 铁血底线
    const int max_replace = std::max(0, K - min_overlap);     // 可替换的 proxy 数
    const float max_d = (float)dim * 255.0f * 255.0f;
    const float max_deg = 300.0f;

    double rr = 0, ii = 0, ms = 0, sw = 0;
    // epoch 标记：覆盖集/在集判定 O(1)，避免 std::set 的对数开销
    std::vector<uint32_t> cov_mark(N ? N : 1, 0), in_s_mark(N ? N : 1, 0);
    uint32_t cov_ep = 0, ins_ep = 0;

    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const VecT* q = qs + (size_t)qi * dim;

        // ---- 1) proxy truth：纯物理 top-K ----
        std::vector<int> proxy;
        {
            auto pq = idx.Search(q, (size_t)K, ef);
            while (!pq.empty()) { proxy.push_back((int)pq.top().second); pq.pop(); }
        }
        std::set<int> inP(proxy.begin(), proxy.end());

        // ---- 2) 候选池：物理检索 + 混合分重排（自适应 α） ----
        struct Cand { int id; float d; float deg; float score; };
        std::vector<Cand> cands;
        {
            auto pq = idx.Search(q, (size_t)L_fetch, ef);
            double sum_d = 0; int n = 0;
            while (!pq.empty()) {
                int id = (int)pq.top().second; float d = pq.top().first; pq.pop();
                cands.push_back({id, d, (float)sg.get_degree(id), 0.0f});
                sum_d += d; ++n;
            }
            float avg_d = n ? (float)(sum_d / n) : 0.0f;
            float ratio = avg_d / (max_d * 0.5f);
            float alpha = std::min(0.8f, ratio * 0.8f);                 // 自适应 α
            for (auto& c : cands) {
                float norm_d = c.d / max_d, norm_i = c.deg / max_deg;
                c.score = (1.0f - alpha) * norm_d - alpha * norm_i;      // 越小越好
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b) { return a.score < b.score; });
        }
        // 追加：全局按度数排序的大 V 池（对应论文的"双源候选检索"）
        if (bydeg) {
            std::set<int> have;
            for (const auto& c : cands) have.insert(c.id);
            for (int m = 0; m < M_glob && m < (int)bydeg->size(); ++m) {
                int id = (*bydeg)[(size_t)m].second;
                if (have.count(id)) continue;
                have.insert(id);
                cands.push_back({id, 0.0f, (float)sg.get_degree(id), 1.0f});
            }
        }

        // ---- 3) 初始集合 = proxy；覆盖集 = 一跳并集 ----
        std::vector<int> S = proxy;
        ++ins_ep;
        for (int x : S) in_s_mark[(size_t)x] = ins_ep;
        size_t cov_cnt = 0;
        auto rebuild_cov = [&]() {
            ++cov_ep; cov_cnt = 0;
            for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
                int v = (int)*p;
                if (cov_mark[(size_t)v] != cov_ep) { cov_mark[(size_t)v] = cov_ep; ++cov_cnt; }
            }
        };
        rebuild_cov();

        // ---- 4) 受许可限制的替换 ----
        int replaced = 0;
        while (replaced < max_replace) {
            // 换出：elig=0 换度数最低者（原论文口径）；elig=1 换距离最远者（保护真实召回）
            int best_v = -1;
            if (elig == 0) {
                float best_v_deg = 1e30f;
                for (int v : S) {
                    if (!inP.count(v)) continue;
                    float dg = (float)sg.get_degree(v);
                    if (dg < best_v_deg) { best_v_deg = dg; best_v = v; }
                }
            } else {
                float best_v_d = -1.0f;
                for (int v : S) {
                    if (!inP.count(v)) continue;
                    float d = vec_L2sqr(q, idx.GetDataByInternalID(v), dim);
                    if (d > best_v_d) { best_v_d = d; best_v = v; }
                }
            }
            if (best_v < 0) break;
            // 换入：非 S 成员中边际覆盖增益最大者
            int best_c = -1, best_gain = 0;
            for (const auto& c : cands) {
                if (in_s_mark[(size_t)c.id] == ins_ep) continue;
                int gain = 0;
                for (const auto* p = sg_begin(sg, c.id); p != sg_end(sg, c.id); ++p)
                    if (cov_mark[(size_t)(int)*p] != cov_ep) ++gain;
                if (gain > best_gain) { best_gain = gain; best_c = c.id; if (gain >= 120) break; }
            }
            if (best_c < 0 || best_gain <= 0) break;
            // 执行替换并重算覆盖（K=10、度≈20，代价可忽略）
            S.erase(std::remove(S.begin(), S.end(), best_v), S.end());
            in_s_mark[(size_t)best_v] = 0;
            S.push_back(best_c);
            in_s_mark[(size_t)best_c] = ins_ep;
            ++replaced; sw += 1.0;
            rebuild_cov();
        }

        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
        int h = 0; for (int x : S) if (gs.count(x)) ++h;
        rr += (float)h / K;
        ii += (float)cov_cnt;
        out.S[(size_t)qi] = S;
    }
    out.recall = rr / nq; out.inf = ii / nq; out.ms = ms / nq;
    out.swaps = sw / nq;
    return out;
}
