#pragma once
// stage2_full.hpp —— 论文完整方法的落地实现（严格对应 Fig.1 / §3）
//
// 逐条对应关系：
//   ① 混合分引导的候选池检索 (Fig.1 "Hybrid Matching: Influence Score Guided Candidate Pool")
//        score = (1-α)·norm_dist − α·norm_deg，α 自适应         imp_hnsw.hpp:875 / :843-856
//   ② proxy 锚定集 + 重叠下限 min_overlap = ⌈K·τ⌉                :3200
//   ③ 1-to-1 Worst Replacement（换掉最差的锚成员）                :3325
//   ④ 2-to-2 Fallback Replacement（1v1 触底时的兜底，防死锁）      :3394 / :2561
//   ⑤ slack 驱动的三档自适应许可（aggressive / conservative / locked） §3.3
//   ⑥ 替换资格：只允许换掉"距离最远"的锚成员（保护真实召回）       本轮新增
//
// 与原始代码的适配：uint8 距离、CSR 图、epoch 标记数组。

#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <queue>

#include "stage3_core.hpp"

template <typename IdxT, typename VecT>
struct FullOut {
    double recall = 0, inf = 0, ms = 0;
    double swaps1 = 0, swaps2 = 0;     // 平均 1v1 / 2v2 次数
    std::vector<std::vector<int>> S;
};

// ---------------------------------------------------------------- 混合分引导的图搜索（Stage-1）
template <typename IdxT, typename VecT, typename GraphT>
static std::vector<std::pair<float, int>> hybrid_search_full(IdxT& idx, const VecT* q, size_t dim,
                                                        size_t k, size_t ef, const GraphT& sg,
                                                        float alpha, float max_d,
                                                        std::vector<uint32_t>& vis, uint32_t& vis_ep) {
    using P = std::pair<float, int>;
    std::vector<P> res;
    if (idx.cur_element_count_ == 0) return res;
    const float max_deg = 300.0f;
    auto sc = [&](float d, int id) {
        return (1.0f - alpha) * (d / max_d) - alpha * ((float)sg.get_degree(id) / max_deg);
    };
    int cur = (int)idx.enterpoint_node_;
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
    ++vis_ep;
    auto visited_test = [&](int id) { if (vis[(size_t)id] == vis_ep) return true; vis[(size_t)id] = vis_ep; return false; };
    std::priority_queue<P> top;
    std::priority_queue<P, std::vector<P>, std::greater<P>> cand;
    float s0 = sc(vec_L2sqr(q, idx.GetDataByInternalID(cur), dim), cur);
    top.emplace(-s0, cur);
    cand.emplace(s0, cur);
    vis[(size_t)cur] = vis_ep;
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
            if (visited_test(nid)) continue;
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

// ---------------------------------------------------------------- 完整方法
template <typename IdxT, typename VecT, typename GraphT>
static FullOut<IdxT, VecT> run_full(IdxT& idx, const VecT* qs, size_t dim, int nq, int K, float tau,
                                    const std::vector<std::vector<int>>& gt, const GraphT& sg,
                                    size_t N, const std::vector<std::pair<int, int>>* bydeg, int M,
                                    int L_fetch, size_t ef, int elig = 1, bool use_hybrid = false,
                                    int theta_cons = 3, bool use_2v2 = true) {
    FullOut<IdxT, VecT> out;
    out.S.resize((size_t)nq);
    const int min_overlap = (int)std::ceil((float)K * tau);
    const int max_replace = std::max(0, K - min_overlap);
    const float max_d = (float)dim * 255.0f * 255.0f;

    std::vector<uint32_t> mark_cov(N ? N : 1, 0), mark_S(N ? N : 1, 0), vis_h(N ? N : 1, 0);
    uint32_t ep_h = 0;
    uint32_t ep_cov = 0, ep_S = 0;
    double rr = 0, ii = 0, ms = 0, s1 = 0, s2 = 0;

    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const VecT* q = qs + (size_t)qi * dim;

        // ---- ① proxy 锚定集（纯物理 top-K，顺带记录距离用于自适应 α）----
        std::vector<int> proxy;
        float avg_d = 0.0f;
        {
            auto pq = idx.Search(q, (size_t)K, ef);
            double sd = 0; int n = 0;
            while (!pq.empty()) { proxy.push_back((int)pq.top().second); sd += pq.top().first; ++n; pq.pop(); }
            avg_d = n ? (float)(sd / n) : 0.0f;
        }
        std::set<int> inP(proxy.begin(), proxy.end());

        // ---- ② 候选池：混合分引导检索 ∪ 全局大 V 池 ----
        std::vector<int> cands;
        {
            // 候选池 = 全局大 V 池（bydeg 已按度数降序） ∪ 混合分引导检索结果
            if (bydeg) for (int m = 0; m < M && m < (int)bydeg->size(); ++m)
                cands.push_back((*bydeg)[(size_t)m].second);
            if (use_hybrid) {
                float alpha = std::min(0.8f, (avg_d / (max_d * 0.5f)) * 0.8f);
                auto hs = hybrid_search_full(idx, q, dim, (size_t)L_fetch, std::min<size_t>(ef, 96),
                                             sg, alpha, max_d, vis_h, ep_h);
                for (auto& pr : hs) cands.push_back(pr.second);
            }
        }

        // ---- ③ 初始 S = proxy ----
        std::vector<int> S = proxy;
        ++ep_S;
        for (int x : S) mark_S[(size_t)x] = ep_S;
        size_t cov_cnt = 0;
        auto rebuild_cov = [&]() {
            ++ep_cov; cov_cnt = 0;
            for (int x : S) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
                int v = (int)*p;
                if (mark_cov[(size_t)v] != ep_cov) { mark_cov[(size_t)v] = ep_cov; ++cov_cnt; }
            }
        };
        rebuild_cov();
        auto gain_of = [&](int c) {
            int g = 0;
            for (const auto* p = sg_begin(sg, c); p != sg_end(sg, c); ++p)
                if (mark_cov[(size_t)(int)*p] != ep_cov) ++g;
            return g;
        };
        auto apply_swap = [&](int vout, int cin) {
            S.erase(std::remove(S.begin(), S.end(), vout), S.end());
            mark_S[(size_t)vout] = 0;
            S.push_back(cin);
            mark_S[(size_t)cin] = ep_S;
        };

        // ---- ④ 迭代替换：slack 三档（aggressive / conservative / locked）+ 1v1 + 2v2 ----
        auto cov_size = [&](const std::vector<int>& T) {
            ++ep_cov; size_t c = 0;
            for (int x : T) for (const auto* p = sg_begin(sg, x); p != sg_end(sg, x); ++p) {
                int v = (int)*p;
                if (mark_cov[(size_t)v] != ep_cov) { mark_cov[(size_t)v] = ep_cov; ++c; }
            }
            return c;
        };
        int replaced = 0;
        for (int iter = 0; iter < 2 * K + 2; ++iter) {
            int slack = (K - replaced) - min_overlap;          // 还能换掉几个锚成员
            if (slack > 0) {
                int theta = (slack >= 2) ? 1 : theta_cons;     // aggressive / conservative
                int v = -1;
                if (elig == 1) {                               // 换距离最远的锚成员（保护真实召回）
                    float bd = -1.0f;
                    for (int x : S) {
                        if (!inP.count(x)) continue;
                        float d = vec_L2sqr(q, idx.GetDataByInternalID(x), dim);
                        if (d > bd) { bd = d; v = x; }
                    }
                } else {                                       // 换度数最低的锚成员（原始口径）
                    float bg = 1e30f;
                    for (int x : S) {
                        if (!inP.count(x)) continue;
                        float dg = (float)sg.get_degree(x);
                        if (dg < bg) { bg = dg; v = x; }
                    }
                }
                if (v < 0) break;
                int bc = -1, bgn = 0;
                const int SCAN1 = std::min<int>(300, (int)cands.size());   // 候选已按度降序，扫前 300 足够
                for (int ci = 0; ci < SCAN1; ++ci) {
                    int c = cands[(size_t)ci];
                    if (mark_S[(size_t)c] == ep_S) continue;
                    int g = gain_of(c);
                    if (g > bgn) { bgn = g; bc = c; }   // 扫完整个候选窗口，取真正的最大边际增益
                }
                if (bc < 0 || bgn < theta) break;
                apply_swap(v, bc);
                ++replaced; s1 += 1.0;
                rebuild_cov();
                continue;
            }
            // ---- ⑤ locked：1v1 触底，用 2-to-2 兜底（防死锁）----
            if (!use_2v2 || replaced < 2) break;
            std::vector<int> freeS;
            for (int x : S) if (!inP.count(x)) freeS.push_back(x);
            if (freeS.size() < 2) break;                       // 没有两个可动的非锚成员
            std::sort(freeS.begin(), freeS.end(),
                      [&](int a, int b) { return sg.get_degree(a) < sg.get_degree(b); });
            int v1 = freeS[0], v2 = freeS[1];
            std::vector<int> base;
            for (int x : S) if (x != v1 && x != v2) base.push_back(x);
            size_t cur = cov_size(S);
            int c1 = -1; size_t best1 = cur;
            const int SCAN = std::min<int>(10, (int)cands.size());
            for (int ci = 0; ci < SCAN; ++ci) {
                int c = cands[(size_t)ci];
                if (mark_S[(size_t)c] == ep_S) continue;
                std::vector<int> tr = base; tr.push_back(c);
                size_t s = cov_size(tr);
                if (s > best1) { best1 = s; c1 = c; if (s >= cur + 120) break; }
            }
            if (c1 < 0) { rebuild_cov(); break; }
            int c2 = -1; size_t best2 = best1;
            for (int ci = 0; ci < SCAN; ++ci) {
                int c = cands[(size_t)ci];
                if (mark_S[(size_t)c] == ep_S || c == c1) continue;
                std::vector<int> tr = base; tr.push_back(c1); tr.push_back(c);
                size_t s = cov_size(tr);
                if (s > best2) { best2 = s; c2 = c; if (s >= best1 + 60) break; }
            }
            if (c2 < 0 || best2 <= cur) { rebuild_cov(); break; }
            // 执行 2v2
            S = base; S.push_back(c1); S.push_back(c2);
            mark_S[(size_t)v1] = 0; mark_S[(size_t)v2] = 0;
            mark_S[(size_t)c1] = ep_S; mark_S[(size_t)c2] = ep_S;
            s2 += 2.0;
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
    out.swaps1 = s1 / nq; out.swaps2 = s2 / nq;
    return out;
}

// ---------------------------------------------------------------- Stage-1 only（消融用）
// 只用混合分检索产出 top-K，不做任何替换。对应论文 §5 的 "hybrid-score retrieval only"。
template <typename IdxT, typename VecT, typename GraphT>
static BOut<IdxT> run_stage1_only(IdxT& idx, const VecT* qs, size_t dim, int nq, int K,
                                  const std::vector<std::vector<int>>& gt, const GraphT& sg,
                                  size_t N, size_t ef, float alpha_fixed = -1.0f, float max_d_override = -1.0f) {
    BOut<IdxT> out;
    const float max_d = (max_d_override > 0) ? max_d_override : (float)dim * 255.0f * 255.0f;
    std::vector<uint32_t> vis(N ? N : 1, 0), mark(N ? N : 1, 0);
    uint32_t ep_v = 0, ep_m = 0;
    double rr = 0, ii = 0, ms = 0;
    for (int qi = 0; qi < nq; ++qi) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const VecT* q = qs + (size_t)qi * dim;
        // 自适应 alpha 或固定 alpha
        float alpha = alpha_fixed;
        if (alpha < 0) {
            auto pq = idx.Search(q, (size_t)K, ef);
            double sd = 0; int n = 0;
            while (!pq.empty()) { sd += pq.top().first; pq.pop(); ++n; }
            float avg_d = n ? (float)(sd / n) : 0.0f;
            alpha = std::min(0.8f, (avg_d / (max_d * 0.5f)) * 0.8f);
        }
        auto hs = hybrid_search_full(idx, q, dim, (size_t)K, ef, sg, alpha, max_d, vis, ep_v);
        ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
        int h = 0; for (auto& pr : hs) if (gs.count(pr.second)) ++h;
        rr += (float)h / K;
        ++ep_m; size_t cov = 0;
        for (auto& pr : hs) for (const auto* p = sg_begin(sg, pr.second); p != sg_end(sg, pr.second); ++p) {
            int v = (int)*p; if (mark[(size_t)v] != ep_m) { mark[(size_t)v] = ep_m; ++cov; }
        }
        ii += (float)cov;
    }
    out.recall = rr / nq; out.inf = ii / nq; out.ms = ms / nq;
    return out;
}