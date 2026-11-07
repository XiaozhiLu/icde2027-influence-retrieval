// ciaodvd_main.cpp —— CiaoDVD 真实信任图实验入口
// 数据来自 scripts/preprocess_ciaodvd.py 的产物：
//   user_emb.bin            [int32 n][int32 d] + float32 数据（行优先，已 L2 归一化）
//   item_emb.bin            同上，物品嵌入
//   query_items.txt         每行一个物品 id（作为查询）
//   item_test_positives.txt 每行 "item_id u1 u2 ..."（该物品的留出正评用户 = 相关性真值）
//   trust_edges.txt         每行 "u v"（信任边，已重映射为 0..n-1）
//
// 任务：给定一个物品，检索 K 个用户；Recall@K = |S ∩ positives| / K；
//       影响力 = S 的一跳并集大小；社交图 = 信任图。
//
// 用法：
//   ./ciaodvd --dir data/processed/ciaodvd --K 10 --tau 0.7
//   ./ciaodvd --dir ... --K 50 --method=greedy     // 贪心基线
//   ./ciaodvd --dir ... --K 10 --method=stage1     // 只用混合分检索
//   ./ciaodvd --dir ... --K 10 --method=base       // 纯 HNSW

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

#include "retrieval_framework.hpp"
#include "social_csr.hpp"
#include "stage3_core.hpp"
#include "baselines.hpp"
#include "stage2_full.hpp"
#include "baselines.hpp"

using namespace anns::graph;

static std::vector<std::pair<int, int>> load_edges_txt(const std::string& path, size_t n) {
    std::ifstream in(path);
    std::vector<std::pair<int, int>> e;
    int u, v;
    while (in >> u >> v) {
        if (u == v || u < 0 || v < 0 || (size_t)u >= n || (size_t)v >= n) continue;
        e.emplace_back(u, v);
    }
    return e;
}

static void build_csr_from_edges(SocialGraphCSR& g, size_t n, const std::vector<std::pair<int, int>>& edges) {
    g.n_ = n;
    std::vector<uint32_t> deg(n, 0);
    for (auto& e : edges) { ++deg[(size_t)e.first]; ++deg[(size_t)e.second]; }
    g.off_.assign(n + 1, 0);
    size_t acc = 0;
    for (size_t i = 0; i < n; ++i) { g.off_[i] = (uint32_t)acc; acc += deg[i]; }
    g.off_[n] = (uint32_t)acc;
    g.adj_.resize(acc);
    std::vector<uint32_t> cur(g.off_.begin(), g.off_.end() - 1);
    for (auto& e : edges) {
        g.adj_[cur[(size_t)e.first]++] = (uint32_t)e.second;
        g.adj_[cur[(size_t)e.second]++] = (uint32_t)e.first;
    }
    double avg = n ? (double)acc / (double)n : 0.0;
    std::printf("[csr] 真实信任图 N=%zu 边=%zu 平均度=%.2f 内存=%.1f MB\n",
                n, acc / 2, avg, (double)g.bytes() / 1048576.0);
}

static bool load_matrix_bin(const std::string& path, std::vector<float>& mat, size_t& n, size_t& d) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t hd[2] = {0, 0};
    if (std::fread(hd, sizeof(int32_t), 2, f) != 2) { std::fclose(f); return false; }
    n = (size_t)hd[0]; d = (size_t)hd[1];
    mat.resize(n * d);
    size_t got = std::fread(mat.data(), sizeof(float), n * d, f);
    std::fclose(f);
    std::printf("[load] %s  %zux%zu  (读到 %zu 个 float)\n", path.c_str(), n, d, got);
    return got == n * d;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::string dir = "data/processed/ciaodvd";
    int K = 10;
    float tau = 0.7f;
    std::string method = "ours";     // ours | greedy | stage1 | base
    int elim = 1;                    // 1=farthest(默认) 0=degree
    bool use_2v2 = true;
    std::string celf_mode = "near";  // near | global
    std::string dump_path = "";
    int M = 16, efC = 200;
    size_t ef = 200;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
        else if (a.rfind("--K=", 0) == 0) K = std::atoi(a.c_str() + 4);
        else if (a.rfind("--tau=", 0) == 0) tau = (float)std::atof(a.c_str() + 6);
        else if (a.rfind("--ef=", 0) == 0) ef = (size_t)std::atoi(a.c_str() + 5);
        else if (a.rfind("--method=", 0) == 0) method = a.substr(9);
        else if (a.rfind("--M=", 0) == 0) M = std::atoi(a.c_str() + 4);
        else if (a == "--elig=degree") elim = 0;
        else if (a == "--elig=farthest") elim = 1;
        else if (a == "--no2v2") use_2v2 = false;
        else if (a.rfind("--celf=", 0) == 0) celf_mode = a.substr(7);
        else if (a.rfind("--dump=", 0) == 0) dump_path = a.substr(7);
    }

    std::vector<float> U, I;
    size_t nu = 0, du = 0, ni = 0, di = 0;
    if (!load_matrix_bin(dir + "/user_emb.bin", U, nu, du)) { std::fprintf(stderr, "缺 user_emb.bin\n"); return 1; }
    if (!load_matrix_bin(dir + "/item_emb.bin", I, ni, di)) { std::fprintf(stderr, "缺 item_emb.bin\n"); return 1; }
    if (du != di) { std::fprintf(stderr, "维度不一致\n"); return 1; }

    // 查询物品 + 相关性真值
    std::vector<int> qitems;
    {
        std::ifstream in(dir + "/query_items.txt");
        int x; while (in >> x) qitems.push_back(x);
    }
    // GT = 精确 kNN（暴力扫描所有 user 的 L2 距离），与原实现 ComputeGroundTruth 完全一致
    std::vector<std::vector<int>> gt(qitems.size());
    {
        std::printf("[gt] 暴力计算 %zu 个查询的 top-%d ...\n", qitems.size(), K);
        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int qi = 0; qi < (int)qitems.size(); ++qi) {
            int it = qitems[(size_t)qi];
            if ((size_t)it >= ni) continue;
            const float* qv = I.data() + (size_t)it * du;
            std::priority_queue<std::pair<float, int>> pq;
            for (size_t j = 0; j < nu; ++j) {
                float d = 0.0f;
                const float* uv = U.data() + j * du;
                for (size_t t = 0; t < du; ++t) { float x = qv[t] - uv[t]; d += x * x; }
                if ((int)pq.size() < K) pq.push({d, (int)j});
                else if (d < pq.top().first) { pq.pop(); pq.push({d, (int)j}); }
            }
            gt[(size_t)qi].resize((size_t)K);
            for (int k = K - 1; k >= 0; --k) { gt[(size_t)qi][(size_t)k] = pq.top().second; pq.pop(); }
        }
        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        std::printf("[gt] 完成 %.1f s\n", sec);
    }
    std::printf("[data] 用户=%zu 物品=%zu 查询=%zu 维度=%zu\n", nu, ni, qitems.size(), du);

    // 索引
    auto idx = std::make_unique<HNSW<float>>(du, nu, (size_t)M, (size_t)efC);   // (D, max_elements, M, efC)
    idx->SetNumThreads(1);
    for (size_t i = 0; i < nu; ++i) idx->AddPoint(U.data() + i * du);
    std::printf("[index] 建好 %zu 个用户\n", idx->cur_element_count_);

    // 社交图
    SocialGraphCSR g;
    {
        auto edges = load_edges_txt(dir + "/trust_edges.txt", nu);
        build_csr_from_edges(g, nu, edges);
    }

    // 查询向量
    std::vector<float> Q;
    Q.reserve(qitems.size() * du);
    for (int it : qitems) {
        if ((size_t)it >= ni) { Q.insert(Q.end(), du, 0.0f); continue; }
        Q.insert(Q.end(), I.begin() + (size_t)it * du, I.begin() + (size_t)it * du + du);
    }
    int nq = (int)qitems.size();
    if (nq > 200) nq = 200;   // 与原实现一致：只用前若干个查询

    auto bydeg = build_bydeg(g, nu, (int)nu);

    if (method == "base") {
        double rr = 0, ii = 0, ms = 0;
        std::vector<uint32_t> mark(nu, 0); uint32_t ep = 0;
        for (int qi = 0; qi < nq; ++qi) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto pq = idx->Search(Q.data() + (size_t)qi * du, (size_t)K, ef);
            std::vector<int> S;
            while (!pq.empty()) { S.push_back((int)pq.top().second); pq.pop(); }
            ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
            int h = 0; for (int x : S) if (gs.count(x)) ++h;
            rr += (float)h / K;
            ++ep; size_t cov = 0;
            for (int x : S) for (const auto* p = sg_begin(g, x); p != sg_end(g, x); ++p) {
                int v = (int)*p; if (mark[(size_t)v] != ep) { mark[(size_t)v] = ep; ++cov; }
            }
            ii += (float)cov;
        }
        std::printf("base    K=%d recall=%.5f inf=%.2f %.3f ms\n", K, rr / nq, ii / nq, ms / nq);
    } else if (method == "stage1") {
        auto r = run_stage1_only<HNSW<float>, float>(*idx, Q.data(), du, nq, K, gt, g, nu, ef, -1.0f, 4.0f);
        std::printf("stage1  K=%d recall=%.5f inf=%.2f %.3f ms\n", K, r.recall, r.inf, r.ms);
    } else if (method == "celf") {
        if (celf_mode == "global") {
            auto r = run_celf_global<HNSW<float>, float, SocialGraphCSR>(*idx, Q.data(), du, nq, K, gt, g, nu, bydeg, (int)nu);
            std::printf("celf-global K=%d recall=%.5f inf=%.2f %.3f ms\n", K, r.recall, r.inf, r.ms);
        } else {
            auto r = run_celf_near<HNSW<float>, float, SocialGraphCSR>(*idx, Q.data(), du, nq, K, gt, g, nu);
            std::printf("celf-near   K=%d recall=%.5f inf=%.2f %.3f ms\n", K, r.recall, r.inf, r.ms);
        }
    } else if (method == "greedy") {
        auto r = run_greedy_replace<HNSW<float>, float, SocialGraphCSR>(*idx, Q.data(), du, nq, K, tau, gt, g, nu, bydeg, (int)nu, ef);
        std::printf("greedy  K=%d tau=%.2f recall=%.5f inf=%.2f %.3f ms\n", K, (double)tau, r.recall, r.inf, r.ms);
    } else {
        auto r = run_full<HNSW<float>, float, SocialGraphCSR>(*idx, Q.data(), du, nq, K, tau, gt, g, nu,
                                                             &bydeg, (int)nu, K, ef, elim, false, 3, use_2v2);
        if (!dump_path.empty()) {
            FILE* fo = std::fopen(dump_path.c_str(), "w");
            if (fo) {
                for (int qi = 0; qi < nq; ++qi) {
                    std::set<int> gs(gt[(size_t)qi].begin(), gt[(size_t)qi].end());
                    int hits = 0;
                    for (int x : r.S[(size_t)qi]) if (gs.count(x)) ++hits;
                    std::set<int> cov;
                    for (int x : r.S[(size_t)qi])
                        for (const auto* p = sg_begin(g, x); p != sg_end(g, x); ++p) cov.insert((int)*p);
                    std::fprintf(fo, "%d %d %zu\n", qi, hits, cov.size());
                }
                std::fclose(fo);
            }
        }
        std::printf("ours    K=%d tau=%.2f recall=%.5f inf=%.2f %.3f ms 1v1=%.2f 2v2=%.2f\n",
                    K, (double)tau, r.recall, r.inf, r.ms, r.swaps1, r.swaps2);
    }
    return 0;
}
