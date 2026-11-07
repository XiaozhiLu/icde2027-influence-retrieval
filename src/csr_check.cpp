// csr_check.cpp —— CSR 社交图与框架 vector<vector<int>> 版本的结构等价性检查（新文件）
//
// 检查三件事：
//   1) 每个节点的度数是否相同
//   2) 每个节点的邻居多重集是否相同（排序后比）
//   3) 邻居顺序是否也相同（CSR 是按同一条边生成顺序填的，理应相同）
// 以及 bydeg（按度数排出的全局大 V 候选池）前 3000 是否逐个相同。
//
// 用法： csr_check.exe <N>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#include "retrieval_framework.hpp"
#include "social_csr.hpp"
#include "stage3_core.hpp"

using namespace std;
using namespace anns::graph;

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : 1000000;
    const int NC = 3000;

    auto t0 = chrono::steady_clock::now();
    SocialGraph vec(N);
    mkG_social(vec, N, 10);
    double tv = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    t0 = chrono::steady_clock::now();
    SocialGraphCSR csr;
    build_social_csr(csr, N, 10);
    double tc = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    printf("[build] vector=%.1f s  csr=%.1f s  csr内存=%.1f MB\n", tv, tc, (double)csr.bytes() / 1048576.0);

    size_t deg_bad = 0, nb_bad = 0, order_bad = 0;
    for (size_t i = 0; i < N; ++i) {
        if ((size_t)vec.get_degree((int)i) != (size_t)csr.get_degree((int)i)) { ++deg_bad; continue; }
        const auto& a = vec.get_neighbors((int)i);
        vector<int> b;
        for (const auto* p = csr.begin((int)i); p != csr.end((int)i); ++p) b.push_back((int)*p);
        if (a.size() != b.size()) { ++nb_bad; continue; }
        vector<int> as(a.begin(), a.end());
        vector<int> bs(b.begin(), b.end());
        sort(as.begin(), as.end());
        sort(bs.begin(), bs.end());
        if (as != bs) ++nb_bad;
        else if (vector<int>(a.begin(), a.end()) != b) ++order_bad;
    }

    auto bv = build_bydeg(vec, N, NC);
    auto bc = build_bydeg(csr, N, NC);
    size_t bydeg_bad = 0;
    for (int k = 0; k < NC; ++k) if (bv[(size_t)k] != bc[(size_t)k]) ++bydeg_bad;

    printf("[cmp] 度数不一致=%zu  邻居多重集不一致=%zu  邻居顺序不同=%zu  bydeg前%d不一致=%zu\n",
           deg_bad, nb_bad, order_bad, NC, bydeg_bad);
    printf("[top] vector: %d %d %d | csr: %d %d %d\n",
           bv[0].first, bv[1].first, bv[2].first, bc[0].first, bc[1].first, bc[2].first);
    bool ok = (deg_bad == 0 && nb_bad == 0 && bydeg_bad == 0);
    printf("=== RESULT: %s ===\n", ok ? "IDENTICAL" : "MISMATCH");
    return ok ? 0 : 1;
}
