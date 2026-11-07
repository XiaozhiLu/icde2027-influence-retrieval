#pragma once
// social_csr.hpp —— 社交图的 CSR 紧凑存储（新文件，不改动任何既有文件）
//
// 为什么要有它：框架里的 SocialGraph 用 vector<vector<int>>，100M 节点就是
// 1 亿次独立分配，实测 163.2 B/节点 → 100M 要 15.2 GiB；换成 CSR 后是
// 4 B 偏移 + 4 B/边 = 84 B/节点 → 约 7.8 GiB，省 7.4 GiB。
//
// 关键：边生成顺序必须与 mkG_social（stage3_core.hpp）**逐位相同**，否则度数、
// bydeg 排序、τ 表都会变。做法是两遍扫描、每遍都从 seed=123 的 mt19937 重新开始，
// 所以两遍看到的是同一串 (i, k, t)；第二遍连"同一节点内邻居的先后顺序"都与
// vector push_back 版本一致（同一条边被 append 的次序相同）。
//
// 注意：框架的 HNSW::SetSocialGraph 只接受 SocialGraph*，但本项目的
// AddPoint / Search 路径根本不用社交图（只有 τ 评测用），所以 CSR 模式下不调用它。

#include <cstdint>
#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

struct SocialGraphCSR {
    size_t n_ = 0;
    std::vector<uint32_t> off_;   // n+1
    std::vector<uint32_t> adj_;   // 每条无向边在两个端点各存一次
    std::vector<uint32_t> cur_;   // 建图时的写游标（仅聚集图构建使用）
    bool drop_isolated_ = true;

    std::vector<uint32_t> degrees() const {
        std::vector<uint32_t> d(n_);
        for (size_t i = 0; i < n_; ++i) d[i] = off_[i + 1] - off_[i];
        return d;
    }
    inline uint32_t get_degree(int u) const {
        if ((size_t)u >= n_) return 0;
        return off_[(size_t)u + 1] - off_[(size_t)u];
    }
    inline const uint32_t* begin(int u) const {
        return adj_.data() + ((size_t)u < n_ ? off_[(size_t)u] : adj_.size());
    }
    inline const uint32_t* end(int u) const {
        return adj_.data() + ((size_t)u < n_ ? off_[(size_t)u + 1] : adj_.size());
    }
    size_t bytes() const { return (off_.size() + adj_.size()) * sizeof(uint32_t); }
};


// ---- 从外部边表文件构建 CSR（用于"真实图复制品"实验）----
#include <cstdio>
#include <string>
static void build_social_csr_from_file(SocialGraphCSR& g, size_t n, const std::string& path) {
    g.n_ = n;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "[csr] open fail: %s\n", path.c_str()); std::exit(1); }
    std::vector<std::pair<uint32_t, uint32_t>> E;
    E.reserve(1 << 21);
    unsigned u, v;
    while (std::fscanf(f, "%u %u", &u, &v) == 2) {
        if (u == v || u >= n || v >= n) continue;
        E.emplace_back(u, v);
    }
    std::fclose(f);
    std::vector<uint32_t> deg(n, 0);
    for (auto& e : E) { ++deg[(size_t)e.first]; ++deg[(size_t)e.second]; }
    g.off_.assign(n + 1, 0);
    size_t acc = 0;
    for (size_t i = 0; i < n; ++i) { g.off_[i] = (uint32_t)acc; acc += deg[i]; }
    g.off_[n] = (uint32_t)acc;
    g.adj_.resize(acc);
    std::vector<uint32_t> cur(g.off_.begin(), g.off_.end() - 1);
    for (auto& e : E) {
        g.adj_[cur[(size_t)e.first]++] = e.second;
        g.adj_[cur[(size_t)e.second]++] = e.first;
    }
    std::fprintf(stderr, "[csr] 外部图 %s  N=%zu 边=%zu 平均度=%.2f 内存=%.1f MB\n",
                 path.c_str(), n, E.size(), 2.0 * (double)E.size() / (double)n,
                 (double)g.bytes() / 1048576.0);
}
// ---- 度校正随机块模型（DCSBM, Karrer & Newman 2011）----
// 参数：mu = 跨社区边比例（越小社区越强）；gamma = 度分布幂律指数；C = 社区数。
// 生成方式：幂律度序列 → 按社区分组存 stub → 每个社区内部配对 (1-mu) 的 stub，
//           剩余 stub 进全局池随机配对。这是带社区结构的标准零模型。
static void build_social_csr_dcsbm(SocialGraphCSR& g, size_t n, double mu = 0.3,
                                   double gamma = 2.5, int dmin = 5, int dmax = 2000,
                                   int C = 200, uint32_t seed = 20261030u) {
    g.n_ = n;                     // ← 必须最先设置，否则 get_degree 全部返回 0
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<uint32_t> comm(n);
    for (size_t i = 0; i < n; ++i) comm[i] = (uint32_t)(i % (size_t)C);

    // 1) 幂律度序列
    std::vector<uint32_t> want(n);
    for (size_t i = 0; i < n; ++i) {
        double u = U(rng);
        double d = (double)dmin * std::pow(1.0 - u * 0.999, -1.0 / (gamma - 1.0));
        if (d > (double)dmax) d = (double)dmax;
        want[i] = (uint32_t)std::max(1.0, d);
    }

    // 2) 按社区收集 stub
    std::vector<std::vector<uint32_t>> pool((size_t)C);
    for (size_t i = 0; i < n; ++i) {
        auto& v = pool[comm[i]];
        for (uint32_t k = 0; k < want[i]; ++k) v.push_back((uint32_t)i);
    }

    // 3) 社区内配对 + 全局池
    std::vector<uint32_t> global_stubs;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(n * 8);
    for (int c = 0; c < C; ++c) {
        auto& v = pool[(size_t)c];
        std::shuffle(v.begin(), v.end(), rng);
        size_t keep = (size_t)((double)v.size() * (1.0 - mu));
        if (keep % 2) --keep;
        for (size_t i = 0; i + 1 < keep; i += 2) edges.emplace_back(v[i], v[i + 1]);
        for (size_t i = keep; i < v.size(); ++i) global_stubs.push_back(v[i]);
    }

    // 4) 全局池随机配对
    std::shuffle(global_stubs.begin(), global_stubs.end(), rng);
    for (size_t i = 0; i + 1 < global_stubs.size(); i += 2)
        edges.emplace_back(global_stubs[i], global_stubs[i + 1]);

    // 5) 去自环、统计度数
    std::vector<uint32_t> deg(n, 0);
    size_t kept = 0;
    for (auto& e : edges) {
        if (e.first == e.second) continue;
        ++deg[(size_t)e.first]; ++deg[(size_t)e.second]; ++kept;
    }

    g.off_.assign(n + 1, 0);
    size_t acc = 0;
    for (size_t i = 0; i < n; ++i) { g.off_[i] = (uint32_t)acc; acc += deg[i]; }
    g.off_[n] = (uint32_t)acc;
    g.adj_.resize(acc);
    std::vector<uint32_t> cur(g.off_.begin(), g.off_.end() - 1);
    for (auto& e : edges) {
        if (e.first == e.second) continue;
        g.adj_[cur[(size_t)e.first]++] = e.second;
        g.adj_[cur[(size_t)e.second]++] = e.first;
    }
    g.drop_isolated_ = false;
    double avg = n ? (double)acc / (double)n : 0.0;
    std::fprintf(stderr, "[csr] DCSBM mu=%.2f gamma=%.2f C=%d  N=%zu 平均度=%.1f 内存=%.1f MB (%.1f B/节点)\n",
                 mu, gamma, C, n, avg, (double)g.bytes() / 1048576.0, (double)g.bytes() / (double)n);
}
// ---- 带社区结构 + 幂律度数的社交图（真实社交网络的标准形态）----
// 结构：n 个节点分成 C 个社区，每个社区有 H 个"本地名人"（hub）。
//   每个节点的边：① 一条指向本社区随机 hub，② 一条指向本社区 top-3 hub（造成邻域重叠），
//                 ③ 其余指向本社区普通成员，④ 少量跨社区边。
// 效果：同一社区的 top hub 粉丝集合高度重叠 → "度数贪心"会重复买同一批人，
//       而"边际覆盖贪心"会被迫换社区，两者开始分道扬镳。
static void build_social_csr_clustered(SocialGraphCSR& g, size_t n, int e, int C = 200, int H = 30) {
    g.n_ = n;
    std::vector<uint32_t> deg(n, 0);
    const size_t S = std::max<size_t>(1, n / (size_t)C);
    // 社区极度不均衡：一半节点属于社区 0（模拟"头部社区/超级名人集中"），其余均分
    const size_t big = n / 2;
    const size_t S2 = std::max<size_t>(1, (n - big) / (size_t)std::max(1, C - 1));
    auto cid_of = [&](size_t i) { return (i < big) ? 0 : (int)(1 + (i - big) / S2); };
    auto comm_begin = [&](int c) { return (c == 0) ? (size_t)0 : (big + (size_t)(c - 1) * S2); };
    auto comm_size = [&](int c) {
        size_t b = comm_begin(c);
        if (b >= n) return (size_t)1;
        size_t s = (c == 0) ? big : S2;
        return std::min(s, n - b);
    };

    auto gen = [&](std::mt19937& rng, int i) -> int {
        int c = cid_of((size_t)i);
        size_t b = comm_begin(c), sz = comm_size(c);
        uint32_t r = rng();
        int k = (int)(r % (uint32_t)std::max(1, e));
        (void)k;
        // 用边序号决定类型：由调用方传 k
        return (int)(b + (r % (uint32_t)sz));
    };
    (void)gen;

    auto pass = [&](bool fill) {
        std::mt19937 rng(20261030u);
        for (int i = 0; i < (int)n; ++i) {
            int c = cid_of((size_t)i);
            size_t b = comm_begin(c), sz = comm_size(c);
            for (int k = 0; k < e; ++k) {
                int t;
                uint32_t r = rng();
                if (k == 0)      t = (int)(b + (r % (uint32_t)std::min<size_t>((size_t)H, sz)));
                else if (k == 1) t = (int)(b + (r % (uint32_t)std::min<size_t>(3, sz)));
                else if ((r % 100) < 10) t = (int)(rng() % (uint32_t)n);       // 跨社区
                else             t = (int)(b + (r % (uint32_t)sz));            // 本社区
                if (t == i) continue;
                if (!fill) { ++deg[(size_t)i]; ++deg[(size_t)t]; }
                else {
                    g.adj_[g.cur_[(size_t)i]++] = (uint32_t)t;
                    g.adj_[g.cur_[(size_t)t]++] = (uint32_t)i;
                }
            }
        }
    };

    pass(false);
    g.off_.assign(n + 1, 0);
    size_t acc = 0;
    for (size_t i = 0; i < n; ++i) { g.off_[i] = (uint32_t)acc; acc += deg[i]; }
    g.off_[n] = (uint32_t)acc;
    g.adj_.resize(acc);
    g.cur_.assign(g.off_.begin(), g.off_.end() - 1);
    pass(true);
    g.cur_.clear();
    std::fprintf(stderr, "[csr] 聚集图 C=%d H=%d  N=%zu 边条目=%zu  内存=%.1f MB (%.1f B/节点)\n",
                 C, H, n, acc, (double)g.bytes() / 1048576.0, (double)g.bytes() / (double)n);
}
// 与 mkG_social(SocialGraph&, n, e) 完全相同的边序列
static void build_social_csr(SocialGraphCSR& g, size_t n, int e) {
    g.n_ = n;
    std::vector<uint32_t> deg(n, 0);

    {   // 第 1 遍：只数度数
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> d(0, (int)n - 1);
        int ni = (int)std::max<size_t>(1, n / 100);
        for (int i = 0; i < (int)n; ++i)
            for (int k = 0; k < e; ++k) {
                int t = (k == 0) ? (int)(rng() % ni) : d(rng);
                if (i != t) { ++deg[(size_t)i]; ++deg[(size_t)t]; }
            }
    }

    g.off_.assign(n + 1, 0);
    size_t acc = 0;
    for (size_t i = 0; i < n; ++i) { g.off_[i] = (uint32_t)acc; acc += deg[i]; }
    g.off_[n] = (uint32_t)acc;
    if (acc > 0xFFFFFFFFull) {
        std::fprintf(stderr, "[csr] 边数 %zu 超过 uint32 上限，需要换 uint64 偏移\n", acc);
        std::exit(1);
    }
    g.adj_.resize(acc);
    std::vector<uint32_t> cur(g.off_.begin(), g.off_.end() - 1);   // 写游标

    {   // 第 2 遍：同一串随机数，真正填边（顺序与 push_back 版一致）
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> d(0, (int)n - 1);
        int ni = (int)std::max<size_t>(1, n / 100);
        for (int i = 0; i < (int)n; ++i)
            for (int k = 0; k < e; ++k) {
                int t = (k == 0) ? (int)(rng() % ni) : d(rng);
                if (i != t) {
                    g.adj_[cur[(size_t)i]++] = (uint32_t)t;
                    g.adj_[cur[(size_t)t]++] = (uint32_t)i;
                }
            }
    }
    std::fprintf(stderr, "[csr] N=%zu 边条目=%zu  内存=%.1f MB (%.1f B/节点)\n",
                 n, acc, (double)g.bytes() / 1048576.0, (double)g.bytes() / (double)n);
}
