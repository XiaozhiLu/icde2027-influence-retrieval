#pragma once
// idx_persist.hpp —— HNSW<u8_t> 索引落盘 / 加载（新文件，不改动任何既有文件）
//
// 关键点（本项目特有，容易踩）：
//   HNSW 的 data_level0_memory_ 里，每个元素在 offset_data_ 处存的是**一个 8 字节指针**，
//   指向调用方传进来的 base 数组（AddPoint 只存指针，不拷贝向量）。
//   所以落盘必须：
//     1) 把 base（N*D 的 uint8）一起存下来；
//     2) 加载后逐元素把指针重新指到新 base 上（否则就是跨进程野指针，必崩或读脏）。
//
// 文件格式（单文件，自带校验）：
//   [header 124B]  magic/version/D,N,M,Mmax,Mmax0,efC,seed,size_links_level0,
//                  size_data_per_element,offset_data,size_links_per_element,
//                  cur_count,max_level,enterpoint,payload_hash
//   [payload]
//     data_level0_memory_            : N * size_data_per_element_ 字节
//     link_lists_[id]                : uint32 len + len 字节（逐元素）
//     element_levels_                : N * int32
//     base                           : N * D 字节
//   payload_hash = FNV-1a(payload)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

#include "retrieval_framework.hpp"
#include "u8vec.hpp"      // 必须在 retrieval_framework.hpp 之后
#include "u8_io.hpp"      // fnv1a_u64
#if defined(__linux__)
#include <unistd.h>
#include <fcntl.h>
#endif

namespace idxio {

static inline uint64_t hash_bytes(uint64_t h, const void* p, size_t n);  // 前置声明（定义在本文件后面）
// ---- 页缓存控制（Linux）：cgroup 会把页缓存算进内存额度，读 28 GB 索引会把它吃满 ----
static inline void drop_file_cache(FILE* f, long long off, size_t len) {
#if defined(__linux__)
    int fd = fileno(f);
    if (fd >= 0) posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
#else
    (void)f; (void)off; (void)len;
#endif
}

// 分块读：每读一块就丢一块缓存，避免页缓存与匿名内存同时逼近 cgroup 上限
static inline void rd_chunked(FILE* f, void* p, size_t n, uint64_t& h) {
    const size_t CH = (size_t)256 << 20;
    uint8_t* q = (uint8_t*)p;
    size_t done = 0;
    while (done < n) {
        size_t take = std::min(CH, n - done);
        if (std::fread(q + done, 1, take, f) != take) {
            std::cerr << "[idxio] read fail / 文件被截断" << std::endl;
            std::exit(1);
        }
        h = hash_bytes(h, q + done, take);
        long long pos = (long long)std::ftell(f);
        drop_file_cache(f, pos - (long long)take, take);
        done += take;
    }
}

static const char MAGIC[8] = {'I', 'D', 'X', 'U', '8', '0', '0', '2'};
static const uint32_t VERSION = 2;

static inline uint64_t hash_bytes(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t v; std::memcpy(&v, b + i, 8); h = fnv1a_u64(h, v); }
    for (; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static inline void wr(FILE* f, const void* p, size_t n, uint64_t& h) {
    if (n == 0) return;
    if (std::fwrite(p, 1, n, f) != n) { std::cerr << "[idxio] write fail" << std::endl; std::exit(1); }
    h = hash_bytes(h, p, n);
}
static inline void rd(FILE* f, void* p, size_t n, uint64_t& h) {
    if (n == 0) return;
    if (std::fread(p, 1, n, f) != n) { std::cerr << "[idxio] read fail / 文件被截断" << std::endl; std::exit(1); }
    h = hash_bytes(h, p, n);
}

struct Header {
    char magic[8];
    uint32_t version;
    uint64_t D, N, M, Mmax, Mmax0, efC, seed;
    uint64_t size_links_level0, size_data_per_element, offset_data, size_links_per_element;
    uint64_t cur_count;
    int32_t max_level, enterpoint;
    uint64_t payload_hash;
};

// ---------------------------------------------------------------- 保存
static double save_index(const anns::graph::HNSW<anns::graph::u8_t>& idx,
                         const uint8_t* base, const std::string& path) {
    auto t0 = std::chrono::steady_clock::now();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "[idxio] open for write fail: " << path << std::endl; std::exit(1); }

    const size_t N = idx.cur_element_count_;
    Header h{};
    std::memcpy(h.magic, MAGIC, 8);
    h.version = VERSION;
    h.D = idx.D_; h.N = N; h.M = idx.M_; h.Mmax = idx.Mmax_; h.Mmax0 = idx.Mmax0_;
    h.efC = idx.ef_construction_; h.seed = (uint64_t)idx.random_seed_;
    h.size_links_level0 = idx.size_links_level0_;
    h.size_data_per_element = idx.size_data_per_element_;
    h.offset_data = idx.offset_data_;
    h.size_links_per_element = idx.size_links_per_element_;
    h.cur_count = N;
    h.max_level = idx.max_level_;
    h.enterpoint = (int32_t)idx.enterpoint_node_;
    h.payload_hash = 0;
    if (std::fwrite(&h, sizeof(h), 1, f) != 1) { std::cerr << "[idxio] header write fail" << std::endl; std::exit(1); }

    uint64_t ph = 1469598103934665603ULL;
    // 1) level0 区（含每元素 8B 的 data 指针，加载后会重绑，这里原样存）
    wr(f, idx.data_level0_memory_.data(), N * idx.size_data_per_element_, ph);
    // 2) 上层链
    for (size_t id = 0; id < N; ++id) {
        uint32_t len = (uint32_t)idx.link_lists_[id].size();
        wr(f, &len, sizeof(len), ph);
        if (len) wr(f, idx.link_lists_[id].data(), len, ph);
    }
    // 3) 层数
    wr(f, idx.element_levels_.data(), N * sizeof(int32_t), ph);
    // 4) 底库
    wr(f, base, N * idx.D_, ph);

    h.payload_hash = ph;
    std::fseek(f, 0, SEEK_SET);
    if (std::fwrite(&h, sizeof(h), 1, f) != 1) { std::cerr << "[idxio] header rewrite fail" << std::endl; std::exit(1); }
    std::fclose(f);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------- 加载
// 返回新建的索引；*base_out 由本函数分配
// capacity_rows > N 时按更大容量分配（续建用）：索引的 max_elements_ / level0 区 /
// link_lists_ / element_levels_ / base 全部按 capacity_rows 开，只读入前 N 行/项
template <typename IdxT>
static IdxT* load_index(const std::string& path, uint8_t** base_out,
                        bool verify, double* t_load = nullptr,
                        size_t capacity_rows = 0) {
    auto t0 = std::chrono::steady_clock::now();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "[idxio] open fail: " << path << std::endl; std::exit(1); }
    drop_file_cache(f, 0, 0);   // 先丢掉该文件的陈旧页缓存（cgroup 计入内存额度）
    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) { std::cerr << "[idxio] header read fail" << std::endl; std::exit(1); }
    if (std::memcmp(h.magic, MAGIC, 8) != 0) { std::cerr << "[idxio] magic mismatch（不是本程序写的索引）" << std::endl; std::exit(1); }
    if (h.version != VERSION) { std::cerr << "[idxio] version mismatch: " << h.version << std::endl; std::exit(1); }

    using anns::graph::HNSW;
    using anns::graph::u8_t;
    const size_t N = (size_t)h.N;
    const size_t D = (size_t)h.D;
    const size_t capN = (capacity_rows > N) ? capacity_rows : N;
    auto* idx = new IdxT(D, capN, (size_t)h.M, (size_t)h.efC, (size_t)h.seed);

    if (idx->size_links_level0_ != h.size_links_level0 || idx->size_data_per_element_ != h.size_data_per_element ||
        idx->offset_data_ != h.offset_data || idx->size_links_per_element_ != h.size_links_per_element) {
        std::cerr << "[idxio] 结构参数与当前代码不一致，拒绝加载" << std::endl; std::exit(1);
    }

    uint64_t ph = 1469598103934665603ULL;
    idx->data_level0_memory_.resize(capN * idx->size_data_per_element_);
    rd_chunked(f, idx->data_level0_memory_.data(), N * idx->size_data_per_element_, ph);

    size_t nz = 0;
    for (size_t id = 0; id < N; ++id) {
        uint32_t len = 0;
        rd(f, &len, sizeof(len), ph);
        if (len) { idx->link_lists_[id].resize(len); rd(f, idx->link_lists_[id].data(), len, ph); ++nz; }
    }
    idx->element_levels_.resize(N);
    rd(f, idx->element_levels_.data(), N * sizeof(int32_t), ph);

    uint8_t* base = new uint8_t[capN * D];
    rd_chunked(f, base, N * D, ph);
    std::fclose(f);

    // 重绑每个元素的 data 指针 ← 这一步是落盘能不能用的关键
    for (size_t id = 0; id < N; ++id) idx->WriteDataByInternalID((int)id, (const u8_t*)(base + id * D));

    idx->cur_element_count_ = N;
    idx->max_level_ = h.max_level;
    idx->enterpoint_node_ = h.enterpoint;
    idx->SetReady(true);

    if (verify && ph != h.payload_hash) {
        std::cerr << "[idxio] payload 校验失败: 期望 " << std::hex << h.payload_hash
                  << " 实得 " << ph << std::dec << std::endl;
        std::exit(1);
    }
    if (t_load) *t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    *base_out = base;
    return idx;
}

// 结构逐项对比（两个 HNSW<u8_t>）
struct Diff {
    size_t scalar = 0, levels = 0, l0_cnt = 0, l0_nb = 0, up_cnt = 0, up_nb = 0, ptr_bad = 0;
    long long total() const { return (long long)(scalar + levels + l0_cnt + l0_nb + up_cnt + up_nb + ptr_bad); }
};

static Diff compare_struct(const anns::graph::HNSW<anns::graph::u8_t>& a,
                           const anns::graph::HNSW<anns::graph::u8_t>& b, size_t N,
                           const uint8_t* base_a, const uint8_t* base_b) {
    using anns::graph::u8_t;
    Diff d;
    d.scalar = (size_t)(a.max_level_ != b.max_level_) + (size_t)(a.enterpoint_node_ != b.enterpoint_node_) +
               (size_t)(a.cur_element_count_ != b.cur_element_count_);
    for (size_t i = 0; i < N; ++i) {
        if (a.element_levels_[i] != b.element_levels_[i]) ++d.levels;
        if (a.GetDataByInternalID((int)i) != (const u8_t*)(base_a + i * a.D_)) ++d.ptr_bad;
        if (b.GetDataByInternalID((int)i) != (const u8_t*)(base_b + i * b.D_)) ++d.ptr_bad;
        const size_t* la = (const size_t*)a.GetLinkByInternalID((int)i, 0);
        const size_t* lb = (const size_t*)b.GetLinkByInternalID((int)i, 0);
        if (la[0] != lb[0]) { ++d.l0_cnt; }
        else {
            const int* na = (const int*)(la + 1);
            const int* nb = (const int*)(lb + 1);
            for (size_t k = 0; k < la[0]; ++k) if (na[k] != nb[k]) ++d.l0_nb;
        }
        int lv = a.element_levels_[i];
        for (int lev = 1; lev <= lv; ++lev) {
            const size_t* ua = (const size_t*)a.GetLinkByInternalID((int)i, lev);
            const size_t* ub = (const size_t*)b.GetLinkByInternalID((int)i, lev);
            if (ua[0] != ub[0]) { ++d.up_cnt; }
            else {
                const int* ma = (const int*)(ua + 1);
                const int* mb = (const int*)(ub + 1);
                for (size_t k = 0; k < ua[0]; ++k) if (ma[k] != mb[k]) ++d.up_nb;
            }
        }
    }
    return d;
}

}  // namespace idxio
