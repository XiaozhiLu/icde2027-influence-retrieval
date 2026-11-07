#pragma once
// u8_io.hpp —— uint8 存储改造专用 IO / 校验工具（新文件，不改动任何既有文件）
//
// 设计要点
//  1) SIFT 原始值全是 0..255 的整数 → float32 存的是"整数值的浮点表示"，
//     L2 平方距离在 float32 里完全精确（最大 128*255^2 = 8,323,200 < 2^24），
//     因此整数口径与 float 口径必须逐位相同；本文件只负责"读/转/校验/摘要"。
//  2) 100M 时不能先整份读成 float 再转（51.6 GB），必须流式 读一块→转一块→写一块。
//  3) 所有校验失败一律 exit(1)，绝不允许"带着脏数据往下跑"。
//
// 支持 4 种文件布局（都能直接读，无需预先转换）：
//   .fvecs  每条 [dim:int32][dim×float32]            值须为 0..255 整数（会校验）
//   .bvecs  每条 [dim:int32][dim×uint8]              官方 ANN_SIFT1B 用这个
//   .u8bin  [npts:int32][dim:int32][npts×dim×uint8]  DiskANN/SPTAG 风格，HF 上的 SIFT100M 用这个
//   .bin/.fbin  同上，但元素宽度**按文件大小自动判别**（可能是 float32）
//   （.bin 名字有歧义：Nanvivi 的 base.bin 是 uint8、maknee 的 fbin 是 float32，靠文件大小推断）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

static inline long long u8io_file_size(FILE* f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (long long)std::ftell(f);
#endif
}
static inline void u8io_seek(FILE* f, long long off) {
#ifdef _WIN32
    _fseeki64(f, off, SEEK_SET);
#else
    std::fseek(f, (long)off, SEEK_SET);
#endif
}

// ---------------------------------------------------------------- float 读取（等价性测试用）
// max_n = 0 表示全读
static float* load_fvecs_f32(const std::string& fn, size_t max_n, size_t& num, size_t& dim) {
    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::cerr << "[u8_io] open fail: " << fn << std::endl; std::exit(1); }
    int32_t d = 0;
    if (std::fread(&d, 4, 1, f) != 1 || d <= 0) { std::cerr << "[u8_io] bad header: " << fn << std::endl; std::exit(1); }
    dim = (size_t)d;
    std::fseek(f, 0, SEEK_END);
    long long fs = u8io_file_size(f);
    const size_t st = 4 + 4 * dim;
    size_t total = (size_t)(fs / (long long)st);
    num = (max_n && max_n < total) ? max_n : total;
    std::vector<float> out(num * dim);

    const size_t CH = 16384;                       // 每批 ≈ 8.5 MB
    std::vector<char> buf(CH * st);
    u8io_seek(f, 0);   // 每条记录 = [dim:4][data:4*dim]，必须从 0 起按整条读
    size_t done = 0;
    while (done < num) {
        size_t take = std::min(CH, num - done);
        if (std::fread(buf.data(), 1, take * st, f) != take * st) {
            std::cerr << "[u8_io] short read at " << done << std::endl; std::exit(1);
        }
        for (size_t i = 0; i < take; ++i) {
            const char* p = buf.data() + i * st;
            int32_t dd; std::memcpy(&dd, p, 4);
            if (dd != d) { std::cerr << "[u8_io] dim mismatch@" << (done + i) << std::endl; std::exit(1); }
            std::memcpy(out.data() + (done + i) * dim, p + 4, dim * 4);
        }
        done += take;
    }
    std::fclose(f);
    float* res = new float[out.size()];
    std::memcpy(res, out.data(), out.size() * sizeof(float));
    return res;
}

// ---------------------------------------------------------------- 格式探测
enum u8io_fmt { U8IO_FVECS = 0, U8IO_BVECS = 1, U8IO_BIN = 2 };

static inline bool u8io_ends_with(const std::string& s, const char* suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static inline u8io_fmt u8io_fmt_of(const std::string& fn) {
    if (u8io_ends_with(fn, ".bvecs")) return U8IO_BVECS;
    if (u8io_ends_with(fn, ".fvecs")) return U8IO_FVECS;
    if (u8io_ends_with(fn, ".u8bin") || u8io_ends_with(fn, ".fbin") || u8io_ends_with(fn, ".bin")) return U8IO_BIN;
    return U8IO_FVECS;
}

struct U8ioProbe {
    size_t num = 0, dim = 0;
    size_t elem = 1;          // 元素字节宽度：1 = uint8，4 = float32
    bool contiguous = false;  // true: [npts][dim][连续数据]；false: 每条 [dim][data]
    long long data_off = 0;   // contiguous 的数据起点
    size_t rec_size = 0;      // 非 contiguous 的每条记录字节数
    long long file_size = 0;
    u8io_fmt fmt = U8IO_FVECS;
    const char* name() const {
        if (fmt == U8IO_BIN) return elem == 1 ? "u8bin" : "f32bin";
        return fmt == U8IO_BVECS ? "bvecs" : "fvecs";
    }
};

static U8ioProbe u8io_probe(const std::string& fn, u8io_fmt fmt) {
    U8ioProbe p;
    p.fmt = fmt;
    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::cerr << "[u8_io] open fail: " << fn << std::endl; std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    p.file_size = u8io_file_size(f);
    u8io_seek(f, 0);

    if (fmt == U8IO_BIN) {
        int32_t np = 0, dd = 0;
        if (std::fread(&np, 4, 1, f) != 1 || std::fread(&dd, 4, 1, f) != 1 || np <= 0 || dd <= 0) {
            std::cerr << "[u8_io] bin 头部非法: " << fn << std::endl; std::exit(1);
        }
        p.num = (size_t)np; p.dim = (size_t)dd;
        p.contiguous = true; p.data_off = 8;
        long long as_u8 = 8 + (long long)p.num * (long long)p.dim;
        long long as_f32 = 8 + 4LL * (long long)p.num * (long long)p.dim;
        if (p.file_size == as_u8) p.elem = 1;
        else if (p.file_size == as_f32) p.elem = 4;
        else {
            std::cerr << "[u8_io] bin 大小与头部不符: file=" << p.file_size
                      << " 期望 u8=" << as_u8 << " 或 f32=" << as_f32 << std::endl;
            std::exit(1);
        }
    } else {
        int32_t dd = 0;
        if (std::fread(&dd, 4, 1, f) != 1 || dd <= 0) { std::cerr << "[u8_io] bad header: " << fn << std::endl; std::exit(1); }
        p.dim = (size_t)dd;
        p.elem = (fmt == U8IO_FVECS) ? 4 : 1;
        p.rec_size = 4 + p.elem * p.dim;
        p.num = (size_t)(p.file_size / (long long)p.rec_size);
        p.contiguous = false;
    }
    std::fclose(f);
    if (p.num == 0) { std::cerr << "[u8_io] 空文件: " << fn << std::endl; std::exit(1); }
    return p;
}

// 把一批 float 值校验并转成 uint8
static inline void u8io_convert(const float* src, uint8_t* dst, size_t cnt, size_t base_idx,
                                size_t& bad, float& mn, float& mx) {
    for (size_t i = 0; i < cnt; ++i) {
        float v = src[i];
        int iv = (int)std::floor(v + 0.5f);
        if (v != (float)iv || iv < 0 || iv > 255) {
            if (bad < 5) std::cerr << "[u8_io] non-uint8 value " << v << " @" << (base_idx + i) << std::endl;
            ++bad;
        }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        dst[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
    }
}

// ---------------------------------------------------------------- 主读取（4 种布局统一入口）
static uint8_t* load_vecs_u8(const std::string& fn, size_t max_n, size_t& num, size_t& dim, u8io_fmt fmt,
                             size_t* n_nonint = nullptr, float* vmin = nullptr, float* vmax = nullptr) {
    U8ioProbe p = u8io_probe(fn, fmt);
    dim = p.dim;
    num = (max_n && max_n < p.num) ? max_n : p.num;

    uint8_t* out = new uint8_t[num * dim];
    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::cerr << "[u8_io] open fail: " << fn << std::endl; std::exit(1); }

    const size_t CH = 16384;
    size_t done = 0, bad = 0;
    float mn = 1e30f, mx = -1e30f;

    if (p.contiguous) {
        const size_t row_bytes = p.dim * p.elem;
        std::vector<char> buf(CH * row_bytes);
        std::vector<float> tmp(p.elem == 4 ? CH * p.dim : 0);
        u8io_seek(f, p.data_off);
        while (done < num) {
            size_t take = std::min(CH, num - done);
            if (std::fread(buf.data(), 1, take * row_bytes, f) != take * row_bytes) {
                std::cerr << "[u8_io] short read at " << done << std::endl; std::exit(1);
            }
            if (p.elem == 4) {
                std::memcpy(tmp.data(), buf.data(), take * row_bytes);
                u8io_convert(tmp.data(), out + done * dim, take * dim, done * dim, bad, mn, mx);
            } else {
                std::memcpy(out + done * dim, buf.data(), take * row_bytes);
            }
            done += take;
        }
    } else {
        std::vector<char> buf(CH * p.rec_size);
        std::vector<float> tmp(p.elem == 4 ? CH * p.dim : 0);
        u8io_seek(f, 0);
        while (done < num) {
            size_t take = std::min(CH, num - done);
            if (std::fread(buf.data(), 1, take * p.rec_size, f) != take * p.rec_size) {
                std::cerr << "[u8_io] short read at " << done << std::endl; std::exit(1);
            }
            if (p.elem == 4) {
                for (size_t i = 0; i < take; ++i) {
                    const char* q = buf.data() + i * p.rec_size;
                    int32_t dd; std::memcpy(&dd, q, 4);
                    if ((size_t)dd != p.dim) { std::cerr << "[u8_io] dim mismatch@" << (done + i) << std::endl; std::exit(1); }
                    std::memcpy(tmp.data() + i * p.dim, q + 4, p.dim * 4);
                }
                u8io_convert(tmp.data(), out + done * dim, take * dim, done * dim, bad, mn, mx);
            } else {
                for (size_t i = 0; i < take; ++i) {
                    const char* q = buf.data() + i * p.rec_size;
                    int32_t dd; std::memcpy(&dd, q, 4);
                    if ((size_t)dd != p.dim) { std::cerr << "[u8_io] dim mismatch@" << (done + i) << std::endl; std::exit(1); }
                    std::memcpy(out + (done + i) * dim, q + 4, p.dim);
                }
            }
            done += take;
        }
    }
    std::fclose(f);

    if (n_nonint) *n_nonint = bad;
    if (vmin) *vmin = mn;
    if (vmax) *vmax = mx;
    if (bad) { std::cerr << "[u8_io] FATAL: " << bad << " 个值不是 0..255 整数，uint8 不是无损的，停止" << std::endl; std::exit(1); }

    std::string shortname = fn;
    size_t sl = shortname.find_last_of("/\\");
    if (sl != std::string::npos) shortname = shortname.substr(sl + 1);
    if (p.elem == 4)
        std::printf("[u8_io] %-32s %-7s N=%zu dim=%zu 非整数=%zu 值域=[%.0f,%.0f]\n",
                    shortname.c_str(), p.name(), num, dim, bad, mn, mx);
    else
        std::printf("[u8_io] %-32s %-7s N=%zu dim=%zu (uint8 原生，无需转换)\n",
                    shortname.c_str(), p.name(), num, dim);
    return out;
}

// 兼容旧调用（一律按 .fvecs 解释）
static uint8_t* load_fvecs_u8(const std::string& fn, size_t max_n, size_t& num, size_t& dim,
                              size_t* n_nonint = nullptr, float* vmin = nullptr, float* vmax = nullptr) {
    return load_vecs_u8(fn, max_n, num, dim, U8IO_FVECS, n_nonint, vmin, vmax);
}

// 只读第 [from, to) 条记录 —— 续建时用，避免整份文件再读一遍
static void load_vecs_u8_range(const std::string& fn, u8io_fmt fmt, size_t from, size_t to, uint8_t* dest, size_t dim) {
    if (to <= from) return;
    U8ioProbe p = u8io_probe(fn, fmt);
    if (p.dim != dim) {
        std::cerr << "[u8_io] dim 不一致: 文件 " << p.dim << " vs 期望 " << dim << std::endl; std::exit(1);
    }
    if (to > p.num) { std::cerr << "[u8_io] range 越界: to=" << to << " > N=" << p.num << std::endl; std::exit(1); }

    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::cerr << "[u8_io] open fail: " << fn << std::endl; std::exit(1); }
    const size_t CH = 16384;
    size_t done = from;
    size_t bad = 0; float mn = 1e30f, mx = -1e30f;

    if (p.contiguous) {
        const size_t row_bytes = p.dim * p.elem;
        std::vector<char> buf(CH * row_bytes);
        std::vector<float> tmp(p.elem == 4 ? CH * p.dim : 0);
        u8io_seek(f, p.data_off + (long long)from * (long long)row_bytes);
        while (done < to) {
            size_t take = std::min(CH, to - done);
            if (std::fread(buf.data(), 1, take * row_bytes, f) != take * row_bytes) {
                std::cerr << "[u8_io] range short read at " << done << std::endl; std::exit(1);
            }
            uint8_t* out = dest + (done - from) * dim;
            if (p.elem == 4) {
                std::memcpy(tmp.data(), buf.data(), take * row_bytes);
                u8io_convert(tmp.data(), out, take * dim, done * dim, bad, mn, mx);
            } else {
                std::memcpy(out, buf.data(), take * row_bytes);
            }
            done += take;
        }
    } else {
        std::vector<char> buf(CH * p.rec_size);
        std::vector<float> tmp(p.elem == 4 ? CH * p.dim : 0);
        u8io_seek(f, (long long)from * (long long)p.rec_size);
        while (done < to) {
            size_t take = std::min(CH, to - done);
            if (std::fread(buf.data(), 1, take * p.rec_size, f) != take * p.rec_size) {
                std::cerr << "[u8_io] range short read at " << done << std::endl; std::exit(1);
            }
            uint8_t* out = dest + (done - from) * dim;
            for (size_t i = 0; i < take; ++i) {
                const char* q = buf.data() + i * p.rec_size;
                int32_t dd; std::memcpy(&dd, q, 4);
                if ((size_t)dd != p.dim) { std::cerr << "[u8_io] dim mismatch@" << (done + i) << std::endl; std::exit(1); }
                if (p.elem == 4) std::memcpy(tmp.data() + i * p.dim, q + 4, p.dim * 4);
                else std::memcpy(out + i * dim, q + 4, p.dim);
            }
            if (p.elem == 4) u8io_convert(tmp.data(), out, take * dim, done * dim, bad, mn, mx);
            done += take;
        }
    }
    std::fclose(f);
    if (bad) { std::cerr << "[u8_io] FATAL: range 里有 " << bad << " 个非 uint8 值" << std::endl; std::exit(1); }
}

// 兼容旧调用（一律按 .fvecs 解释）
static void load_fvecs_u8_range(const std::string& fn, size_t from, size_t to, uint8_t* dest, size_t dim) {
    load_vecs_u8_range(fn, U8IO_FVECS, from, to, dest, dim);
}

// float 缓冲 → uint8 缓冲（严格校验，用于同进程等价性测试）
static uint8_t* f32_to_u8(const float* src, size_t n, size_t* n_nonint = nullptr) {
    uint8_t* out = new uint8_t[n];
    size_t bad = 0; float mn = 1e30f, mx = -1e30f;
    u8io_convert(src, out, n, 0, bad, mn, mx);
    if (n_nonint) *n_nonint = bad;
    if (bad) { std::cerr << "[u8_io] FATAL: f32_to_u8 出现 " << bad << " 个非 uint8 值" << std::endl; std::exit(1); }
    return out;
}

// ---------------------------------------------------------------- 摘要 / 计时
static inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    for (int k = 0; k < 8; ++k) { h ^= (uint8_t)((v >> (8 * k)) & 0xFF); h *= 1099511628211ULL; }
    return h;
}
static inline uint64_t fnv1a_init() { return 1469598103934665603ULL; }
static inline uint32_t f32_bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

static inline double u8io_now_s() {
    return (double)std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
