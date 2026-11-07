#pragma once
// u8vec.hpp —— uint8 向量类型 + 与 float 口径逐位等价的 L2 距离（新文件）
//
// 为什么不用 vector_ops.hpp 里现成的 vec_L2sqr(const uint8_t*, ...)？
//   那三个分支（AVX512/AVX/SSE）里的 uint8 版本都用了 _mm256_extractf32x4_ps /
//   _mm512_extractf32x4_ps 这类 **AVX512VL 专用 intrinsic**，在只到 AVX2 的机器上
//   GCC 11.2 直接报 "inlining failed ... target specific option mismatch"，编译不过。
//   （float 版本用的是 _mm256_extractf128_ps，所以以前编 float 程序从没暴露这个问题。）
//
// 本实现：AVX2 下用 unpack+madd 做整数累加，最后一次性转 float。
//   SIFT 值域 0..255、维度 128 → 平方和最大 8,323,200 < 2^24，
//   float32 对它是精确表示；整数口径与 float 口径算出的是**同一个整数**，
//   因此转 float 后逐位相同（这正是 uint8 改造能"无损"的根本原因）。
//
// 关键用法：必须 #include 在 "retrieval_framework.hpp" **之后**。
//   放在后面时：HNSW 模板定义点的非限定查找只能看到全局的 float 重载（正常），
//   而实例化点由 ADL 补上 anns::graph 里本文件的 u8_t 重载。
//   放在前面则会「名字隐藏」掉全局 float 重载，HNSW<float> 全部编译失败。

#include <cstdint>
#include <cstddef>
#if defined(_MSC_VER)
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns {
namespace graph {

// 1 字节包装类型：仅为让 HNSW<vdim_t> 的距离调用走到本文件的整数重载
struct u8_t {
    uint8_t b;
};
static_assert(sizeof(u8_t) == 1, "u8_t 必须是 1 字节，否则指针步长错");

inline float vec_L2sqr(const u8_t* x, const u8_t* y, size_t d) {
    const uint8_t* a = reinterpret_cast<const uint8_t*>(x);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(y);
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    const __m256i z = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i lo = _mm256_sub_epi16(_mm256_unpacklo_epi8(va, z), _mm256_unpacklo_epi8(vb, z));
        __m256i hi = _mm256_sub_epi16(_mm256_unpackhi_epi8(va, z), _mm256_unpackhi_epi8(vb, z));
        lo = _mm256_madd_epi16(lo, lo);  // int16 对乘加 → int32，单对最大 2*65025
        hi = _mm256_madd_epi16(hi, hi);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(lo, hi));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t total = _mm_cvtsi128_si32(s);
    for (; i < d; ++i) { int t = (int)a[i] - (int)b[i]; total += t * t; }
    return (float)total;
#else
    int32_t total = 0;
    for (size_t i = 0; i < d; ++i) { int t = (int)a[i] - (int)b[i]; total += t * t; }
    return (float)total;
#endif
}

}  // namespace graph
}  // namespace anns
