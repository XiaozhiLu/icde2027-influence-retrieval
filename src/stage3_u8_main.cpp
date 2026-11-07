// stage3_u8_main.cpp —— uint8 存储版 stage3 生产入口（新文件；逻辑与 stage3_main.cpp 逐行等价）
//
// 用途：10M / 100M 规模实验用。
// 与 stage3_main.cpp 的差异**只有存储与读入**：
//   1) base / query 以 uint8 流式读入（不再常驻 51.2 GB 的 float 底库），支持 fvecs/bvecs/u8bin
//   2) HNSW<u8_t>，距离走 u8vec.hpp 的整数重载（与 float 口径逐位等价）
//   3) --save=/--load=/--resume=/--save-at=：索引落盘、断点续建、一条轨迹出多档
//   4) --fast：改走 HNSWFast（visited 复用 + 预取），语义等价，结果必须逐位相同
//   5) --graph=csr|vec：社交图存储（默认 CSR，84 B/节点）
//
// 用法：
//   stage3u8.exe <N> [nq] [--ef=300] [--nogt] [--fast] [--graph=csr|vec]
//                [--save=data/x.idx] [--save-at=1000000,10000000]
//                [--load=data/x.idx] [--resume=data/x.idx]
//                [--base=path] [--query=path] [--fmt=fvecs|bvecs|bin] [--noverify]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "retrieval_framework.hpp"
#include "stage3_core.hpp"
#include "idx_persist.hpp"
#include "hnsw_fast.hpp"     // 必须在 retrieval_framework.hpp / u8vec.hpp 之后
#include "stage2_strategy.hpp" // 论文最终算法（proxy + 重叠下限 + 受许可替换）
#include "baselines.hpp"       // 外部对照基线：CELF-Near / CELF-Global / Greedy-Replace
#include "stage2_full.hpp"     // 完整方法：混合分检索 + 1v1 + 2v2 + slack 三档

using namespace std;
using namespace anns::graph;

static vector<size_t> parse_list(const string& s) {
    vector<size_t> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        string tok = s.substr(i, (j == string::npos ? s.size() : j) - i);
        if (!tok.empty()) v.push_back((size_t)atoll(tok.c_str()));
        if (j == string::npos) break;
        i = j + 1;
    }
    std::sort(v.begin(), v.end());
    return v;
}

static long long file_mb(const string& p) {
    FILE* fp = fopen(p.c_str(), "rb");
    if (!fp) return -1;
    std::fseek(fp, 0, SEEK_END);
    long long fs = u8io_file_size(fp);
    fclose(fp);
    return (long long)((double)fs / 1048576.0);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);          // 无缓冲：进程被 OOM 杀掉也不丢日志
    {
        FILE* mf = fopen("/sys/fs/cgroup/memory.current", "rb");
        if (mf) {
            unsigned long long v = 0;
            if (fscanf(mf, "%llu", &v) == 1) printf("[mem] cgroup current = %.1f GB\n", v / 1073741824.0);
            fclose(mf);
        }
    }
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : 1000000;
    int nq = (argc > 2) ? atoi(argv[2]) : 100;
    bool do_gt = true, verify = true;
    bool use_csr = true;   // 社交图默认用 CSR（省 7.4 GiB）；--graph=vec 走框架自带的 vector<vector<int>>
    bool use_fast = false; // --fast：HNSWFast（visited 复用 + 预取）
    bool do_baseline = false;   // --baseline / --baseline-only：打印「纯 HNSW top-K」基线行
    bool baseline_only = false; // 只跑基线行（测干净延迟）
    bool tau_only = false;      // 只跑 τ 行
    bool gt_only = false;       // 只算 GT 并存缓存后退出
    int warmup = 0;             // 评测前先跑 n 次查询热身，消除冷缓存
    bool use_s3 = false;        // --method=s3：走论文最终算法
    string method = "ours";     // ours | s3 | celf-near | celf-global | greedy-rep
    int theta_cons = 3;         // 保守档的增益阈值
    int g_deg = 10;             // 社交图平均度（每个节点出边数）
    string g_type = "random";   // random | cluster | dcsbm
    string g_file = "";         // --graphfile=边表文件（覆盖 g_type）
    double g_mu = 0.3;          // DCSBM 的跨社区边比例
    double g_gamma = 2.5;       // DCSBM 的度分布幂律指数
    bool use_2v2 = true;        // 是否启用 2-to-2 兜底替换
    bool use_hybrid = false;    // 是否启用混合分引导检索（--hybrid=1 开启）
    int lfetch = 30;            // --lfetch=N：候选池大小
    int elig = 1;               // --elig=degree|farthest：替换资格规则（默认 farthest，保护真实召回）
    vector<size_t> ef_list;     // --ef=300,500,800 可一次跑多个 ef
    string gt_path;             // --gt=path：GT 缓存
    string save_prefix, load_prefix, resume_path, saveat;
    string base_path = "data/sift-128-euclidean.train.fvecs";
    string query_path = "data/sift-128-euclidean.test.fvecs";
    string fmt_opt;   // --fmt=bvecs 可强制按 bvecs 解释（按扩展名也自动识别）
    int K = 10;                 // --K= 可覆盖（默认 10）
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--nogt") do_gt = false;
        else if (a == "--noverify") verify = false;
        else if (a == "--fast") use_fast = true;
        else if (a == "--baseline") do_baseline = true;
        else if (a == "--graph=vec") use_csr = false;
        else if (a == "--graph=csr") use_csr = true;
        else if (a.rfind("--ef=", 0) == 0) ef_list = parse_list(a.substr(5));
        else if (a.rfind("--gt=", 0) == 0) gt_path = a.substr(5);
        else if (a == "--gt-only") gt_only = true;
        else if (a == "--baseline-only") { baseline_only = true; do_baseline = true; }
        else if (a == "--tau-only") { tau_only = true; do_baseline = false; }
        else if (a.rfind("--warmup=", 0) == 0) warmup = atoi(a.c_str() + 9);
        else if (a.rfind("--method=", 0) == 0) { method = a.substr(9); if (method == "s3") use_s3 = true; }
        else if (a.rfind("--theta=", 0) == 0) theta_cons = atoi(a.c_str() + 8);
        else if (a == "--no2v2") use_2v2 = false;
        else if (a.rfind("--K=", 0) == 0) K = atoi(a.c_str() + 4);
        else if (a == "--hybrid=1") use_hybrid = true;
        else if (a.rfind("--graphtype=", 0) == 0) g_type = a.substr(12);
        else if (a.rfind("--graphfile=", 0) == 0) g_file = a.substr(12);
        else if (a.rfind("--mu=", 0) == 0) g_mu = atof(a.c_str() + 5);
        else if (a.rfind("--gamma=", 0) == 0) g_gamma = atof(a.c_str() + 8);
        else if (a.rfind("--deg=", 0) == 0) g_deg = atoi(a.c_str() + 6);
        else if (a.rfind("--lfetch=", 0) == 0) lfetch = atoi(a.c_str() + 9);
        else if (a == "--elig=degree") elig = 0;
        else if (a == "--elig=farthest") elig = 1;
        else if (a.rfind("--save=", 0) == 0) save_prefix = a.substr(7);
        else if (a.rfind("--load=", 0) == 0) load_prefix = a.substr(7);
        else if (a.rfind("--resume=", 0) == 0) resume_path = a.substr(9);
        else if (a.rfind("--save-at=", 0) == 0) saveat = a.substr(10);
        else if (a.rfind("--base=", 0) == 0) base_path = a.substr(7);
        else if (a.rfind("--query=", 0) == 0) query_path = a.substr(8);
        else if (a.rfind("--fmt=", 0) == 0) fmt_opt = a.substr(6);
    }
    auto pick_fmt = [](const std::string& path, const std::string& opt) -> u8io_fmt {
        if (opt.empty()) return u8io_fmt_of(path);   // 按扩展名：.bvecs/.fvecs/.u8bin/.fbin/.bin
        if (opt == "bvecs") return U8IO_BVECS;
        if (opt == "fvecs") return U8IO_FVECS;
        return U8IO_BIN;                             // bin / u8bin / fbin
    };
    const u8io_fmt base_fmt = pick_fmt(base_path, fmt_opt);
    const u8io_fmt query_fmt = pick_fmt(query_path, fmt_opt);
    const int NC = 3000;
    if (ef_list.empty()) ef_list.push_back(300);
    vector<size_t> save_at = parse_list(saveat);
    size_t si = 0;

    size_t nb = 0, db = 0, nqq = 0, dq = 0;
    size_t nonint = 0; float vmn = 0, vmx = 0;
    uint8_t* base_raw = nullptr;
    unique_ptr<HNSW<u8_t>> holder;               // 持有所有权（基类指针，可接管派生对象）
    HNSW<u8_t>* idx = nullptr;                   // 通用接口指针
    fastidx::HNSWFast<u8_t>* fast_idx = nullptr; // 建图专用（AddPoint 非虚，必须走派生指针）
    vector<int> ext;

    double t0 = u8io_now_s();
    uint8_t* qs_raw = load_vecs_u8(query_path, (size_t)nq, nqq, dq, query_fmt, &nonint, &vmn, &vmx);
    const u8_t* qs = (const u8_t*)qs_raw;

    auto add_point = [&](const u8_t* p) -> int {
        return fast_idx ? (int)fast_idx->AddPoint(p) : (int)idx->AddPoint(p);
    };
    auto do_save = [&](size_t cnt, const string& path) {
        double ts = idxio::save_index(*idx, base_raw, path);
        printf("[ckpt] %zu 点 -> %s  %lld MB  %.2f s\n", cnt, path.c_str(), file_mb(path), ts);
        fflush(stdout);
    };
    auto tick_ckpt = [&](size_t cnt) {
        while (si < save_at.size() && save_at[si] == cnt) {
            string base = save_prefix.empty() ? string("data/idx") : save_prefix;
            do_save(cnt, base + ".ckpt" + to_string(cnt));
            ++si;
        }
    };

    if (!load_prefix.empty() || !resume_path.empty()) {
        const string p = !resume_path.empty() ? resume_path : load_prefix;
        // 续建时 base 要能装下 N 行（loader 只填已有前缀那部分）
        size_t cap = (resume_path.empty()) ? 0 : N;
        uint8_t* b = nullptr;
        double tl = 0;
        if (use_fast) {
            holder.reset(idxio::load_index<fastidx::HNSWFast<u8_t>>(p, &b, verify, &tl, cap));
            fast_idx = (fastidx::HNSWFast<u8_t>*)holder.get();
        } else {
            holder.reset(idxio::load_index<HNSW<u8_t>>(p, &b, verify, &tl, cap));
        }
        idx = holder.get();
        base_raw = b;
        db = idx->D_;
        size_t N0 = idx->cur_element_count_;
        ext.assign(std::max(N, N0), -1);
        for (size_t i = 0; i < N0; ++i) ext[i] = (int)i;
        printf("[load-index] %s  %.2f s  N0=%zu dim=%zu max_level=%d\n", p.c_str(), tl, N0, db, idx->max_level_);
        fflush(stdout);

        if (!resume_path.empty() && N > N0) {
            // 1) 补齐后段向量（只读文件的 [N0, N)，不会重读前面）
            load_vecs_u8_range(base_path, base_fmt, N0, N, base_raw + N0 * db, db);
            // 2) 重放随机层抽签：AddPoint 每个点恰好抽一次（retrieval_framework.hpp:181,1972）
            for (size_t i = 0; i < N0; ++i) idx->GetRandomLevel(idx->mult_);
            // 3) 继续建图
            double tb = u8io_now_s();
            for (size_t i = N0; i < N; ++i) {
                int c = add_point((const u8_t*)(base_raw + i * db));
                ext[(size_t)c] = (int)i;
                tick_ckpt(i + 1);
            }
            printf("[build] 续建 %zu -> %zu: %.1f s  max_level=%d ep=%d  mode=%s\n", N0, N, u8io_now_s() - tb,
                   idx->max_level_, (int)idx->enterpoint_node_, use_fast ? "fast" : "base");
            N0 = N;
        } else if (!resume_path.empty() && N < N0) {
            printf("[warn] 目标 N=%zu 小于断点 N0=%zu，按断点评测\n", N, N0);
        }
        N = N0;
        nb = N0;
        if (!save_prefix.empty()) do_save(N, save_prefix);
        fflush(stdout);
    } else {
        base_raw = load_vecs_u8(base_path, N, nb, db, base_fmt, &nonint, &vmn, &vmx);
        printf("[load] base=%zu dim=%zu q=%zu  %.1f s\n", nb, db, nqq, u8io_now_s() - t0);
        fflush(stdout);
        if (use_fast) {
            holder.reset(new fastidx::HNSWFast<u8_t>(db, N, 16, 200));
            fast_idx = (fastidx::HNSWFast<u8_t>*)holder.get();
        } else {
            holder.reset(new HNSW<u8_t>(db, N, 16, 200));
        }
        idx = holder.get();
        idx->SetAlphaParams(0.0f, false, 200000.0f, 300.0f);
        ext.assign(N, -1);
        double tb = u8io_now_s();
        for (size_t i = 0; i < N; ++i) {
            int c = add_point((const u8_t*)(base_raw + i * db));
            ext[(size_t)c] = (int)i;
            tick_ckpt(i + 1);
        }
        idx->SetReady(true);
        printf("[build] %.1f s  max_level=%d ep=%d  mode=%s\n", u8io_now_s() - tb, idx->max_level_,
               (int)idx->enterpoint_node_, use_fast ? "fast" : "base");
        fflush(stdout);
        if (!save_prefix.empty()) do_save(N, save_prefix);
    }

    const u8_t* base = (const u8_t*)base_raw;
    if (dq && dq != db) { printf("[FATAL] query dim=%zu 与 base dim=%zu 不一致\n", dq, db); return 1; }

    // 内存账（在建社交图之前打印：base + index 是必须常驻的部分）
    double base_mb = (double)nb * db / 1048576.0;
    double idx_mb = (double)idx->IndexSize() / 1048576.0;
    printf("[mem] base=%.1f MB index=%.1f MB sum=%.1f MB\n", base_mb, idx_mb, base_mb + idx_mb);
    double sc = 100000000.0 / (double)nb;
    printf("[est] 100M 同口径: base=%.1f GB index=%.1f GB\n",
           100000000.0 * db / 1073741824.0, idx_mb / 1024.0 * sc);
    fflush(stdout);

    // GT：优先读缓存，否则算一次并写盘（ef 扫描时省掉每次 12 分钟的暴力 GT）
    vector<vector<int>> gt;
    bool gt_from_cache = false;
    if (!gt_path.empty()) gt_from_cache = load_gt_cache(gt_path, nq, K, gt);
    if (gt_from_cache) {
        printf("[gt] 从缓存加载 %s（%d x %d）\n", gt_path.c_str(), nq, K);
    } else if (do_gt) {
        double tg = u8io_now_s();
        compute_gt<u8_t>(base, nb, db, qs, nq, K, gt, 0);
        printf("[gt] %d x %zu 暴力 uint8 GT: %.1f s\n", nq, nb, u8io_now_s() - tg);
        if (!gt_path.empty() && save_gt_cache(gt_path, gt, K))
            printf("[gt] 已写入缓存 %s\n", gt_path.c_str());
    } else {
        gt.assign((size_t)nq, vector<int>(K, -1));
    }
    fflush(stdout);
    if (gt_only) { printf("[gt-only] 结束\n"); return 0; }

    auto do_warmup = [&]() {
        if (warmup <= 0) return;
        for (int w = 0; w < warmup; ++w)
            idx->Search(qs + (size_t)(w % nq) * db, K, ef_list[0]);
        printf("[warmup] 已跑 %d 次查询（不计入统计）\n", warmup);
        fflush(stdout);
    };

    if (use_csr) {
        SocialGraphCSR g;
        double tg = u8io_now_s();
        if (!g_file.empty())         build_social_csr_from_file(g, N, g_file);
        else if (g_type == "cluster") build_social_csr_clustered(g, N, g_deg);
        else if (g_type == "dcsbm")  build_social_csr_dcsbm(g, N, g_mu, g_gamma, 5, 2000, 200);
        else                         build_social_csr(g, N, g_deg);
        printf("[graph] CSR 构建 %.1f s  内存 %.1f MB\n", u8io_now_s() - tg, (double)g.bytes() / 1048576.0);
        fflush(stdout);
        do_warmup();
        if (method == "ours" && !baseline_only) {
            auto bydeg3 = build_bydeg(g, N, NC);
            for (float tau : {0.7f, 0.8f, 0.9f}) {
                auto r = run_full<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, tau, gt, g, N,
                                                                    &bydeg3, NC, lfetch, ef_list[0], elig, use_hybrid, theta_cons, use_2v2);
                printf("ours tau=%.1f ef=%zu lf=%d recall=%.5f inf=%.2f %.4f ms 1v1=%.2f 2v2=%.2f\n",
                       (double)tau, ef_list[0], lfetch, r.recall, r.inf, r.ms, r.swaps1, r.swaps2);
                fflush(stdout);
            }
        } else if (use_s3) {
            auto bydeg3 = build_bydeg(g, N, NC);
            for (float tau : {0.7f, 0.8f, 0.9f}) {
                auto r = run_strategy3<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, tau, gt, g, lfetch, ef_list[0], &bydeg3, NC, N, elig);
                printf("s3 tau=%.1f ef=%zu lf=%d recall=%.5f inf=%.2f %.4f ms swaps=%.2f\n",
                       (double)tau, ef_list[0], lfetch, r.recall, r.inf, r.ms, r.swaps);
                fflush(stdout);
            }
        } else if (method == "stage1") {
            auto r = run_stage1_only<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, gt, g, N, ef_list[0]);
            printf("stage1-only recall=%.5f inf=%.2f %.4f ms\n", r.recall, r.inf, r.ms);
        } else if (method == "celf-near") {
            auto r = run_celf_near<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, gt, g, N);
            printf("celf-near recall=%.5f inf=%.2f %.4f ms\n", r.recall, r.inf, r.ms);
        } else if (method == "celf-global" || method == "greedy-rep") {
            auto bydeg3 = build_bydeg(g, N, NC);
            if (method == "celf-global") {
                auto r = run_celf_global<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, gt, g, N, bydeg3, NC);
                printf("celf-global recall=%.5f inf=%.2f %.4f ms\n", r.recall, r.inf, r.ms);
                fflush(stdout);
            } else {
                for (float tau : {0.7f, 0.8f, 0.9f}) {
                    auto r = run_greedy_replace<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, tau, gt, g, N, bydeg3, NC, ef_list[0]);
                    printf("greedy-rep tau=%.1f recall=%.5f inf=%.2f %.4f ms\n", (double)tau, r.recall, r.inf, r.ms);
                    fflush(stdout);
                }
            }
        } else
        run_all_taus<HNSW<u8_t>, u8_t, SocialGraphCSR>(*idx, qs, db, nq, K, ef_list, ext, gt, g, N, NC, do_baseline, !tau_only && !baseline_only);
    } else {
        SocialGraph g(N);
        double tg = u8io_now_s();
        mkG_social(g, N, g_deg);
        printf("[graph] vector<vector<int>> 构建 %.1f s\n", u8io_now_s() - tg);
        fflush(stdout);
        do_warmup();
        if (method == "ours" && !baseline_only) {
            auto bydeg3 = build_bydeg(g, N, NC);
            for (float tau : {0.7f, 0.8f, 0.9f}) {
                auto r = run_full<HNSW<u8_t>, u8_t, SocialGraph>(*idx, qs, db, nq, K, tau, gt, g, N,
                                                                    &bydeg3, NC, lfetch, ef_list[0], elig, use_hybrid, theta_cons, use_2v2);
                printf("ours tau=%.1f ef=%zu lf=%d recall=%.5f inf=%.2f %.4f ms 1v1=%.2f 2v2=%.2f\n",
                       (double)tau, ef_list[0], lfetch, r.recall, r.inf, r.ms, r.swaps1, r.swaps2);
                fflush(stdout);
            }
        } else if (use_s3) {
            auto bydeg3 = build_bydeg(g, N, NC);
            for (float tau : {0.7f, 0.8f, 0.9f}) {
                auto r = run_strategy3<HNSW<u8_t>, u8_t, SocialGraph>(*idx, qs, db, nq, K, tau, gt, g, lfetch, ef_list[0], &bydeg3, NC, N, elig);
                printf("s3 tau=%.1f ef=%zu lf=%d recall=%.5f inf=%.2f %.4f ms swaps=%.2f\n",
                       (double)tau, ef_list[0], lfetch, r.recall, r.inf, r.ms, r.swaps);
                fflush(stdout);
            }
        } else
        run_all_taus<HNSW<u8_t>, u8_t, SocialGraph>(*idx, qs, db, nq, K, ef_list, ext, gt, g, N, NC, do_baseline, !tau_only && !baseline_only);
    }
    return 0;
}
