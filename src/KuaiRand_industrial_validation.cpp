#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cstdint>

#include "retrieval_framework.hpp"

using namespace std;
using namespace anns::graph;


float* load_embedding_bin(const string& filename, size_t& num, size_t& dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    int32_t n = 0, d = 0;
    in.read((char*)&n, sizeof(int32_t));
    in.read((char*)&d, sizeof(int32_t));

    if (!in || n <= 0 || d <= 0) {
        cerr << "Error: Invalid embedding header: " << filename << endl;
        exit(1);
    }

    num = (size_t)n;
    dim = (size_t)d;

    float* data = new float[num * dim];
    in.read((char*)data, sizeof(float) * num * dim);
    if (!in) {
        cerr << "Error: Failed to read embedding data: " << filename << endl;
        delete[] data;
        exit(1);
    }
    return data;
}

pair<double,double> LoadCalibration(const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    double a, b;
    in >> a >> b;
    return {a, b};
}

double LoadConfigValue(const string& filename, const string& key) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    string k;
    double v;
    while (in >> k >> v) {
        if (k == key) return v;
    }

    cerr << "Error: key not found in config: " << key << endl;
    exit(1);
}

void LoadGraphEdges(SocialGraph& sg, const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    int a, b;
    while (in >> a >> b) {
        if (a != b) sg.add_edge(a, b);
    }
}

struct EvalRow {
    int user;
    int item;
    int click;
};

vector<EvalRow> LoadEvalImpressions(const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }

    vector<EvalRow> rows;
    while (true) {
        EvalRow r;
        if (!(in >> r.user >> r.item >> r.click)) break;
        rows.push_back(r);
    }
    return rows;
}


inline double sigmoid(double x) {
    if (x >= 0) {
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = exp(x);
        return z / (1.0 + z);
    }
}

double dot_vecs(const float* a, const float* b, size_t dim) {
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) s += (double)a[i] * (double)b[i];
    return s;
}

double l2norm(const float* a, size_t dim) {
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) s += (double)a[i] * (double)a[i];
    return sqrt(s) + 1e-12;
}

// mode1: separate weighted sum
double score_mode1(int user_id, int item_id,
                   const float* user_emb,
                   const float* item_emb,
                   const float* item_useragg_emb,
                   size_t dim,
                   double alpha) {
    const float* u = user_emb + (size_t)user_id * dim;
    const float* e = item_emb + (size_t)item_id * dim;
    const float* g = item_useragg_emb + (size_t)item_id * dim;

    double s_item = dot_vecs(u, e, dim);
    double s_useragg = dot_vecs(u, g, dim);

    return alpha * s_item + (1.0 - alpha) * s_useragg;
}

// mode2: fused normalized item embedding
double score_mode2(int user_id, int item_id,
                   const float* user_emb,
                   const float* fused_item_emb_mode2,
                   size_t dim) {
    const float* u = user_emb + (size_t)user_id * dim;
    const float* f = fused_item_emb_mode2 + (size_t)item_id * dim;

    double nu = l2norm(u, dim);
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) s += ((double)u[i] / nu) * (double)f[i];
    return s;
}

unordered_set<int> ToSet(const vector<int>& ids) {
    return unordered_set<int>(ids.begin(), ids.end());
}

vector<int> MergeItems(const vector<int>& a, const vector<int>& b) {
    vector<int> out = a;
    unordered_set<int> seen(a.begin(), a.end());
    for (int x : b) {
        if (seen.insert(x).second) out.push_back(x);
    }
    return out;
}


double ComputeAUC(const vector<double>& scores, const vector<int>& labels) {
    int n = (int)scores.size();
    vector<pair<double,int>> v(n);
    for (int i = 0; i < n; ++i) v[i] = {scores[i], labels[i]};
    sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    long long pos = 0, neg = 0;
    for (int y : labels) {
        if (y == 1) pos++;
        else neg++;
    }
    if (pos == 0 || neg == 0) return 0.5;

    double rank_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (v[i].second == 1) rank_sum += (i + 1);
    }
    return (rank_sum - (double)pos * (pos + 1) / 2.0) / ((double)pos * neg);
}

double ComputeLogloss(const vector<double>& probs, const vector<int>& labels) {
    const double eps = 1e-12;
    double loss = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        double p = min(max(probs[i], eps), 1.0 - eps);
        int y = labels[i];
        loss += -(y * log(p) + (1 - y) * log(1.0 - p));
    }
    return loss / probs.size();
}

double ComputeCTRatK(const unordered_map<int, vector<pair<double,int>>>& user_scores, int K) {
    double total_ctr = 0.0;
    int valid_users = 0;

    for (const auto& kv : user_scores) {
        auto arr = kv.second;
        if (arr.empty()) continue;

        sort(arr.begin(), arr.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        int cutoff = min(K, (int)arr.size());
        int clicks = 0;
        for (int i = 0; i < cutoff; ++i) clicks += arr[i].second;

        total_ctr += (double)clicks / cutoff;
        valid_users++;
    }

    if (valid_users == 0) return 0.0;
    return total_ctr / valid_users;
}


struct EvalResult {
    double ctr10_base = 0.0, ctr10_ours = 0.0, ctr10_merge = 0.0;
    double auc_base = 0.0, auc_ours = 0.0, auc_merge = 0.0;
    double logloss_base = 0.0, logloss_ours = 0.0, logloss_merge = 0.0;
    double time_ms_per_user = 0.0;
};

EvalResult EvaluateOneMode(
    HNSW<float>& index,
    const vector<EvalRow>& eval_rows,
    const vector<int>& eval_users,
    const float* user_emb,
    const float* item_emb,
    const float* item_useragg_emb,
    const float* fused_item_emb_mode2,
    size_t dim,
    int K_items,
    int ef_search,
    int topK_eval,
    float epsilon,
    float tau,
    int score_mode,
    double alpha_mode1,
    double a_calib,
    double b_calib
) {
    auto t1 = chrono::high_resolution_clock::now();

    unordered_map<int, unordered_set<int>> cand_base, cand_ours, cand_merge;

    for (int u : eval_users) {
        const float* q_ptr = user_emb + (size_t)u * dim;

        vector<int> base_items = index.SearchBaseline(q_ptr, K_items, ef_search);
        vector<int> ours_items = index.SearchIterative_Strategy_final(q_ptr, K_items, epsilon, tau);
        vector<int> merge_items = MergeItems(base_items, ours_items);

        cand_base[u] = ToSet(base_items);
        cand_ours[u] = ToSet(ours_items);
        cand_merge[u] = ToSet(merge_items);
    }

    vector<double> probs_base, probs_ours, probs_merge;
    vector<double> scores_base, scores_ours, scores_merge;
    vector<int> labels;
    unordered_map<int, vector<pair<double,int>>> user_scores_base, user_scores_ours, user_scores_merge;

    probs_base.reserve(eval_rows.size());
    probs_ours.reserve(eval_rows.size());
    probs_merge.reserve(eval_rows.size());
    scores_base.reserve(eval_rows.size());
    scores_ours.reserve(eval_rows.size());
    scores_merge.reserve(eval_rows.size());
    labels.reserve(eval_rows.size());

    for (const auto& r : eval_rows) {
        int u = r.user;
        int it = r.item;
        int y = r.click;

        labels.push_back(y);

        double raw = 0.0;
        if (score_mode == 1) {
            raw = score_mode1(u, it, user_emb, item_emb, item_useragg_emb, dim, alpha_mode1);
        } else {
            raw = score_mode2(u, it, user_emb, fused_item_emb_mode2, dim);
        }

        double p_base = 1e-6, p_ours = 1e-6, p_merge = 1e-6;

        if (cand_base[u].count(it)) p_base = sigmoid(a_calib * raw + b_calib);
        if (cand_ours[u].count(it)) p_ours = sigmoid(a_calib * raw + b_calib);
        if (cand_merge[u].count(it)) p_merge = sigmoid(a_calib * raw + b_calib);

        probs_base.push_back(p_base);
        probs_ours.push_back(p_ours);
        probs_merge.push_back(p_merge);

        scores_base.push_back(p_base);
        scores_ours.push_back(p_ours);
        scores_merge.push_back(p_merge);

        user_scores_base[u].push_back({p_base, y});
        user_scores_ours[u].push_back({p_ours, y});
        user_scores_merge[u].push_back({p_merge, y});
    }

    auto t2 = chrono::high_resolution_clock::now();

    EvalResult res;
    res.auc_base = ComputeAUC(scores_base, labels);
    res.auc_ours = ComputeAUC(scores_ours, labels);
    res.auc_merge = ComputeAUC(scores_merge, labels);

    res.logloss_base = ComputeLogloss(probs_base, labels);
    res.logloss_ours = ComputeLogloss(probs_ours, labels);
    res.logloss_merge = ComputeLogloss(probs_merge, labels);

    res.ctr10_base = ComputeCTRatK(user_scores_base, topK_eval);
    res.ctr10_ours = ComputeCTRatK(user_scores_ours, topK_eval);
    res.ctr10_merge = ComputeCTRatK(user_scores_merge, topK_eval);

    res.time_ms_per_user = chrono::duration<double, milli>(t2 - t1).count() / max(1, (int)eval_users.size());
    return res;
}

int main() {

    string root = "/processed_kuairand1k_itemgraph_ctr";

    string user_emb_path = root + "/user_emb.bin";
    string item_emb_path = root + "/item_emb.bin";
    string item_useragg_emb_path = root + "/item_useragg_emb.bin";
    string fused_item_emb_mode2_path = root + "/fused_item_emb_mode2.bin";
    string item_graph_edges_path = root + "/item_graph_edges.txt";
    string eval_impressions_path = root + "/eval_impressions_random.txt";
    string calib_mode1_path = root + "/calibration_mode1.txt";
    string calib_mode2_path = root + "/calibration_mode2.txt";
    string config_path = root + "/config.txt";

    size_t n_users = 0, dim_u = 0;
    size_t n_items = 0, dim_i = 0;
    size_t n_items2 = 0, dim_i2 = 0;
    size_t n_items3 = 0, dim_i3 = 0;

    float* user_emb = load_embedding_bin(user_emb_path, n_users, dim_u);
    float* item_emb = load_embedding_bin(item_emb_path, n_items, dim_i);
    float* item_useragg_emb = load_embedding_bin(item_useragg_emb_path, n_items2, dim_i2);
    float* fused_item_emb_mode2 = load_embedding_bin(fused_item_emb_mode2_path, n_items3, dim_i3);

    if (!(dim_u == dim_i && dim_i == dim_i2 && dim_i2 == dim_i3)) {
        cerr << "Embedding dim mismatch." << endl;
        return -1;
    }
    if (!(n_items == n_items2 && n_items2 == n_items3)) {
        cerr << "Item count mismatch." << endl;
        return -1;
    }

    auto calib1 = LoadCalibration(calib_mode1_path);
    auto calib2 = LoadCalibration(calib_mode2_path);
    double alpha_mode1 = LoadConfigValue(config_path, "alpha_mode1");

    auto eval_rows = LoadEvalImpressions(eval_impressions_path);

    unordered_set<int> eval_user_set;
    for (const auto& r : eval_rows) eval_user_set.insert(r.user);
    vector<int> eval_users(eval_user_set.begin(), eval_user_set.end());


    SocialGraph item_graph(n_items);
    LoadGraphEdges(item_graph, item_graph_edges_path);


    HNSW<float> index_mode1(dim_i, n_items, 16, 200);
    index_mode1.SetSocialGraph(&item_graph);
    index_mode1.SetAlphaParams(0.0f, false, 200000.0f, 300.0f);

    #pragma omp parallel for
    for (int it = 0; it < (int)n_items; ++it) {
        index_mode1.AddPoint(item_emb + (size_t)it * dim_i);
    }
    index_mode1.SetReady(true);


    HNSW<float> index_mode2(dim_i, n_items, 16, 200);
    index_mode2.SetSocialGraph(&item_graph);
    index_mode2.SetAlphaParams(0.0f, false, 200000.0f, 300.0f);

    #pragma omp parallel for
    for (int it = 0; it < (int)n_items; ++it) {
        index_mode2.AddPoint(fused_item_emb_mode2 + (size_t)it * dim_i);
    }
    index_mode2.SetReady(true);

    int K_items = 100;
    int ef_search = 100;
    int topK_eval = 10;
    float epsilon = 0.2f;
    vector<float> taus = {0.70f, 0.80f, 0.90f};

    cout << "\n============================================================\n";
    cout << "KuaiRand-1K CTR Experiment on Item Graph\n";
    cout << "============================================================\n";
    cout << "n_users=" << n_users
         << ", n_items=" << n_items
         << ", eval_impressions=" << eval_rows.size()
         << ", eval_users=" << eval_users.size()
         << ", K_items=" << K_items
         << ", topK_eval=" << topK_eval
         << endl;

    for (float tau : taus) {
        auto res1 = EvaluateOneMode(
            index_mode1,
            eval_rows, eval_users,
            user_emb, item_emb, item_useragg_emb, fused_item_emb_mode2,
            dim_i, K_items, ef_search, topK_eval, epsilon, tau,
            1, alpha_mode1, calib1.first, calib1.second
        );

        auto res2 = EvaluateOneMode(
            index_mode2,
            eval_rows, eval_users,
            user_emb, item_emb, item_useragg_emb, fused_item_emb_mode2,
            dim_i, K_items, ef_search, topK_eval, epsilon, tau,
            2, alpha_mode1, calib2.first, calib2.second
        );

        cout << "\n============================================================\n";
        cout << "tau = " << tau << endl;
        cout << "============================================================\n";

        cout << "\n[ScoreMode1_Separate]\n";
        cout << left
             << setw(24) << "Method"
             << setw(16) << "CTR@10(%)"
             << setw(16) << "AUC(%)"
             << setw(16) << "Logloss"
             << setw(16) << "Time(ms/user)"
             << endl;
        cout << "------------------------------------------------------------------------\n";

        cout << left
             << setw(24) << "Baseline-Recall"
             << setw(16) << (res1.ctr10_base * 100.0)
             << setw(16) << (res1.auc_base * 100.0)
             << setw(16) << res1.logloss_base
             << setw(16) << res1.time_ms_per_user
             << endl;

        cout << left
             << setw(24) << "Ours-Recall"
             << setw(16) << (res1.ctr10_ours * 100.0)
             << setw(16) << (res1.auc_ours * 100.0)
             << setw(16) << res1.logloss_ours
             << setw(16) << res1.time_ms_per_user
             << endl;


        cout << "\n[ScoreMode2_Fused]\n";
        cout << left
             << setw(24) << "Method"
             << setw(16) << "CTR@10(%)"
             << setw(16) << "AUC(%)"
             << setw(16) << "Logloss"
             << setw(16) << "Time(ms/user)"
             << endl;
        cout << "------------------------------------------------------------------------\n";

        cout << left
             << setw(24) << "Baseline-Recall"
             << setw(16) << (res2.ctr10_base * 100.0)
             << setw(16) << (res2.auc_base * 100.0)
             << setw(16) << res2.logloss_base
             << setw(16) << res2.time_ms_per_user
             << endl;

        cout << left
             << setw(24) << "Ours-Recall"
             << setw(16) << (res2.ctr10_ours * 100.0)
             << setw(16) << (res2.auc_ours * 100.0)
             << setw(16) << res2.logloss_ours
             << setw(16) << res2.time_ms_per_user
             << endl;

    }

    delete[] user_emb;
    delete[] item_emb;
    delete[] item_useragg_emb;
    delete[] fused_item_emb_mode2;
    return 0;
}
