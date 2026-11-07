#pragma once

// #include <index_status.hpp>
// #include <graph/visited_list_pool.hpp>
#include "vector_ops.hpp"

#include "random"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include "queue"
#include <deque>
#include <memory>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

#include "binary_io.hpp"
#include "stimer.hpp"

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns
{

  namespace graph
  {

    class SocialGraph {
    private:
        std::vector<std::vector<int>> adj_;
        size_t num_nodes_;

    public:
        SocialGraph(size_t n) : num_nodes_(n) {
            adj_.resize(n);
        }

        void add_edge(int u, int v) {
            if (u >= num_nodes_ || v >= num_nodes_) return;
            adj_[u].push_back(v);
            adj_[v].push_back(u);
        }

        float get_degree(int u) const {
            if (u >= num_nodes_) return 0.0f;
            return (float)adj_[u].size();
        }

        const std::vector<int>& get_neighbors(int u) const {
            static const std::vector<int> empty;
            if (u >= num_nodes_) return empty;
            return adj_[u];
        }
    };

    template <typename vdim_t>
    class HNSW
    {
    using id_t = int;
    public:
      size_t max_elements_{0};
      size_t cur_element_count_{0};
      size_t size_data_per_element_{0};
      size_t size_links_per_element_{0};

      size_t M_{0};
      size_t Mmax_{0};
      size_t Mmax0_{0};

      size_t ef_construction_{0};

      double mult_{0.0};
      double rev_size_{0.0};
      int max_level_{0};

      std::mutex global_;
      std::unique_ptr<std::vector<std::mutex>> link_list_locks_;

      id_t enterpoint_node_{0};

      size_t size_links_level0_{0};
      size_t offset_data_{0};

      std::vector<char> data_level0_memory_;
      std::vector<std::vector<char>> link_lists_;
      std::vector<int> element_levels_;

      size_t data_size_{0};
      size_t D_{0};

      std::default_random_engine level_generator_;
      int random_seed_{100};

      bool ready_{false};

      size_t num_threads_{1};

      std::atomic<size_t> comparison_{0};

      SocialGraph* social_graph_{nullptr};

      float alpha_global_ = 0.0f;
      bool use_adaptive_alpha_ = false;

      float max_dist_norm_ = 200000.0f;
      float max_degree_norm_ = 300.0f;

      void SetAlphaParams(float alpha, bool adaptive, float max_dist, float max_deg) {
          alpha_global_ = alpha;
          use_adaptive_alpha_ = adaptive;
          max_dist_norm_ = max_dist;
          max_degree_norm_ = max_deg;
          std::cout << "[Config] Alpha=" << alpha
                    << ", Adaptive=" << adaptive
                    << ", NormDist=" << max_dist
                    << ", NormDeg=" << max_deg << std::endl;
      }

      HNSW(
          size_t D,
          size_t max_elements,
          size_t M = 16,
          size_t ef_construction = 128,
          size_t random_seed = 123) :

                                      D_(D), max_elements_(max_elements), M_(M), Mmax_(M), Mmax0_(2 * M),
                                      ef_construction_(std::max(ef_construction, M)), random_seed_(random_seed), element_levels_(max_elements)
      {
        level_generator_.seed(random_seed);

        data_size_ = D * sizeof(vdim_t);
        size_links_level0_ = Mmax0_ * sizeof(id_t) + sizeof(size_t);
        size_data_per_element_ = size_links_level0_ + sizeof(vdim_t *);
        offset_data_ = size_links_level0_;

        data_level0_memory_.resize(max_elements_ * size_data_per_element_, 0x00);

        cur_element_count_ = 0;
        enterpoint_node_ = -1;
        max_level_ = -1;

        link_lists_.resize(max_elements);
        link_list_locks_ = std::make_unique<std::vector<std::mutex>>(max_elements_);
        element_levels_.resize(max_elements, -1);

        size_links_per_element_ = Mmax_ * sizeof(id_t) + sizeof(size_t);

        mult_ = 1 / log(1.0 * M_);
        rev_size_ = 1.0 / mult_;
      }

      void SetSocialGraph(SocialGraph* sg) {
          social_graph_ = sg;
      }

      id_t AddPoint(const vdim_t *data_point)
      {
        if (cur_element_count_ >= max_elements_)
        {
          std::cerr << "The number of elements exceeds the specified limit" << std::endl;
          exit(1);
        }

        id_t cur_id;
        int cur_level;
        {
          std::unique_lock<std::mutex> temp_lock(global_);
          cur_level = GetRandomLevel(mult_);
          cur_id = cur_element_count_++;
        }

        std::unique_lock<std::mutex> lock_el((*link_list_locks_)[cur_id]);


        element_levels_[cur_id] = cur_level;

        std::unique_lock<std::mutex> temp_lock(global_);
        int max_level_copy = max_level_;
        id_t cur_obj = enterpoint_node_;
        id_t enterpoint_node_copy = enterpoint_node_;
        if (cur_level <= max_level_)
          temp_lock.unlock();

        WriteDataByInternalID(cur_id, data_point);

        if (cur_level)
        {
          link_lists_[cur_id].resize(size_links_per_element_ * cur_level, 0x00);
        }

        if (enterpoint_node_copy != -1)
        {
          if (cur_level < max_level_copy)
          {
            float cur_dist = vec_L2sqr(data_point, GetDataByInternalID(cur_obj), D_);
            for (int lev = max_level_copy; lev > cur_level; lev--)
            {
              bool changed = true;
              while (changed)
              {
                changed = false;
                std::unique_lock<std::mutex> wlock(link_list_locks_->at(cur_obj));
                size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
                size_t num_neighbors = *ll_cur;
                id_t *neighbors = (id_t *)(ll_cur + 1);

                for (size_t i = 0; i < num_neighbors; i++)
                {
                  id_t cand = neighbors[i];
                  float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
                  if (d < cur_dist)
                  {
                    cur_dist = d;
                    cur_obj = cand;
                    changed = true;
                  }
                }
              }
            }
          }
          for (int lev = std::min(cur_level, max_level_copy); lev >= 0; lev--)
          {
            auto top_candidates = SearchBaseLayer(cur_obj, data_point, lev, ef_construction_);
            cur_obj = MutuallyConnectNewElement(data_point, cur_id, top_candidates, lev);
          }
        }
        else
        {
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }

        if (cur_level > max_level_copy)
        {
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }
        return cur_id;
      }

      void Populate(const std::vector<vdim_t> &raw_data)
      {
        size_t N = raw_data.size() / D_;
        assert(N <= max_elements_ && "data size too large!");

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < N; i++)
        {
          AddPoint(raw_data.data() + i * D_);
        }

        ready_ = true;
      }

      void Populate(const std::vector<const vdim_t *> &raw_data) {
          size_t N = raw_data.size();
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
          for (size_t i = 0; i < N; i++) {
              AddPoint(raw_data[i]);
          }
          ready_ = true;
      }

      std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef)
      {

          assert(ready_ && "Index uninitialized!");
          assert(ef >= k && "ef > k!");
          if (cur_element_count_ == 0) return std::priority_queue<std::pair<float, id_t>>();
          size_t comparison = 0;
          id_t cur_obj = enterpoint_node_;
          float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
          comparison++;
          for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--) {
              bool changed = true;
              while (changed) {
                  changed = false;
                  size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
                  size_t num_neighbors = *ll_cur;
                  id_t *neighbors = (id_t *)(ll_cur + 1);
                  for (size_t i = 0; i < num_neighbors; i++) {
                      id_t cand = neighbors[i];
                      if (cand < 0 || cand > max_elements_) exit(1);
                      float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
                      if (d < cur_dist) {
                          cur_dist = d;
                          cur_obj = cand;
                          changed = true;
                      }
                  }
                  comparison += num_neighbors;
              }
          }
          auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);
          while (top_candidates.size() > k) top_candidates.pop();
          comparison_.fetch_add(comparison);
          return top_candidates;
      }

      std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef, id_t ep)
      {
        assert(ready_ && "Index uninitialized!");

        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;

        id_t cur_obj = ep;
        float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        for (int lev = element_levels_[ep]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
            size_t num_neighbors = *ll_cur;
            id_t *neighbors = (id_t *)(ll_cur + 1);

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              if (cand < 0 || cand > max_elements_)
              {
                std::cerr << "cand error" << std::endl;
                exit(1);
              }
              float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      void Search(const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
      {
        size_t nq = queries.size();
        vids.clear();
        dists.clear();
        vids.resize(nq);
        dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = vids[i];
          auto &dist = dists[i];

          auto r = Search(query.data(), k, ef);
          vid.reserve(r.size());
          dist.reserve(r.size());
          while (r.size())
          {
            const auto &te = r.top();
            vid.emplace_back(te.second);
            dist.emplace_back(te.first);
            r.pop();
          }
        }
      }

            std::vector<int> SearchBaseline(const vdim_t* query_data, size_t K, size_t ef_search)
            {

                auto pq = Search(query_data, K, ef_search);

                std::vector<int> result_ids;
                result_ids.reserve(K);

                while (!pq.empty()) {
                    result_ids.push_back(pq.top().second);
                    pq.pop();
                }

                std::reverse(result_ids.begin(), result_ids.end());

                return result_ids;
            }



      bool Ready() { return ready_; }

      size_t GetNumThreads()
      {
        return num_threads_;
      }

      void SetNumThreads(size_t num_threads)
      {
        num_threads_ = num_threads;
      }

      void SetReady(bool ready) { ready_ = ready; }

      size_t GetComparisonAndClear()
      {
        return comparison_.exchange(0);
      }

      size_t IndexSize() const
      {
        size_t sz = 0;
        sz += data_level0_memory_.size() * sizeof(char);
        std::for_each(link_lists_.begin(), link_lists_.end(), [&](const std::vector<char> &bytes_arr)
                      { sz += bytes_arr.size() * sizeof(char); });
        // element levels
        sz += cur_element_count_ * sizeof(int);
        return sz;
      }

      inline const vdim_t *GetDataByInternalID(id_t id) const
      {
        return *((vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_));
      }

      inline void WriteDataByInternalID(id_t id, const vdim_t *data_point)
      {
        *((const vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_)) = data_point;
      }

      inline char *GetLinkByInternalID(id_t id, int level) const
      {
        if (level > 0)
          return (char *)(link_lists_[id].data() + (level - 1) * size_links_per_element_);

        return (char *)(data_level0_memory_.data() + id * size_data_per_element_);
      }

      id_t MutuallyConnectNewElement(
          const vdim_t *data_point,
          id_t id,
          std::priority_queue<std::pair<float, id_t>> &top_candidates,
          int level)
      {
          size_t Mcurmax = level ? Mmax_ : Mmax0_;

          GetNeighborsByHeuristic(top_candidates, M_);

          std::vector<id_t> selected_neighbors;
          selected_neighbors.reserve(M_);
          while (top_candidates.size())
          {
            selected_neighbors.emplace_back(top_candidates.top().second);
            top_candidates.pop();
          }

          id_t next_closet_entry_point = selected_neighbors.back();

          /// @brief Edge-slots check and Add neighbors for current vector
          {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::mutex> lock((*link_list_locks_)[id], std::defer_lock);
            size_t *ll_cur = (size_t *)GetLinkByInternalID(id, level);
            size_t num_neighbors = *ll_cur;

            if (num_neighbors)
            {
              std::cerr << "The newly inserted element should have blank link list" << std::endl;
              exit(1);
            }

            *ll_cur = selected_neighbors.size();

            id_t *neighbors = (id_t *)(ll_cur + 1);
            for (size_t i = 0; i < selected_neighbors.size(); i++)
            {
              if (neighbors[i])
              {
                std::cerr << "Possible memory corruption" << std::endl;
                exit(1);
              }
              if (level > element_levels_[selected_neighbors[i]])
              {
                std::cerr << "Trying to make a link on a non-existent level" << std::endl;
                exit(1);
              }

              neighbors[i] = selected_neighbors[i];
            }
          }

          for (size_t i = 0; i < selected_neighbors.size(); i++)
          {
            std::unique_lock<std::mutex> lock((*link_list_locks_)[selected_neighbors[i]]);

            size_t *ll_other = (size_t *)GetLinkByInternalID(selected_neighbors[i], level);
            size_t sz_link_list_other = *ll_other;

            if (sz_link_list_other > Mcurmax || sz_link_list_other < 0)
            {
              std::cerr << "Bad value of sz_link_list_other" << std::endl;
              exit(1);
            }
            if (selected_neighbors[i] == id)
            {
              std::cerr << "Trying to connect an element to itself" << std::endl;
              exit(1);
            }
            if (level > element_levels_[selected_neighbors[i]])
            {
              std::cerr << "Trying to make a link on a non-existent level" << std::endl;
              exit(1);
            }

            id_t *neighbors = (id_t *)(ll_other + 1);

            if (sz_link_list_other < Mcurmax)
            {
              neighbors[sz_link_list_other] = id;
              *ll_other = sz_link_list_other + 1;
            }
            else
            {
              // finding the "farest" element to replace it with the new one
              float d_max = vec_L2sqr(GetDataByInternalID(id), GetDataByInternalID(selected_neighbors[i]), D_);
              // Heuristic:
              std::priority_queue<std::pair<float, id_t>> candidates;
              candidates.emplace(d_max, id);

              for (size_t j = 0; j < sz_link_list_other; j++)
              {
                candidates.emplace(vec_L2sqr(GetDataByInternalID(neighbors[j]), GetDataByInternalID(selected_neighbors[i]), D_), neighbors[j]);
              }

              GetNeighborsByHeuristic(candidates, Mcurmax);

              // Copy neighbors and add edges.
              size_t nn = 0;
              while (candidates.size())
              {
                neighbors[nn] = candidates.top().second;
                candidates.pop();
                nn++;
              }
              *ll_other = nn;
            }
          }

          return next_closet_entry_point;
      }

      void GetNeighborsByHeuristic(std::priority_queue<std::pair<float, id_t>> &top_candidates, size_t NN)
      {
          if (top_candidates.size() < NN) return;

          struct Candidate {
              id_t id;
              float dist;
              float score;

              bool operator<(const Candidate& other) const {
                  return score < other.score;
              }
          };

          std::vector<Candidate> all_candidates;
          all_candidates.reserve(top_candidates.size());

          float sum_dist = 0.0f;
          int count = 0;


          while (!top_candidates.empty()) {
              float d = top_candidates.top().first;
              id_t id = top_candidates.top().second;
              top_candidates.pop();

              all_candidates.push_back({id, d, 0.0f});

              sum_dist += d;
              count++;
          }

          float current_alpha = alpha_global_;

          if (use_adaptive_alpha_) {

              float avg_dist = (count > 0) ? (sum_dist / count) : 0.0f;

              float ratio = avg_dist / (max_dist_norm_ * 0.5f);
              if (ratio > 1.0f) ratio = 1.0f;

              current_alpha = ratio * 0.8f;
          }

          for (auto& cand : all_candidates) {
              float dist_score = cand.dist;

              if (current_alpha > 1e-6f && social_graph_) {
                  float degree = social_graph_->get_degree(cand.id);

                  float norm_d = cand.dist / max_dist_norm_;
                  float norm_i = degree / max_degree_norm_;
                  if (norm_d > 1.0f) norm_d = 1.0f;
                  if (norm_i > 1.0f) norm_i = 1.0f;

                  dist_score = (1.0f - current_alpha) * norm_d - current_alpha * norm_i;
              }
              cand.score = dist_score;
          }

          std::sort(all_candidates.begin(), all_candidates.end());

          std::vector<Candidate> return_list;
          for (const auto& curen : all_candidates) {
              if (return_list.size() >= NN) break;

              bool good = true;
              for (const auto& prev : return_list) {
                  float dist_between = vec_L2sqr(GetDataByInternalID(curen.id),
                                                 GetDataByInternalID(prev.id), D_);

                  if (dist_between < curen.dist) {
                      good = false;
                      break;
                  }
              }

              if (good) {
                  return_list.push_back(curen);
              }
          }


          for (const auto& elem : return_list) {
              top_candidates.emplace(-elem.dist, elem.id);
          }
      }




      std::vector<int> SearchIterative(const vdim_t* query_data, size_t K, float epsilon, float beta, float alpha_search = 100.0f)
      {
          if (!social_graph_) return {};

          struct Node { int id; float dist; };
          std::vector<Node> all_fetched;
          std::unordered_set<int> unique_candidates;


          auto pq_phys = Search(query_data, K, std::max((size_t)100, K*2));

          while (!pq_phys.empty()) {
              auto top = pq_phys.top();
              pq_phys.pop();
              int id = (int)top.second;

              if (id < 0 || id >= max_elements_) continue;

              if (unique_candidates.find(id) == unique_candidates.end()) {
                  all_fetched.push_back({id, top.first});
                  unique_candidates.insert(id);
              }
          }

          size_t L_dynamic = K * 2;
          std::vector<int> ids_dynamic = SearchDynamicAlpha(query_data, L_dynamic, 200, alpha_search, 1 /*Linear*/);

          for (int id : ids_dynamic) {
              if (id < 0 || id >= max_elements_) continue;

              if (unique_candidates.find(id) == unique_candidates.end()) {

                  float d = vec_L2sqr(query_data, GetDataByInternalID(id), D_);
                  all_fetched.push_back({id, d});
                  unique_candidates.insert(id);
              }
          }

          std::sort(all_fetched.begin(), all_fetched.end(), [](const Node& a, const Node& b){
              return a.dist < b.dist;
          });


          std::vector<Node> S;
          S.reserve(K);
          std::unordered_set<int> in_S_set;

          for (size_t i = 0; i < all_fetched.size() && S.size() < K; ++i) {
              S.push_back(all_fetched[i]);
              in_S_set.insert(all_fetched[i].id);
          }
          if (S.empty()) return {};


          std::unordered_map<int, int> coverage_map;
          float d_bound = 0.0f;
          float sum_dist = 0.0f;
          float sum_degree = 0.0f;

          for (const auto& node : S) {
              if (node.dist > d_bound) d_bound = node.dist;
              sum_dist += node.dist;
              sum_degree += social_graph_->get_degree(node.id);

              const auto& nbs = social_graph_->get_neighbors(node.id);
              for (int nb : nbs) coverage_map[nb]++;
          }

          float lambda = d_bound * (1.0f + epsilon);

          float avg_dist = sum_dist / (float)S.size();
          float avg_degree = sum_degree / (float)S.size();

          float theta = 0.0f;
          if (std::abs(avg_dist) < 1e-9) theta = 1e9f;
          else theta = beta * (avg_degree / avg_dist);

          std::deque<int> queue;
          std::unordered_set<int> visited;
          for(const auto& node : S) visited.insert(node.id);

          for (const auto& node : S) {
              size_t *ll = (size_t *)GetLinkByInternalID(node.id, 0);
              size_t num = *ll;
              int *phys_nbs = (int *)(ll + 1);

              for (size_t i = 0; i < num; ++i) {
                  int nb_id = phys_nbs[i];
                  if (nb_id < 0 || nb_id >= cur_element_count_) continue;

                  if (visited.find(nb_id) == visited.end()) {
                      float d = vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_);
                      if (d <= lambda) {
                          queue.push_back(nb_id);
                          visited.insert(nb_id);
                      }
                  }
              }
          }

          float degree_threshold = avg_degree * 0.5f;
          for (size_t i = K; i < all_fetched.size(); ++i) {
              int id = all_fetched[i].id;

              if (visited.find(id) == visited.end() && all_fetched[i].dist <= lambda) {

                  if (social_graph_->get_degree(id) >= degree_threshold) {
                      queue.push_back(id);
                      visited.insert(id);
                  }
              }
          }

          while (!queue.empty()) {
              int c_id = queue.front();
              queue.pop_front();

              if (c_id < 0 || c_id >= cur_element_count_) continue;
              float d_c = vec_L2sqr(query_data, GetDataByInternalID(c_id), D_);

              if (d_c > lambda) continue;


              int gain_c = 0;
              const auto& c_nbs = social_graph_->get_neighbors(c_id);
              for (int nb : c_nbs) {
                  if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_c++;
              }

              int best_victim_idx = -1;
              float max_score = -1e9f;
              bool do_swap_1v1 = false;

              for (int i = 0; i < S.size(); ++i) {
                  float d_v = S[i].dist;
                  int v_id = S[i].id;

                  int loss_v = 0;
                  const auto& v_nbs = social_graph_->get_neighbors(v_id);
                  for (int nb : v_nbs) if (coverage_map[nb] == 1) loss_v++;

                  bool better = false;

                  if (d_c <= d_v && gain_c > loss_v) better = true;

                  else if (d_c > d_v && d_c <= lambda) {
                      float gd = (float)(gain_c - loss_v);
                      float dd = d_c - d_v;
                      if (dd < 1e-9) dd = 1e-9;
                      if (gd > 0 && (gd / dd) > theta) better = true;
                  }

                  if (better) {
                      float score = (float)(gain_c - loss_v);
                      if (score > max_score) {
                          max_score = score;
                          best_victim_idx = i;
                          do_swap_1v1 = true;
                      }
                  }
              }


              bool do_swap_2v2 = false;
              int v1_idx = -1, v2_idx = -1;
              int filler_id = -1;
              float filler_dist = 0.0f;

              if (!do_swap_1v1 && S.size() >= 2) {


                  for (const auto& node : all_fetched) {
                      if (in_S_set.find(node.id) == in_S_set.end() && node.id != c_id) {
                          filler_id = node.id;
                          filler_dist = node.dist;
                          break;
                      }
                  }

                  if (filler_id != -1 && filler_dist <= lambda) {
                      int gain_filler = 0;
                      const auto& f_nbs = social_graph_->get_neighbors(filler_id);
                      for (int nb : f_nbs) {
                          if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_filler++;
                      }

                      std::vector<std::pair<int, int>> victims_sorted;
                      for(int i=0; i<S.size(); ++i) {
                          int loss = 0;
                          const auto& nbs = social_graph_->get_neighbors(S[i].id);
                          for(int nb : nbs) if(coverage_map[nb]==1) loss++;
                          victims_sorted.push_back({loss, i});
                      }
                      std::sort(victims_sorted.begin(), victims_sorted.end());

                      v1_idx = victims_sorted[0].second;
                      v2_idx = victims_sorted[1].second;
                      int total_loss = victims_sorted[0].first + victims_sorted[1].first;

                      float dist_in = d_c + filler_dist;
                      float dist_out = S[v1_idx].dist + S[v2_idx].dist;
                      int gain_in = gain_c + gain_filler;

                      float gd = (float)(gain_in - total_loss);
                      float dd = dist_in - dist_out;


                      if (dd <= 0 && gd > 0) do_swap_2v2 = true;
                      else if (dd > 0 && gd > 0) {
                          if (dd < 1e-9) dd = 1e-9;
                          if ((gd / dd) > theta) do_swap_2v2 = true;
                      }
                  }
              }


              if (do_swap_1v1) {
                  int v_id = S[best_victim_idx].id;
                  in_S_set.erase(v_id); in_S_set.insert(c_id);
                  S[best_victim_idx] = {c_id, d_c};

                  for (int nb : social_graph_->get_neighbors(v_id)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;

                  size_t *ll = (size_t *)GetLinkByInternalID(c_id, 0);
                  size_t num = *ll;
                  int *phys_nbs = (int *)(ll + 1);
                  for(size_t i=0; i<num; ++i) {
                      int nb_id = phys_nbs[i];
                      if(visited.find(nb_id) == visited.end() &&
                              vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_) <= lambda) {
                          queue.push_back(nb_id); visited.insert(nb_id);
                      }
                  }

              } else if (do_swap_2v2) {
                  int vid1 = S[v1_idx].id;
                  int vid2 = S[v2_idx].id;

                  in_S_set.erase(vid1); in_S_set.erase(vid2);
                  in_S_set.insert(c_id); in_S_set.insert(filler_id);

                  S[v1_idx] = {c_id, d_c};
                  S[v2_idx] = {filler_id, filler_dist};

                  for (int nb : social_graph_->get_neighbors(vid1)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(vid2)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;
                  for (int nb : social_graph_->get_neighbors(filler_id)) coverage_map[nb]++;

                  int new_ids[2] = {c_id, filler_id};
                  for(int new_id : new_ids) {
                      size_t *ll = (size_t *)GetLinkByInternalID(new_id, 0);
                      size_t num = *ll;
                      int *phys_nbs = (int *)(ll + 1);
                      for(size_t i=0; i<num; ++i) {
                          int nb_id = phys_nbs[i];
                          if(visited.find(nb_id) == visited.end() &&
                                  vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_) <= lambda) {
                              queue.push_back(nb_id); visited.insert(nb_id);
                          }
                      }
                  }
              }


              if (do_swap_1v1 || do_swap_2v2) {
                  d_bound = 0.0f;
                  for (const auto& node : S) d_bound = std::max(d_bound, node.dist);
                  lambda = d_bound * (1.0f + epsilon);
              }
          }

          std::vector<int> result_ids;
          for (const auto& node : S) result_ids.push_back(node.id);
          return result_ids;
      }



      std::vector<int> SearchDynamicAlpha(const vdim_t* query_data, size_t K, size_t ef, float alpha_max, int curve_type = 0)
      {
          if (!ready_) throw std::runtime_error("Index not initialized!");
          if (cur_element_count_ == 0) return {};
          if (!social_graph_) return {};

          id_t cur_obj = enterpoint_node_;
          float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);

          for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--) {
              bool changed = true;
              while (changed) {
                  changed = false;
                  size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
                  size_t num_neighbors = *ll_cur;
                  id_t *neighbors = (id_t *)(ll_cur + 1);

                  for (size_t i = 0; i < num_neighbors; i++) {
                      id_t cand = neighbors[i];
                      float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
                      if (d < cur_dist) {
                          cur_dist = d;
                          cur_obj = cand;
                          changed = true;
                      }
                  }
              }
          }

          auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

          std::priority_queue<
              std::pair<float, id_t>,
              std::vector<std::pair<float, id_t>>,
              std::greater<std::pair<float, id_t>>
          > candidate_set;

          std::priority_queue<std::pair<float, id_t>> top_candidates;

          std::unordered_map<int, int> coverage_counts;

          float initial_score = cur_dist;
          top_candidates.emplace(initial_score, cur_obj);
          candidate_set.emplace(initial_score, cur_obj);
          mass_visited->at(cur_obj) = true;

          const auto& init_nbs = social_graph_->get_neighbors(cur_obj);
          for (int nb : init_nbs) {
              coverage_counts[nb]++;
          }

          float lower_bound = initial_score;

          while (!candidate_set.empty()) {
              auto curr_el_pair = candidate_set.top();
              float curr_score = curr_el_pair.first;
              id_t curr_node_id = curr_el_pair.second;
              candidate_set.pop();

              if (curr_score > lower_bound) {
                  if (top_candidates.size() == ef) {
                      break;
                  }
              }

              std::unique_lock<std::mutex> lock((*link_list_locks_)[curr_node_id]);
              size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id, 0);
              size_t num_neighbors = *ll_cur;
              id_t *neighbors = (id_t *)(ll_cur + 1);

              float current_alpha = 0.0f;
              float progress = (ef > 0) ? ((float)top_candidates.size() / (float)ef) : 1.0f;
              if (progress > 1.0f) progress = 1.0f;

              switch (curve_type) {
              case 0:
                  current_alpha = (top_candidates.size() >= ef) ? alpha_max : 0.0f;
                  break;
              case 1:
                  current_alpha = alpha_max * progress;
                  break;
              case 2:
                  current_alpha = alpha_max * progress * progress;
                  break;
              case 3:
                  if (progress < 0.5f) current_alpha = 0.0f;
                  else current_alpha = alpha_max * ((progress - 0.5f) / 0.5f);
                  break;
              default:
                  current_alpha = (top_candidates.size() >= ef) ? alpha_max : 0.0f;
                  break;
              }

              for (size_t j = 0; j < num_neighbors; j++) {
                  id_t neighbor_id = neighbors[j];

                  if (mass_visited->at(neighbor_id) == false) {
                      mass_visited->at(neighbor_id) = true;

                      float dist = vec_L2sqr(query_data, GetDataByInternalID(neighbor_id), D_);

                      int marginal_gain = 0;
                      const auto& social_nbs = social_graph_->get_neighbors(neighbor_id);

                      if (current_alpha > 1e-6f) {
                          for (int nb : social_nbs) {
                              if (coverage_counts.find(nb) == coverage_counts.end() || coverage_counts[nb] == 0) {
                                  marginal_gain++;
                              }
                          }
                      }

                      float norm_dist = dist / std::max(1e-6f, max_dist_norm_);
                      if (norm_dist > 1.0f) norm_dist = 1.0f;

                      float norm_gain = (float)marginal_gain / std::max(1.0f, max_degree_norm_);
                      if (norm_gain > 1.0f) norm_gain = 1.0f;

                      float alpha_mix = current_alpha / (current_alpha + 1.0f);
                      if (alpha_mix < 0.0f) alpha_mix = 0.0f;
                      if (alpha_mix > 1.0f) alpha_mix = 1.0f;

                      float score = (1.0f - alpha_mix) * norm_dist - alpha_mix * norm_gain;

                      if (top_candidates.size() < ef || score < lower_bound) {
                          candidate_set.emplace(score, neighbor_id);
                          top_candidates.emplace(score, neighbor_id);

                          for (int nb : social_nbs) {
                              coverage_counts[nb]++;
                          }

                          if (top_candidates.size() > ef) {
                              id_t evicted_id = top_candidates.top().second;
                              top_candidates.pop();

                              const auto& evicted_nbs = social_graph_->get_neighbors(evicted_id);
                              for (int nb : evicted_nbs) {
                                  if (coverage_counts[nb] > 0) {
                                      coverage_counts[nb]--;
                                  }
                              }
                          }

                          if (!top_candidates.empty()) {
                              lower_bound = top_candidates.top().first;
                          }
                      }
                  }
              }
          }

          std::vector<int> result_ids;
          while (!top_candidates.empty()) {
              result_ids.push_back(top_candidates.top().second);
              top_candidates.pop();
          }
          std::reverse(result_ids.begin(), result_ids.end());

          if (result_ids.size() > K) {
              result_ids.resize(K);
          }

          return result_ids;
      }




      std::vector<int> SearchIterativeOriginal1v1(const vdim_t* query_data, size_t K, float epsilon, float beta)
      {
          if (!social_graph_) return {};

          size_t ef_search = std::max((size_t)100, K * 2);
          auto pq = Search(query_data, K, ef_search);

          if (pq.empty()) return {};

          struct Node { int id; float dist; };
          std::vector<Node> S;
          S.reserve(K);

          while (!pq.empty()) {
              auto top = pq.top(); pq.pop();
              if (top.second >= 0 && top.second < max_elements_) {
                  S.push_back({(int)top.second, top.first});
              }
          }
          std::reverse(S.begin(), S.end());

          if (S.empty()) return {};

          std::unordered_map<int, int> coverage_map;
          float d_bound = 0.0f;
          float sum_dist = 0.0f;
          float sum_degree = 0.0f;

          for (const auto& node : S) {
              if (node.dist > d_bound) d_bound = node.dist;
              sum_dist += node.dist;
              sum_degree += social_graph_->get_degree(node.id);

              const auto& nbs = social_graph_->get_neighbors(node.id);
              for (int nb : nbs) coverage_map[nb]++;
          }

          float lambda = d_bound * (1.0f + epsilon);
          float avg_dist = sum_dist / (float)S.size();
          float avg_degree = sum_degree / (float)S.size();
          float theta = 0.0f;
          if (std::abs(avg_dist) < 1e-9) theta = 1e9f;
          else theta = beta * (avg_degree / avg_dist);

          std::deque<int> queue;
          std::unordered_set<int> visited;
          for(const auto& node : S) visited.insert(node.id);

          for (const auto& node : S) {
              size_t *ll = (size_t *)GetLinkByInternalID(node.id, 0);
              size_t num = *ll;
              int *phys_nbs = (int *)(ll + 1);

              for (size_t i = 0; i < num; ++i) {
                  int nb_id = phys_nbs[i];
                  if (nb_id < 0 || nb_id >= cur_element_count_) continue;

                  if (visited.find(nb_id) == visited.end()) {
                      float d = vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_);
                      if (d <= lambda) {
                          queue.push_back(nb_id);
                          visited.insert(nb_id);
                      }
                  }
              }
          }


          while (!queue.empty()) {
              int c_id = queue.front();
              queue.pop_front();

              float d_c = vec_L2sqr(query_data, GetDataByInternalID(c_id), D_);

              if (d_c > lambda) continue;

              int gain_c = 0;
              const auto& c_nbs = social_graph_->get_neighbors(c_id);
              for (int nb : c_nbs) {
                  if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_c++;
              }

              int best_victim_idx = -1;
              float max_score = -1e9f;
              bool do_swap = false;

              for (int i = 0; i < S.size(); ++i) {
                  float d_v = S[i].dist;
                  int v_id = S[i].id;

                  int loss_v = 0;
                  const auto& v_nbs = social_graph_->get_neighbors(v_id);
                  for (int nb : v_nbs) if (coverage_map[nb] == 1) loss_v++;

                  bool better = false;

                  if (d_c <= d_v && gain_c > loss_v) better = true;

                  else if (d_c > d_v && d_c <= lambda) {
                      float gd = (float)(gain_c - loss_v);
                      float dd = d_c - d_v;
                      if (dd < 1e-9) dd = 1e-9;
                      if (gd > 0 && (gd / dd) > theta) better = true;
                  }

                  if (better) {
                      float score = (float)(gain_c - loss_v);
                      if (score > max_score) {
                          max_score = score;
                          best_victim_idx = i;
                          do_swap = true;
                      }
                  }
              }

              if (do_swap && best_victim_idx != -1) {
                  int v_id = S[best_victim_idx].id;

                  S[best_victim_idx] = {c_id, d_c};

                  for (int nb : social_graph_->get_neighbors(v_id)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;

                  d_bound = 0.0f;
                  for (const auto& node : S) d_bound = std::max(d_bound, node.dist);
                  lambda = d_bound * (1.0f + epsilon);

                  size_t *ll = (size_t *)GetLinkByInternalID(c_id, 0);
                  size_t num = *ll;
                  int *phys_nbs = (int *)(ll + 1);
                  for(size_t i=0; i<num; ++i) {
                      int nb_id = phys_nbs[i];
                      if (nb_id < 0 || nb_id >= cur_element_count_) continue;

                      if(visited.find(nb_id) == visited.end()) {
                          float d = vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_);
                          if (d <= lambda) {
                              queue.push_back(nb_id);
                              visited.insert(nb_id);
                          }
                      }
                  }
              }
          }

          std::vector<int> result_ids;
          for (const auto& node : S) result_ids.push_back(node.id);
          return result_ids;
      }




      std::vector<int> SearchIterativeWithOverfetch(const vdim_t* query_data, size_t K, float epsilon, float beta)
      {
          if (!social_graph_) return {};

          size_t L_fetch = K * 3;
          size_t ef_search = std::max((size_t)100, L_fetch * 2);
          auto pq = Search(query_data, L_fetch, ef_search);

          if (pq.empty()) return {};

          struct Node { int id; float dist; };
          std::vector<Node> all_fetched;
          all_fetched.reserve(L_fetch);

          while (!pq.empty()) {
              auto top = pq.top(); pq.pop();
              if (top.second >= 0 && top.second < max_elements_) {
                  all_fetched.push_back({(int)top.second, top.first});
              }
          }

          std::reverse(all_fetched.begin(), all_fetched.end());

          std::vector<Node> S;
          S.reserve(K);
          std::unordered_set<int> visited;

          for (size_t i = 0; i < all_fetched.size() && S.size() < K; ++i) {
              S.push_back(all_fetched[i]);
              visited.insert(all_fetched[i].id);
          }

          if (S.empty()) return {};

          std::unordered_map<int, int> coverage_map;
          float d_bound = 0.0f;
          float sum_dist = 0.0f;
          float sum_degree = 0.0f;

          for (const auto& node : S) {
              if (node.dist > d_bound) d_bound = node.dist;
              sum_dist += node.dist;
              sum_degree += social_graph_->get_degree(node.id);

              const auto& nbs = social_graph_->get_neighbors(node.id);
              for (int nb : nbs) coverage_map[nb]++;
          }

          float lambda = d_bound * (1.0f + epsilon);

          float avg_dist = sum_dist / (float)S.size();
          float avg_degree = sum_degree / (float)S.size();
          float theta = 0.0f;
          if (std::abs(avg_dist) < 1e-9) theta = 1e9f;
          else theta = beta * (avg_degree / avg_dist);

          std::deque<int> queue;

          for (const auto& node : S) {
              size_t *ll = (size_t *)GetLinkByInternalID(node.id, 0);
              size_t num = *ll;
              int *phys_nbs = (int *)(ll + 1);

              for (size_t i = 0; i < num; ++i) {
                  int nb_id = phys_nbs[i];
                  if (nb_id < 0 || nb_id >= cur_element_count_) continue;

                  if (visited.find(nb_id) == visited.end()) {
                      float d = vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_);
                      if (d <= lambda) {
                          queue.push_back(nb_id);
                          visited.insert(nb_id);
                      }
                  }
              }
          }

          for (size_t i = K; i < all_fetched.size(); ++i) {
              int id = all_fetched[i].id;
              float dist = all_fetched[i].dist;

              if (visited.find(id) == visited.end()) {
                  if (dist <= lambda) {
                      queue.push_back(id);
                      visited.insert(id);
                  }
              }
          }


          while (!queue.empty()) {
              int c_id = queue.front();
              queue.pop_front();

              float d_c = vec_L2sqr(query_data, GetDataByInternalID(c_id), D_);
              if (d_c > lambda) continue;

              int gain_c = 0;
              const auto& c_nbs = social_graph_->get_neighbors(c_id);
              for (int nb : c_nbs) {
                  if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_c++;
              }

              int best_victim_idx = -1;
              float max_score = -1e9f;
              bool do_swap = false;

              for (int i = 0; i < S.size(); ++i) {
                  float d_v = S[i].dist;
                  int v_id = S[i].id;

                  int loss_v = 0;
                  const auto& v_nbs = social_graph_->get_neighbors(v_id);
                  for (int nb : v_nbs) if (coverage_map[nb] == 1) loss_v++;

                  bool better = false;

                  if (d_c <= d_v && gain_c > loss_v) better = true;

                  else if (d_c > d_v && d_c <= lambda) {
                      float gd = (float)(gain_c - loss_v);
                      float dd = d_c - d_v;
                      if (dd < 1e-9) dd = 1e-9;
                      if (gd > 0 && (gd / dd) > theta) better = true;
                  }

                  if (better) {
                      float score = (float)(gain_c - loss_v);
                      if (score > max_score) {
                          max_score = score;
                          best_victim_idx = i;
                          do_swap = true;
                      }
                  }
              }

              if (do_swap && best_victim_idx != -1) {
                  int v_id = S[best_victim_idx].id;

                  S[best_victim_idx] = {c_id, d_c};

                  for (int nb : social_graph_->get_neighbors(v_id)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;

                  d_bound = 0.0f;
                  for (const auto& node : S) d_bound = std::max(d_bound, node.dist);
                  lambda = d_bound * (1.0f + epsilon);

                  size_t *ll = (size_t *)GetLinkByInternalID(c_id, 0);
                  size_t num = *ll;
                  int *phys_nbs = (int *)(ll + 1);
                  for(size_t i=0; i<num; ++i) {
                      int nb_id = phys_nbs[i];
                      if(visited.find(nb_id) == visited.end() &&
                              vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_) <= lambda) {
                          queue.push_back(nb_id); visited.insert(nb_id);
                      }
                  }
              }
          }

          std::vector<int> result_ids;
          for (const auto& node : S) result_ids.push_back(node.id);
          return result_ids;
      }




      std::vector<int> SearchIterative2to2final(const vdim_t* query_data, size_t K, float epsilon, float beta)
      {
          if (!social_graph_) return {};

          size_t L_fetch = K * 3;
          size_t ef_search = std::max((size_t)100, L_fetch * 2);

          auto pq = Search(query_data, L_fetch, ef_search);

          if (pq.empty()) return {};

          struct Node { int id; float dist; };
          std::vector<Node> all_fetched;
          all_fetched.reserve(L_fetch);

          while (!pq.empty()) {
              auto top = pq.top(); pq.pop();
              if (top.second >= 0 && top.second < max_elements_) {
                  all_fetched.push_back({(int)top.second, top.first});
              }
          }
          std::reverse(all_fetched.begin(), all_fetched.end());
          std::vector<Node> S;
          S.reserve(K);
          std::unordered_set<int> visited;
          std::unordered_set<int> in_S_set;


          for (size_t i = 0; i < all_fetched.size() && S.size() < K; ++i) {
              S.push_back(all_fetched[i]);
              visited.insert(all_fetched[i].id);
              in_S_set.insert(all_fetched[i].id);
          }

          if (S.empty()) return {};

          std::unordered_map<int, int> coverage_map;
          float d_bound = 0.0f;
          float sum_dist = 0.0f;
          float sum_degree = 0.0f;

          for (const auto& node : S) {
              if (node.dist > d_bound) d_bound = node.dist;
              sum_dist += node.dist;
              sum_degree += social_graph_->get_degree(node.id);
              const auto& nbs = social_graph_->get_neighbors(node.id);
              for (int nb : nbs) coverage_map[nb]++;
          }

          float lambda = d_bound * (1.0f + epsilon);
          float avg_dist = sum_dist / (float)S.size();
          float avg_degree = sum_degree / (float)S.size();

          float theta = 0.0f;
          if (std::abs(avg_dist) < 1e-9) theta = 1e9f;
          else theta = beta * (avg_degree / avg_dist);

          std::deque<int> queue;

          for (const auto& node : S) {
              size_t *ll = (size_t *)GetLinkByInternalID(node.id, 0);
              size_t num = *ll;
              int *phys_nbs = (int *)(ll + 1);
              for (size_t i = 0; i < num; ++i) {
                  int nb_id = phys_nbs[i];
                  if (nb_id < 0 || nb_id >= cur_element_count_) continue;
                  if (visited.find(nb_id) == visited.end()) {
                      float d = vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_);
                      if (d <= lambda) {
                          queue.push_back(nb_id);
                          visited.insert(nb_id);
                      }
                  }
              }
          }

          for (size_t i = K; i < all_fetched.size(); ++i) {
              int id = all_fetched[i].id;
              if (visited.find(id) == visited.end() && all_fetched[i].dist <= lambda) {
                  queue.push_back(id);
                  visited.insert(id);
              }
          }


          while (!queue.empty()) {
              int c_id = queue.front();
              queue.pop_front();

              float d_c = vec_L2sqr(query_data, GetDataByInternalID(c_id), D_);
              if (d_c > lambda) continue;

              int gain_c = 0;
              const auto& c_nbs = social_graph_->get_neighbors(c_id);
              for (int nb : c_nbs) {
                  if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_c++;
              }

              int best_victim_idx = -1;
              float max_score = -1e9f;
              bool do_swap_1v1 = false;

              for (int i = 0; i < S.size(); ++i) {
                  float d_v = S[i].dist;
                  int v_id = S[i].id;

                  int loss_v = 0;
                  const auto& v_nbs = social_graph_->get_neighbors(v_id);
                  for (int nb : v_nbs) if (coverage_map[nb] == 1) loss_v++;

                  bool better = false;

                  if (d_c <= d_v && gain_c > loss_v) better = true;

                  else if (d_c > d_v && d_c <= lambda) {
                      float gd = (float)(gain_c - loss_v);
                      float dd = d_c - d_v;
                      if (dd < 1e-9) dd = 1e-9;
                      if (gd > 0 && (gd / dd) > theta) better = true;
                  }

                  if (better) {
                      float score = (float)(gain_c - loss_v);
                      if (score > max_score) {
                          max_score = score;
                          best_victim_idx = i;
                          do_swap_1v1 = true;
                      }
                  }
              }

              bool do_swap_2v2 = false;
              int v1_idx = -1, v2_idx = -1;
              int filler_id = -1;
              float filler_dist = 0.0f;

              if (!do_swap_1v1 && S.size() >= 2) {

                  for (const auto& node : all_fetched) {
                      if (in_S_set.find(node.id) == in_S_set.end() && node.id != c_id) {
                          filler_id = node.id;
                          filler_dist = node.dist;
                          break;
                      }
                  }

                  if (filler_id != -1 && filler_dist <= lambda) {

                      int gain_filler = 0;
                      const auto& f_nbs = social_graph_->get_neighbors(filler_id);
                      for (int nb : f_nbs) {
                          if (coverage_map.find(nb) == coverage_map.end() || coverage_map[nb] == 0) gain_filler++;
                      }


                      std::vector<std::pair<int, int>> victims_sorted;
                      for(int i=0; i<S.size(); ++i) {
                          int loss = 0;
                          const auto& nbs = social_graph_->get_neighbors(S[i].id);
                          for(int nb : nbs) if(coverage_map[nb]==1) loss++;
                          victims_sorted.push_back({loss, i});
                      }

                      std::sort(victims_sorted.begin(), victims_sorted.end());

                      v1_idx = victims_sorted[0].second;
                      v2_idx = victims_sorted[1].second;
                      int total_loss = victims_sorted[0].first + victims_sorted[1].first;

                      float dist_in = d_c + filler_dist;
                      float dist_out = S[v1_idx].dist + S[v2_idx].dist;
                      int gain_in = gain_c + gain_filler;

                      float gd = (float)(gain_in - total_loss);
                      float dd = dist_in - dist_out;

                      if (dd <= 0 && gd > 0) do_swap_2v2 = true;

                      else if (dd > 0 && gd > 0) {
                          if (dd < 1e-9) dd = 1e-9;
                          if ((gd / dd) > theta) do_swap_2v2 = true;
                      }
                  }
              }

              if (do_swap_1v1) {
                  int v_id = S[best_victim_idx].id;

                  in_S_set.erase(v_id); in_S_set.insert(c_id);
                  S[best_victim_idx] = {c_id, d_c};

                  for (int nb : social_graph_->get_neighbors(v_id)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;


                  size_t *ll = (size_t *)GetLinkByInternalID(c_id, 0);
                  size_t num = *ll;
                  int *phys_nbs = (int *)(ll + 1);
                  for(size_t i=0; i<num; ++i) {
                      int nb_id = phys_nbs[i];
                      if(visited.find(nb_id) == visited.end() &&
                              vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_) <= lambda) {
                          queue.push_back(nb_id); visited.insert(nb_id);
                      }
                  }

              } else if (do_swap_2v2) {

                  int vid1 = S[v1_idx].id;
                  int vid2 = S[v2_idx].id;

                  in_S_set.erase(vid1); in_S_set.erase(vid2);
                  in_S_set.insert(c_id); in_S_set.insert(filler_id);

                  S[v1_idx] = {c_id, d_c};
                  S[v2_idx] = {filler_id, filler_dist};


                  for (int nb : social_graph_->get_neighbors(vid1)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(vid2)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;
                  for (int nb : social_graph_->get_neighbors(filler_id)) coverage_map[nb]++;


                  int new_ids[2] = {c_id, filler_id};
                  for(int new_id : new_ids) {
                      size_t *ll = (size_t *)GetLinkByInternalID(new_id, 0);
                      size_t num = *ll;
                      int *phys_nbs = (int *)(ll + 1);
                      for(size_t i=0; i<num; ++i) {
                          int nb_id = phys_nbs[i];
                          if(visited.find(nb_id) == visited.end() &&
                                  vec_L2sqr(query_data, GetDataByInternalID(nb_id), D_) <= lambda) {
                              queue.push_back(nb_id); visited.insert(nb_id);
                          }
                      }
                  }
              }


              if (do_swap_1v1 || do_swap_2v2) {
                  d_bound = 0.0f;
                  for (const auto& node : S) d_bound = std::max(d_bound, node.dist);
                  lambda = d_bound * (1.0f + epsilon);
              }
          }

          std::vector<int> result_ids;
          for (const auto& node : S) result_ids.push_back(node.id);
          return result_ids;
      }



      std::vector<int> SearchIterative_Strategy_final(const vdim_t* query_data, size_t K, float epsilon, float target_recall)
      {
          if (!social_graph_) return {};

          auto proxy_pq = Search(query_data, K, 300);

          struct Node { int id; float dist; };

          std::vector<Node> proxy_nodes;
          proxy_nodes.reserve(K);
          std::unordered_set<int> proxy_set;

          while (!proxy_pq.empty()) {
              int id = (int)proxy_pq.top().second;
              float dist = proxy_pq.top().first;
              proxy_pq.pop();

              proxy_nodes.push_back({id, dist});
              proxy_set.insert(id);
          }

          std::reverse(proxy_nodes.begin(), proxy_nodes.end());

          int min_overlap = (int)std::ceil((float)K * target_recall);

          int max_replace_proxy = K - min_overlap;

          std::vector<Node> S = proxy_nodes;
          std::unordered_set<int> in_S_set;
          for (const auto& node : S) in_S_set.insert(node.id);

          if (S.empty()) return {};

          std::unordered_map<int, int> coverage_map;
          float d_bound = 0.0f;

          for (const auto& node : S) {
              d_bound = std::max(d_bound, node.dist);
              for (int nb : social_graph_->get_neighbors(node.id)) {
                  coverage_map[nb]++;
              }
          }

          float lambda = d_bound * (1.0f + epsilon);

          float theta = 1.0f;

          size_t L_fetch = K * 3;
          size_t ef_fetch = std::max((size_t)100, L_fetch * 2);

          std::vector<int> fetched_ids = SearchDynamicAlpha(query_data, L_fetch, ef_fetch, 200.0f, 1);

          std::vector<Node> all_fetched;
          all_fetched.reserve(fetched_ids.size());

          std::unordered_set<int> unique_ids;
          for (const auto& node : S) unique_ids.insert(node.id);

          for (int id : fetched_ids) {
              if (unique_ids.find(id) == unique_ids.end()) {
                  float dist = vec_L2sqr(query_data, GetDataByInternalID(id), D_);
                  all_fetched.push_back({id, dist});
                  unique_ids.insert(id);
              }
          }

          std::sort(all_fetched.begin(), all_fetched.end(), [](const Node& a, const Node& b){
              return a.dist < b.dist;
          });

          std::deque<int> queue;
          std::unordered_set<int> visited;
          for (const auto& node : S) visited.insert(node.id);

          for (const auto& node : S) {
              size_t *ll = (size_t *)GetLinkByInternalID(node.id, 0);
              int *phys_nbs = (int *)(ll + 1);

              for (size_t i = 0; i < *ll; ++i) {
                  int nb = phys_nbs[i];
                  if (nb >= 0 && nb < (int)cur_element_count_ && visited.find(nb) == visited.end()) {
                      if (vec_L2sqr(query_data, GetDataByInternalID(nb), D_) <= lambda) {
                          queue.push_back(nb);
                          visited.insert(nb);
                      }
                  }
              }
          }

          for (const auto& node : all_fetched) {
              if (visited.find(node.id) == visited.end() && node.dist <= lambda) {
                  queue.push_back(node.id);
                  visited.insert(node.id);
              }
          }

          while (!queue.empty()) {
              int c_id = queue.front();
              queue.pop_front();

              if (c_id < 0 || c_id >= (int)cur_element_count_) continue;

              float d_c = vec_L2sqr(query_data, GetDataByInternalID(c_id), D_);
              if (d_c > lambda) continue;

              int gain_c = 0;
              for (int nb : social_graph_->get_neighbors(c_id)) {
                  if (coverage_map[nb] == 0) gain_c++;
              }

              int current_overlap = 0;
              for (const auto& node : S) {
                  if (proxy_set.count(node.id)) current_overlap++;
              }

              bool is_c_in_proxy = proxy_set.count(c_id) > 0;

              int best_vic = -1;
              float max_score = -1e9f;
              bool do_1v1 = false;

              for (int i = 0; i < (int)S.size(); ++i) {
                  int v_id = S[i].id;
                  bool v_in_proxy = proxy_set.count(v_id) > 0;

                  int potential_overlap = current_overlap - (v_in_proxy ? 1 : 0) + (is_c_in_proxy ? 1 : 0);
                  int proxy_replace_count = (v_in_proxy ? 1 : 0) - (is_c_in_proxy ? 1 : 0);

                  if (potential_overlap < min_overlap || proxy_replace_count > max_replace_proxy) {
                      continue;
                  }

                  float d_v = S[i].dist;
                  int loss_v = 0;
                  for (int nb : social_graph_->get_neighbors(v_id)) {
                      if (coverage_map[nb] == 1) loss_v++;
                  }

                  bool better = false;
                  if (d_c <= d_v && gain_c > loss_v) {
                      better = true;
                  } else if (d_c > d_v && d_c <= lambda) {
                      float dd = std::max(1e-9f, d_c - d_v);
                      if ((gain_c - loss_v) > 0 && ((gain_c - loss_v) / dd) > theta) {
                          better = true;
                      }
                  }

                  if (better && (gain_c - loss_v) > max_score) {
                      max_score = (float)(gain_c - loss_v);
                      best_vic = i;
                      do_1v1 = true;
                  }
              }

              bool do_2v2 = false;
              int v1 = -1, v2 = -1, filler = -1;
              float f_dist = 0.0f;

              if (!do_1v1 && S.size() >= 2) {

                  for (const auto& node : all_fetched) {
                      if (in_S_set.find(node.id) == in_S_set.end() && node.id != c_id) {
                          filler = node.id;
                          f_dist = node.dist;
                          break;
                      }
                  }

                  if (filler != -1 && f_dist <= lambda) {
                      bool is_f_in_proxy = proxy_set.count(filler) > 0;
                      int gain_f = 0;
                      for (int nb : social_graph_->get_neighbors(filler)) {
                          if (coverage_map[nb] == 0) gain_f++;
                      }

                      std::vector<std::pair<int, int>> vics;
                      for (int i = 0; i < (int)S.size(); ++i) {
                          int loss = 0;
                          for (int nb : social_graph_->get_neighbors(S[i].id)) {
                              if (coverage_map[nb] == 1) loss++;
                          }
                          vics.push_back({loss, i});
                      }
                      std::sort(vics.begin(), vics.end());
                      v1 = vics[0].second;
                      v2 = vics[1].second;

                      int v1_in_proxy = proxy_set.count(S[v1].id) ? 1 : 0;
                      int v2_in_proxy = proxy_set.count(S[v2].id) ? 1 : 0;
                      int potential_overlap_2v2 = current_overlap - v1_in_proxy - v2_in_proxy
                              + (is_c_in_proxy ? 1 : 0) + (is_f_in_proxy ? 1 : 0);
                      int proxy_replace_count_2v2 = (v1_in_proxy + v2_in_proxy) - ((is_c_in_proxy ? 1 : 0) + (is_f_in_proxy ? 1 : 0));

                      if (potential_overlap_2v2 >= min_overlap && proxy_replace_count_2v2 <= max_replace_proxy) {
                          float dd = (d_c + f_dist) - (S[v1].dist + S[v2].dist);
                          float gd = gain_c + gain_f - vics[0].first - vics[1].first;
                          if ((dd <= 0 && gd > 0) || (dd > 0 && gd > 0 && (gd / std::max(1e-9f, dd)) > theta)) {
                              do_2v2 = true;
                          }
                      }
                  }
              }

              if (do_1v1) {
                  int v_id = S[best_vic].id;
                  in_S_set.erase(v_id);
                  in_S_set.insert(c_id);
                  S[best_vic] = {c_id, d_c};

                  for (int nb : social_graph_->get_neighbors(v_id)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;

                  d_bound = 0.0f;
                  for (const auto& node : S) {
                      d_bound = std::max(d_bound, node.dist);
                  }
                  lambda = d_bound * (1.0f + epsilon);

                  size_t *ll = (size_t *)GetLinkByInternalID(c_id, 0);
                  int *phys_nbs = (int *)(ll + 1);
                  for (size_t i = 0; i < *ll; ++i) {
                      int nb = phys_nbs[i];
                      if (nb >= 0 && nb < (int)cur_element_count_ && visited.find(nb) == visited.end()) {
                          if (vec_L2sqr(query_data, GetDataByInternalID(nb), D_) <= lambda) {
                              queue.push_back(nb);
                              visited.insert(nb);
                          }
                      }
                  }
              } else if (do_2v2) {
                  int vid1 = S[v1].id;
                  int vid2 = S[v2].id;
                  in_S_set.erase(vid1);
                  in_S_set.erase(vid2);
                  in_S_set.insert(c_id);
                  in_S_set.insert(filler);
                  S[v1] = {c_id, d_c};
                  S[v2] = {filler, f_dist};

                  for (int nb : social_graph_->get_neighbors(vid1)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(vid2)) coverage_map[nb]--;
                  for (int nb : social_graph_->get_neighbors(c_id)) coverage_map[nb]++;
                  for (int nb : social_graph_->get_neighbors(filler)) coverage_map[nb]++;

                  d_bound = 0.0f;
                  for (const auto& node : S) {
                      d_bound = std::max(d_bound, node.dist);
                  }
                  lambda = d_bound * (1.0f + epsilon);

                  int nids[] = {c_id, filler};
                  for (int nid : nids) {
                      size_t *ll = (size_t *)GetLinkByInternalID(nid, 0);
                      int *phys_nbs = (int *)(ll + 1);
                      for (size_t i = 0; i < *ll; ++i) {
                          int nb = phys_nbs[i];
                          if (nb >= 0 && nb < (int)cur_element_count_ && visited.find(nb) == visited.end()) {
                              if (vec_L2sqr(query_data, GetDataByInternalID(nb), D_) <= lambda) {
                                  queue.push_back(nb);
                                  visited.insert(nb);
                              }
                          }
                      }
                  }
              }
          }

          std::vector<int> res;
          for (const auto& node : S) res.push_back(node.id);
          return res;
      }



      int GetRandomLevel(double reverse_size)
      {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
      }

      std::priority_queue<std::pair<float, id_t>> SearchBaseLayer(
          id_t ep_id,
          const vdim_t *data_point,
          int level,
          size_t ef)
      {
        size_t comparison = 0;

        // auto vl = visited_list_pool_->GetFreeVisitedList();
        // auto mass_visited = vl->mass_.data();
        // auto curr_visited = vl->curr_visited_;

        auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        float dist = vec_L2sqr(data_point, GetDataByInternalID(ep_id), D_);
        comparison++;

        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        // mass_visited[ep_id] = curr_visited;
        mass_visited->at(ep_id) = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();
          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;

          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;

          std::unique_lock<std::mutex> lock((*link_list_locks_)[curr_node_id]);

          size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id, level);
          size_t num_neighbors = *ll_cur;
          id_t *neighbors = (id_t *)(ll_cur + 1);

          // #if defined(__SSE__)
          //   /// @brief Prefetch cache lines to speed up cpu caculation.
          //   _mm_prefetch((char *) (mass_visited + *neighbors), _MM_HINT_T0);
          //   _mm_prefetch((char *) (mass_visited + *neighbors + 64), _MM_HINT_T0);
          //   _mm_prefetch((char *) (GetDataByInternalID(*neighbors)), _MM_HINT_T0);
          // #endif

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neighbor_id = neighbors[j];

            // #if defined(__SSE__)
            //   _mm_prefetch((char *) (mass_visited + *(neighbors + j + 1)), _MM_HINT_T0);
            //   _mm_prefetch((char *) (GetDataByInternalID(*(neighbors + j + 1))), _MM_HINT_T0);
            // #endif

            if (mass_visited->at(neighbor_id) == false)
            {
              // mass_visited[neighbor_id] = curr_visited;
              mass_visited->at(neighbor_id) = true;

              float dist = vec_L2sqr(data_point, GetDataByInternalID(neighbor_id), D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                // #if defined(__SSE__)
                //   _mm_prefetch((char *) (GetLinkByInternalID(candidate_set.top().second, 0)), _MM_HINT_T0);
                // #endif

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        // visited_list_pool_->ReleaseVisitedList(vl);
        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      id_t GetClosestPoint(const vdim_t *data_point)
      {
        if (cur_element_count_ == 0)
          throw std::runtime_error("empty graph");
        id_t wander = enterpoint_node_;
        size_t comparison = 0;

        float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_);
        comparison++;

        for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
        {
          bool moving = true;
          while (moving)
          {
            moving = false;
            size_t *ll = (size_t *)GetLinkByInternalID(wander, lev);
            size_t sz = *ll;
            id_t *adj = (id_t *)(ll + 1);
            for (size_t i = 0; i < sz; i++)
            {
              id_t cand = adj[i];
              float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
              if (d < dist)
              {
                dist = d;
                wander = cand;
                moving = true;
              }
            }
            comparison += sz;
          }
        }
        comparison_.fetch_add(comparison);
        return wander;
      }

      std::vector<float> GetSearchLengthLevel0(id_t ep_id, const vdim_t *query_data, size_t k, size_t ef, std::priority_queue<std::pair<float, id_t>> &top_candidates)
      {
        std::vector<float> length;

        size_t comparison = 0;

        // auto vl = visited_list_pool_->GetFreeVisitedList();
        // auto mass_visited = vl->mass_.data();
        // auto curr_visited = vl->curr_visited_;

        auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

        // std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        float dist = vec_L2sqr(query_data, GetDataByInternalID(ep_id), D_);
        comparison++;

        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        // mass_visited[ep_id] = curr_visited;
        mass_visited->at(ep_id) = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();

          length.emplace_back(-curr_el_pair.first);

          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;

          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;

          std::unique_lock<std::mutex> lock((*link_list_locks_)[curr_node_id]);

          size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id, 0);
          size_t num_neighbors = *ll_cur;
          id_t *neighbors = (id_t *)(ll_cur + 1);

          // #if defined(__SSE__)
          //   /// @brief Prefetch cache lines to speed up cpu caculation.
          //   _mm_prefetch((char *) (mass_visited + *neighbors), _MM_HINT_T0);
          //   _mm_prefetch((char *) (mass_visited + *neighbors + 64), _MM_HINT_T0);
          //   _mm_prefetch((char *) (GetDataByInternalID(*neighbors)), _MM_HINT_T0);
          // #endif

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neighbor_id = neighbors[j];

            // #if defined(__SSE__)
            //   _mm_prefetch((char *) (mass_visited + *(neighbors + j + 1)), _MM_HINT_T0);
            //   _mm_prefetch((char *) (GetDataByInternalID(*(neighbors + j + 1))), _MM_HINT_T0);
            // #endif

            if (mass_visited->at(neighbor_id) == false)
            {
              mass_visited->at(neighbor_id) = true;

              float dist = vec_L2sqr(query_data, GetDataByInternalID(neighbor_id), D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                // #if defined(__SSE__)
                //   _mm_prefetch((char *) (GetLinkByInternalID(candidate_set.top().second, 0)), _MM_HINT_T0);
                // #endif

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        // visited_list_pool_->ReleaseVisitedList(vl);
        comparison_.fetch_add(comparison);

        return length;
      }

      std::vector<std::vector<float>> GetSearchLength(
          const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
      {
        size_t nq = queries.size();
        vids.clear();
        dists.clear();
        vids.resize(nq);
        dists.resize(nq);

        std::vector<std::vector<float>> lengths(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = vids[i];
          auto &dist = dists[i];

          std::priority_queue<std::pair<float, id_t>> r;

          lengths[i] = GetSearchLength(query.data(), k, ef, r);
          vid.reserve(r.size());
          dist.reserve(r.size());
          while (r.size())
          {
            const auto &te = r.top();
            vid.emplace_back(te.second);
            dist.emplace_back(te.first);
            r.pop();
          }
        }

        return lengths;
      }

      std::vector<float> GetSearchLength(const vdim_t *query_data, size_t k, size_t ef, std::priority_queue<std::pair<float, id_t>> &top_candidates)
      {
        assert(ready_ && "Index uninitialized!");

        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::vector<float>();

        std::vector<float> length;

        size_t comparison = 0;

        id_t cur_obj = enterpoint_node_;
        float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          if (length.size())
            length.pop_back();
          bool changed = true;
          while (changed)
          {
            length.emplace_back(cur_dist);
            changed = false;
            size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
            size_t num_neighbors = *ll_cur;
            id_t *neighbors = (id_t *)(ll_cur + 1);

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        // std::priority_queue<std::pair<float, id_t>> top_candidates;

        auto length0 = GetSearchLengthLevel0(cur_obj, query_data, 0, ef, top_candidates);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        length.insert(length.end(), length0.begin(), length0.end());

        return length;
      }
    };

  }; // namespace graph

}; // namespace index
