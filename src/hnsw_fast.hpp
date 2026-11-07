#pragma once
// hnsw_fast.hpp —— 建图热路径优化版（新文件；**不修改** retrieval_framework.hpp）
//
// 思路：用继承代替复制。只重写两个成员——
//   AddPoint            （原版 :169-252，逐行照抄，唯一改动是调用下面的 SearchBaseLayerFast）
//   SearchBaseLayerFast （原版 SearchBaseLayer :1979-2068 的等价实现 + 两处优化）
// 其余全部继承：MutuallyConnectNewElement / GetNeighborsByHeuristic / 存储布局 / 落盘格式。
// 所以图的结构与语义必须与原版**完全一致**，验收标准是 1M 上 τ 表摘要逐位相同。
//
// 两处优化
//   1) visited 复用：原版每次调用都 make_unique<vector<bool>>(max_elements_)，等于每插入一个点
//      就把整张 N 位图清零一次 → 建图总代价被抬到 O(N²)。这里改成 epoch 标记数组，O(1) 复位。
//      代价：额外 max_elements × 4 B（100M 时 400 MB）。
//   2) 预取：邻居循环里提前把下一个邻居的链接表、下下个邻居的向量指针槽、下一个邻居的向量数据
//      拉进 cache。_mm_prefetch 只是提示，**不可能改变计算结果**。
//
// 用法： HNSWFast<u8_t> idx(D, N, 16, 200); 其余接口与 HNSW<u8_t> 相同。

#include <cstdint>
#include <cstddef>
#include <queue>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <mutex>

#include "retrieval_framework.hpp"
#include "u8vec.hpp"          // 必须在 retrieval_framework.hpp 之后

namespace fastidx {

template <typename vdim_t>
class HNSWFast : public anns::graph::HNSW<vdim_t> {
    using Base = anns::graph::HNSW<vdim_t>;
    using id_t = int;

public:
    std::vector<uint32_t> visited_epoch_;
    uint32_t cur_epoch_ = 0;

    HNSWFast(size_t D, size_t max_elements, size_t M = 16, size_t ef_construction = 128, size_t random_seed = 123)
        : Base(D, max_elements, M, ef_construction, random_seed) {
        visited_epoch_.assign(max_elements, 0u);
    }

    // 返回 true 表示本 epoch 已经访问过
    inline bool visited_before(id_t id) const { return visited_epoch_[(size_t)id] == cur_epoch_; }
    inline void visited_mark(id_t id) { visited_epoch_[(size_t)id] = cur_epoch_; }

    std::priority_queue<std::pair<float, id_t>> SearchBaseLayerFast(
        id_t ep_id, const vdim_t* data_point, int level, size_t ef) {
        size_t comparison = 0;

        // 换 epoch 等价于整张标记数组清零；回绕时（理论上 2^32 次调用才会）真的清一次
        if (++cur_epoch_ == 0u) {
            std::fill(visited_epoch_.begin(), visited_epoch_.end(), 0u);
            cur_epoch_ = 1u;
        }

        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        float dist = vec_L2sqr(data_point, this->GetDataByInternalID(ep_id), this->D_);
        comparison++;

        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
        visited_mark(ep_id);

        float low_bound = dist;
        while (candidate_set.size()) {
            auto curr_el_pair = candidate_set.top();
            if (-curr_el_pair.first > low_bound && top_candidates.size() == ef) break;

            candidate_set.pop();
            id_t curr_node_id = curr_el_pair.second;

            std::unique_lock<std::mutex> lock((*this->link_list_locks_)[curr_node_id]);

            size_t* ll_cur = (size_t*)this->GetLinkByInternalID(curr_node_id, level);
            size_t num_neighbors = *ll_cur;
            id_t* neighbors = (id_t*)(ll_cur + 1);

            for (size_t j = 0; j < num_neighbors; j++) {
                // ---- 预取（纯提示，不影响结果）----
                if (j + 2 < num_neighbors) {
                    const char* slot = (const char*)(this->data_level0_memory_.data() +
                                                     (size_t)neighbors[j + 2] * this->size_data_per_element_ +
                                                     this->offset_data_);
                    _mm_prefetch(slot, _MM_HINT_T0);
                    _mm_prefetch((const char*)this->GetLinkByInternalID(neighbors[j + 2], level), _MM_HINT_T0);
                }
                if (j + 1 < num_neighbors) {
                    const char* p = (const char*)this->GetDataByInternalID(neighbors[j + 1]);
                    _mm_prefetch(p, _MM_HINT_T0);
                    _mm_prefetch(p + 64, _MM_HINT_T0);   // 128 维 uint8 = 128 B = 两条 cache line
                }

                id_t neighbor_id = neighbors[j];
                if (visited_before(neighbor_id)) continue;
                visited_mark(neighbor_id);

                float d = vec_L2sqr(data_point, this->GetDataByInternalID(neighbor_id), this->D_);
                comparison++;

                if (top_candidates.top().first > d || top_candidates.size() < ef) {
                    candidate_set.emplace(-d, neighbor_id);
                    top_candidates.emplace(d, neighbor_id);
                    if (top_candidates.size() > ef) top_candidates.pop();
                    if (top_candidates.size()) low_bound = top_candidates.top().first;
                }
            }
        }
        this->comparison_.fetch_add(comparison);
        return top_candidates;
    }

    // 与 HNSW::AddPoint 逐行等价，唯一区别是 lev 循环里调用 SearchBaseLayerFast
    id_t AddPoint(const vdim_t* data_point) {
        if (this->cur_element_count_ >= this->max_elements_) {
            std::cerr << "The number of elements exceeds the specified limit" << std::endl;
            exit(1);
        }

        id_t cur_id;
        int cur_level;
        {
            std::unique_lock<std::mutex> temp_lock(this->global_);
            cur_level = this->GetRandomLevel(this->mult_);
            cur_id = this->cur_element_count_++;
        }

        std::unique_lock<std::mutex> lock_el((*this->link_list_locks_)[cur_id]);

        this->element_levels_[cur_id] = cur_level;

        std::unique_lock<std::mutex> temp_lock(this->global_);
        int max_level_copy = this->max_level_;
        id_t cur_obj = this->enterpoint_node_;
        id_t enterpoint_node_copy = this->enterpoint_node_;
        if (cur_level <= this->max_level_) temp_lock.unlock();

        this->WriteDataByInternalID(cur_id, data_point);

        if (cur_level) {
            this->link_lists_[cur_id].resize(this->size_links_per_element_ * cur_level, 0x00);
        }

        if (enterpoint_node_copy != -1) {
            if (cur_level < max_level_copy) {
                float cur_dist = vec_L2sqr(data_point, this->GetDataByInternalID(cur_obj), this->D_);
                for (int lev = max_level_copy; lev > cur_level; lev--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> wlock(this->link_list_locks_->at(cur_obj));
                        size_t* ll_cur = (size_t*)this->GetLinkByInternalID(cur_obj, lev);
                        size_t num_neighbors = *ll_cur;
                        id_t* neighbors = (id_t*)(ll_cur + 1);

                        for (size_t i = 0; i < num_neighbors; i++) {
                            id_t cand = neighbors[i];
                            float d = vec_L2sqr(data_point, this->GetDataByInternalID(cand), this->D_);
                            if (d < cur_dist) {
                                cur_dist = d;
                                cur_obj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            for (int lev = std::min(cur_level, max_level_copy); lev >= 0; lev--) {
                auto top_candidates = SearchBaseLayerFast(cur_obj, data_point, lev, this->ef_construction_);
                cur_obj = this->MutuallyConnectNewElement(data_point, cur_id, top_candidates, lev);
            }
        } else {
            this->enterpoint_node_ = cur_id;
            this->max_level_ = cur_level;
        }

        if (cur_level > max_level_copy) {
            this->enterpoint_node_ = cur_id;
            this->max_level_ = cur_level;
        }
        return cur_id;
    }
};

}  // namespace fastidx
