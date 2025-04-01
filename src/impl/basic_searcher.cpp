
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "basic_searcher.h"

#include <limits>

#include "utils/linear_congruential_generator.h"

namespace vsag {

BasicSearcher::BasicSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)) {
}

uint32_t
BasicSearcher::visit(const GraphInterfacePtr& graph,
                     const VisitedListPtr& vl,
                     const std::pair<float, uint64_t>& current_node_pair,
                     const FilterPtr& filter,
                     float skip_ratio,
                     bool use_inner_id_filter,
                     Vector<InnerIdType>& to_be_visited_rid,
                     Vector<InnerIdType>& to_be_visited_id) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> neighbors(allocator_);

    if (this->mutex_array_ != nullptr) {
        SharedLock lock(this->mutex_array_, current_node_pair.second);
        graph->GetNeighbors(current_node_pair.second, neighbors);
    } else {
        graph->GetNeighbors(current_node_pair.second, neighbors);
    }

    float skip_threshold =
        (filter != nullptr ? (1 - ((1 - filter->ValidRatio()) * skip_ratio)) : 0.0F);

    for (uint32_t i = 0; i < prefetch_jump_visit_size_ and neighbors.size() > i; i++) {
        vl->Prefetch(neighbors[i]);
    }

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_jump_visit_size_ < neighbors.size()) {
            vl->Prefetch(neighbors[i + prefetch_jump_visit_size_]);
        }
        if (not vl->Get(neighbors[i])) {
            if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                filter->CheckValid(neighbors[i], use_inner_id_filter)) {
                to_be_visited_rid[count_no_visited] = i;
                to_be_visited_id[count_no_visited] = neighbors[i];
                count_no_visited++;
            }
            vl->Set(neighbors[i]);
        }
    }
    return count_no_visited;
}

MaxHeap
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const float* query,
                      const InnerSearchParam& inner_search_param,
                      vsag::IteratorContextPtr* iter_ctx) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param, iter_ctx);
    }
    return this->search_impl<RANGE_SEARCH>(graph, flatten, vl, query, inner_search_param);
}

template <InnerSearchMode mode>
MaxHeap
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const float* query,
                           const InnerSearchParam& inner_search_param,
                           vsag::IteratorContextPtr* iter_ctx) const {
    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto use_inner_id_filter = inner_search_param.use_inner_id_filter;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    float dist = 0.0F;
    uint64_t ids_cnt = 1;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph->MaximumDegree(), allocator_);
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), allocator_);
    Vector<float> line_dists(graph->MaximumDegree(), allocator_);

    if (iter_ctx != nullptr && !(*iter_ctx)->IsFirstUsed()) {
        if ((*iter_ctx)->Empty())
            return top_candidates;
        while (!(*iter_ctx)->Empty()) {
            uint32_t cur_inner_id = (*iter_ctx)->GetTopID();
            float cur_dist = (*iter_ctx)->GetTopDist();
            if (!vl->Get(cur_inner_id) && (*iter_ctx)->CheckPoint(cur_inner_id)) {
                vl->Set(cur_inner_id);
                lower_bound = std::max(lower_bound, cur_dist);
                flatten->Query(&cur_dist, computer, &cur_inner_id, 1);
                top_candidates.emplace(cur_dist, cur_inner_id);
                candidate_set.emplace(cur_dist, cur_inner_id);
                if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                    if (cur_dist > inner_search_param.radius and not top_candidates.empty()) {
                        top_candidates.pop();
                    }
                }
            }
            (*iter_ctx)->PopDiscard();
        }
    } else {
        flatten->Query(&dist, computer, &ep, 1);
        if (not is_id_allowed || is_id_allowed->CheckValid(ep, use_inner_id_filter)) {
            top_candidates.emplace(dist, ep);
            lower_bound = top_candidates.top().first;
        }
        if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
            if (dist > inner_search_param.radius and not top_candidates.empty()) {
                top_candidates.pop();
            }
        }
        candidate_set.emplace(-dist, ep);
        vl->Set(ep);
    }

    while (not candidate_set.empty()) {
        hops++;
        auto current_node_pair = candidate_set.top();

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates.size() == ef) {
                break;
            }
        }
        candidate_set.pop();

        if (not candidate_set.empty()) {
            graph->Prefetch(candidate_set.top().second, 0);
        }

        count_no_visited = visit(graph,
                                 vl,
                                 current_node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 inner_search_param.use_inner_id_filter,
                                 to_be_visited_rid,
                                 to_be_visited_id);

        dist_cmp += count_no_visited;

        flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (top_candidates.size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                if (iter_ctx != nullptr && !(*iter_ctx)->CheckPoint(to_be_visited_id[i])) {
                    continue;
                }
                candidate_set.emplace(-dist, to_be_visited_id[i]);
                // flatten->Prefetch(candidate_set.top().second);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i], use_inner_id_filter)) {
                    top_candidates.emplace(dist, to_be_visited_id[i]);
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates.size() > ef) {
                        if (iter_ctx != nullptr &&
                            (*iter_ctx)->CheckPoint(top_candidates.top().second)) {
                            auto cur_node_pair = top_candidates.top();
                            (*iter_ctx)->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
                        }
                        top_candidates.pop();
                    }
                }

                if (not top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates.size() > inner_search_param.topk) {
            auto cur_node_pair = top_candidates.top();
            if (iter_ctx != nullptr && (*iter_ctx)->CheckPoint(cur_node_pair.second)) {
                (*iter_ctx)->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
            }
            top_candidates.pop();
        }
    } else if constexpr (mode == RANGE_SEARCH) {
        if (inner_search_param.range_search_limit_size > 0) {
            while (top_candidates.size() > inner_search_param.range_search_limit_size) {
                top_candidates.pop();
            }
        }
        while (not top_candidates.empty() &&
               top_candidates.top().first > inner_search_param.radius + THRESHOLD_ERROR) {
            top_candidates.pop();
        }
    }

    return top_candidates;
}

}  // namespace vsag
