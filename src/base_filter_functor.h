
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

#pragma once

#include <functional>

#include "bitset_impl.h"
#include "common.h"
#include "label_table.h"
#include "typing.h"
#include "vsag/filter.h"

namespace vsag {

class UniqueFilter : public Filter {
public:
    UniqueFilter(const std::function<bool(int64_t)>& fallback_func)
        : fallback_func_(fallback_func), is_bitset_filter_(false){};

    UniqueFilter(const BitsetPtr& bitset) : bitset_(bitset), is_bitset_filter_(true){};

    [[nodiscard]] bool
    CheckValid(int64_t id, bool use_inner_id = false) const override {
        if (is_bitset_filter_) {
            int64_t bit_index = id & ROW_ID_MASK;
            return not bitset_->Test(bit_index);
        } else {
            return not fallback_func_(id);
        }
    }

private:
    std::function<bool(int64_t)> fallback_func_{nullptr};
    const BitsetPtr bitset_{nullptr};
    const bool is_bitset_filter_{false};
};

class CommonInnerIdFilter : public Filter {
public:
    CommonInnerIdFilter(const FilterPtr filter_impl, const LabelTable& label_table)
        : filter_impl_(filter_impl), label_table_(label_table){};

    [[nodiscard]] bool
    CheckValid(int64_t inner_id, bool use_inner_id = false) const override {
        return filter_impl_->CheckValid(use_inner_id ? inner_id : label_table_.GetLabelById(inner_id), use_inner_id);
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_impl_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_impl_->FilterDistribution();
    }

private:
    const FilterPtr filter_impl_;
    const LabelTable& label_table_;
};

}  // namespace vsag
