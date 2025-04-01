
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

#include "data_cell/extra_info_datacell_parameter.h"
#include "data_cell/flatten_datacell_parameter.h"
#include "data_cell/graph_interface_parameter.h"
#include "parameter.h"

namespace vsag {

class HGraphParameter : public Parameter {
public:
    explicit HGraphParameter(const JsonType& json);

    HGraphParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    FlattenDataCellParamPtr base_codes_param{nullptr};
    FlattenDataCellParamPtr precise_codes_param{nullptr};
    GraphInterfaceParamPtr bottom_graph_param{nullptr};
    ExtraInfoDataCellParamPtr extra_info_param{nullptr};

    bool use_reorder{false};
    bool ignore_reorder{false};
    uint64_t ef_construction{400};
    uint64_t build_thread_count{100};

    std::string name;
};

using HGraphParameterPtr = std::shared_ptr<HGraphParameter>;

class HGraphSearchParameters {
public:
    static HGraphSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    bool use_reorder{false};
    float skip_ratio{0.7f};
    bool use_inner_id_filter{false};
private:
    HGraphSearchParameters() = default;
};

}  // namespace vsag
