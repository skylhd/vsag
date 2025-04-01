
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

#include <vsag/vsag.h>

#include <iostream>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Create HNSW Index *****************/
    auto hnsw_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_parameters).value();

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Hnsw contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: internalError" << build_result.error().message
                  << std::endl;
        exit(-1);
    }

    /******************* Prepare Query *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* Prepare Bitset Filter *****************/
    auto filter_bitset = vsag::Bitset::Make();
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto id = base->GetIds()[i];
        if (id % 2 == 0) {
            filter_bitset->Set(id);
        }
    }

    /******************* Prepare Filter Function *****************/
    std::function<bool(int64_t)> filter_func = [](int64_t id) { return id % 2 == 0; };

    /******************* Prepare Filter Object *****************/
    class MyFilter : public vsag::Filter {
    public:
        bool
        CheckValid(int64_t id, bool is_inner_id) const override {
            return id % 2;
        }

        float
        ValidRatio() const override {
            return 0.618f;
        }
    };
    auto filter_object = std::make_shared<MyFilter>();

    /******************* HNSW Filter Search With Bitset *****************/
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto search_result = index->KnnSearch(query, topk, hnsw_search_parameters, filter_bitset);
    if (not search_result.has_value()) {
        std::cerr << "Failed to search index with filter" << search_result.error().message
                  << std::endl;
        exit(-1);
    }
    auto result = search_result.value();

    // print result with filter, the result id is odd not even.
    std::cout << "bitset filter results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    /******************* HNSW Filter Search With filter function *****************/
    search_result = index->KnnSearch(query, topk, hnsw_search_parameters, filter_func);
    if (not search_result.has_value()) {
        std::cerr << "Failed to search index with filter" << search_result.error().message
                  << std::endl;
        exit(-1);
    }
    result = search_result.value();

    // print result with filter, the result id is odd not even.
    std::cout << "function filter results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    /******************* HNSW Filter Search With Filter Object *****************/
    search_result = index->KnnSearch(query, topk, hnsw_search_parameters, filter_object);
    if (not search_result.has_value()) {
        std::cerr << "Failed to search index with filter" << search_result.error().message
                  << std::endl;
        exit(-1);
    }
    result = search_result.value();

    // print result with filter, the result id is odd not even.
    std::cout << "object filter results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}
