#include "fry/types.hpp"
#include "fry/vector_store.h"

#include <iostream>

auto main() -> int {
    fry::VectorStore store(3);
    const fry::VectorId id0 = store.insert(fry::Vector<float>{1.0f, 0.0f, 0.0f}).value();
    const fry::VectorId id1 = store.insert(fry::Vector<float>{0.0f, 1.0f, 0.0f}).value();
    const fry::VectorId id2 = store.insert(fry::Vector<float>{0.0f, 0.0f, 1.0f}).value();

    std::cout << "Inserted ids: " << id0 << ", " << id1 << ", " << id2 << '\n';

    // search() also returns Result<VectorId>.
    const auto result = store.search(fry::Vector<>{0.9F, 0.1F, 0.0F});
    if (!result) {
        std::cerr << "Search failed: " << result.error().message << '\n';
        return 1;
    }
    std::cout << "Nearest neighbour id: " << *result << '\n'; // expected: 0
    return 0;
}
