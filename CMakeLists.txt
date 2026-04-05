cmake_minimum_required(VERSION 3.25)
project(fry_vector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── vcpkg dependencies ────────────────────────────────────────────────────────
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog        CONFIG REQUIRED)

# ── Compiler warnings ────────────────────────────────────────────────────────
add_library(fry_warnings INTERFACE)
target_compile_options(fry_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow>
)

# ── SIMD detection ───────────────────────────────────────────────────────────
# Detects architecture at configure time and enables the right SIMD flags.
# ARM64 (Apple M-series): NEON is always present — no flag needed, but we
#   define FRY_HAVE_NEON so code can #ifdef against it.
# x86_64 (Intel/AMD): AVX2 is optional — -mavx2 -mfma enables the intrinsics
#   and defines __AVX2__ which our headers use for compile-time dispatch.
add_library(fry_simd INTERFACE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "fry-vector: ARM64 detected — NEON enabled")
    target_compile_definitions(fry_simd INTERFACE FRY_HAVE_NEON=1)
    # -mfpu=neon is implicit on arm64; adding it would be a warning on AppleClang
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "fry-vector: x86_64 detected — enabling AVX2 + FMA")
    target_compile_options(fry_simd INTERFACE -mavx2 -mfma)
    target_compile_definitions(fry_simd INTERFACE FRY_HAVE_AVX2=1)
else()
    message(STATUS "fry-vector: Unknown architecture — SIMD disabled, scalar only")
endif()

# ── AddressSanitizer helper ──────────────────────────────────────────────────
add_library(fry_asan INTERFACE
        include/fry/flat_index.hpp)
if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    target_compile_options(fry_asan INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(fry_asan INTERFACE -fsanitize=address,undefined)
endif()

# ── Library target ───────────────────────────────────────────────────────────
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_library(fry_vector_lib STATIC ${LIB_SOURCES}
        include/fry/distance.hpp
        include/fry/flat_index.hpp
        include/fry/distance_neon.hpp
        include/fry/distance_avx2.hpp
        include/fry/distance_simd.hpp
        include/fry/hnsw_index.hpp)
target_include_directories(fry_vector_lib PUBLIC include)
target_link_libraries(fry_vector_lib
    PUBLIC
        fry_warnings
        fry_asan
        fry_simd
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(fry_vector main.cpp)
target_link_libraries(fry_vector PRIVATE fry_vector_lib)

# ── Tests ────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
