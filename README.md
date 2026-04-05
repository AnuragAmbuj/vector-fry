# fry-vector

A vector database written from scratch in C++11 — because **I don't know what I cannot build.**

I also just wanted a reason to write C++ again. It's been about 15 years.


## What is it?

fry-vector is a fully hand-rolled vector database. No shortcuts, no wrappers around existing ANN libraries. Everything from the memory layout to the graph index is written from the ground up.

When it's done, it will support:

- Exact brute-force search (`FlatIndex`) and approximate search (`HNSW`)
- Three distance metrics — L2, Cosine, Inner Product
- SIMD-accelerated distance kernels (ARM NEON on M-series, AVX2+FMA on x86)
- Persistence with a Write-Ahead Log and memory-mapped segments
- An HTTP API for inserts, queries, and filtered search
- Thread-safe reads and writes

Right now Phases 1–3 are done. The core types, flat index, all three metrics, and dual-platform SIMD are working and tested. Phase 4 (HNSW) is in progress.

See [PLAN.md](./PLAN.md) for the full roadmap.


## Why build this?

I wanted to understand vector databases from the inside — not just use one. Reading papers and docs only gets you so far. Building it yourself is the only way to really understand why the memory layout matters, why HNSW converges, and why approximate search is a tradeoff and not a bug.

The C++ part is secondary motivation. I wanted to write something real in a language I used to know well and see how much had changed. Turns out, quite a bit.


## Building

```bash
cmake --preset asan    # debug + AddressSanitizer
cmake --preset release # optimised build
./run-tests.sh         # build and run all tests
```

**My setup:**
- MacBook Pro M3 16GB — primary dev machine, ARM64/NEON
- Linux box (Intel i7, AMD64) — secondary build target, AVX2
- Linux container (AARCH64, 2GB) — CI-style sanity checks
- CLion + clang-tidy + clang-format

No Windows machine. You're welcome to try.

**Dependencies** (via vcpkg):
- `nlohmann-json` — for segment manifests and the HTTP API payload
- `spdlog` — structured logging
- `catch2` — tests


## How I work on it

- I write the implementation. No AI writes code for me — I'm still learning and that's the point.
- I use Claude for research, code reviews, and generating basic skeleton and test cases before I start each feature (TDD).
- Tests come first. If the tests pass and the code is clean, the feature is done.

If you want to follow along or implement something yourself, pull the repo and check PLAN.md. Each phase has clear goals and the test suite tells you when you're done.


## Why C/C++ ?

**C/C++** — fast, mature, and well understood. I wanted to write real systems code without fighting the language, and C++ lets me do that. Memory safety is on me, which is exactly the point. A great programmer handles memory well and I intend to become one.

**Not Rust** — still maturing. The ecosystem is good but a lot of open source Rust code right now is AI written. It also feels like the language is still figuring itself out.

**Not Zig** — by the time it stabilizes, the window will have closed. Likely ends up where D did but for a different reason.

**Others** - Personally, I don't find others a good candidate to implement a vector database. If they are good, thier syntax is unintuitive for me, or if their syntax is intuitive, they suffer from performance issues.

## Developer Blog

If you want to understand what's being built and why, there's a blog series in [`docs/blogs/`](./docs/blogs/) that covers each phase — from "what is a vector database" to SIMD intrinsics. Written for readers of all levels.

*C++11 · CMake · Catch2 · clang-tidy · ARM NEON · x86 AVX2*
