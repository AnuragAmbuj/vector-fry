#pragma once

#include <new>        // placement new
#include <stdexcept>
#include <string>
#include <type_traits>  // std::aligned_storage
#include <cstdint>

namespace fry {

//
// A lightweight error type for hot-path operations.
//
// Why not just use std::error_code or exceptions?
//   - std::error_code works well for OS/system errors but requires registering
//     custom error categories — heavy boilerplate for in-process logic.
//   - Exceptions are zero-cost when not thrown, but catastrophically expensive
//     when they are. A search called 1M times/sec can't afford stack unwinding.
//   - Our Error is a plain struct: a machine-readable code and a human-readable
//     message. It fits in two words (pointer + enum). No heap allocation
//     beyond the message string itself.
//
// The Code enum is the authoritative set of errors this library can produce.
// Add new codes here as new modules are added — never use raw integers.
//
struct Error {
    enum class Code : std::uint8_t {
        Ok = 0,
        DimensionMismatch,   // query dim != index dim
        NotFound,            // ID does not exist in the store
        StoreEmpty,          // search on an empty index
        InvalidArgument,     // generic bad input (e.g. k == 0)
        IoError,             // disk read/write failure (Phase 5)
    };

    Code        code;
    std::string message;  // Human-readable detail.

    // Default-constructable as "no error".
    Error() : code(Code::Ok) {}
    Error(Code c, std::string msg) : code(c), message(std::move(msg)) {}

    // Convenience factories — prefer these over direct construction so that
    // error messages stay consistent across the codebase.
    static Error dimension_mismatch(std::size_t got, std::size_t expected) {
        return Error(Code::DimensionMismatch,
                     "dimension mismatch: got " + std::to_string(got) +
                     ", expected " + std::to_string(expected));
    }

    static Error not_found(std::uint64_t id) {
        return Error(Code::NotFound, "id " + std::to_string(id) + " not found");
    }

    static Error store_empty() {
        return Error(Code::StoreEmpty, "index is empty");
    }

    static Error invalid_argument(std::string msg) {
        return Error(Code::InvalidArgument, std::move(msg));
    }
};


// ── ResultImpl<T, E> ──────────────────────────────────────────────────────────
//
// A discriminated union that holds EITHER a value of type T OR an error of
// type E — never both, never neither.
//
// This is the C++11 hand-rolled equivalent of C++23's std::expected<T,E>.
//
// ── Key C++ concepts demonstrated ──────────────────────────────────────────
//
// 1. std::aligned_storage<Size, Align>
//    Provides a raw byte buffer of the right size and alignment.
//    It holds nothing until you explicitly construct something into it.
//    Think of it as a correctly-sized piece of memory waiting to be used.
//
// 2. Placement new  — new (&storage_) T(val)
//    Constructs an object at a specific memory address (our buffer).
//    This does NOT allocate memory — it only calls the constructor.
//    The buffer was already allocated as part of ResultImpl itself (on the
//    stack or wherever ResultImpl lives).
//
// 3. Explicit destructor call  — ptr->~T()
//    The only way to destroy an object constructed via placement new.
//    Normal delete would try to free memory we don't own — undefined behaviour.
//
// 4. Rule of Five
//    Whenever you define a custom destructor, the compiler no longer generates
//    copy/move constructors and copy/move assignment operators. You must write
//    all five yourself, otherwise the compiler silently does the wrong thing
//    (e.g. a bitwise copy of the buffer would duplicate a T without calling
//    T's copy constructor — undefined behaviour if T owns resources).
//
// ── Storage layout ────────────────────────────────────────────────────────────
//
//   [ ok_ : bool ][ storage_ : max(sizeof(T), sizeof(E)) bytes ]
//
//   ok_ == true  → storage_ contains a live T object
//   ok_ == false → storage_ contains a live E object
//   Never both, never neither.
//
template<typename T, typename E>
class ResultImpl {
    // Compute the size and alignment requirements for the buffer.
    // It must be large enough and aligned enough to hold either T or E.
    static const std::size_t kSize  = sizeof(T)  > sizeof(E)  ? sizeof(T)  : sizeof(E);
    static const std::size_t kAlign = alignof(T) > alignof(E) ? alignof(T) : alignof(E);

    // The raw buffer. Nothing is constructed here yet.
    // aligned_storage guarantees that reinterpret_cast<T*>(&storage_) is safe.
    typedef typename std::aligned_storage<kSize, kAlign>::type Storage;

    bool    ok_;
    Storage storage_;

    // Typed accessors into the buffer.
    // Safe to call only when you know which type is currently stored.
    T*       as_value()       { return reinterpret_cast<T*>(&storage_); }
    const T* as_value() const { return reinterpret_cast<const T*>(&storage_); }
    E*       as_error()       { return reinterpret_cast<E*>(&storage_); }
    const E* as_error() const { return reinterpret_cast<const E*>(&storage_); }

    // Private constructor — use the static factories below.
    explicit ResultImpl(bool ok) : ok_(ok) {}

public:
    // ── Factory methods ───────────────────────────────────────────────────────

    // Construct a success result holding val.
    static ResultImpl ok(T val) {
        ResultImpl r(true);
        // Placement new: construct T inside our buffer, moving val in.
        new (&r.storage_) T(std::move(val));
        return r;
        // NRVO (Named Return Value Optimisation) typically elides the move
        // of r here — but even if it doesn't, our move constructor handles it.
    }

    // Construct an error result holding e.
    static ResultImpl err(E e) {
        ResultImpl r(false);
        new (&r.storage_) E(std::move(e));
        return r;
    }

    // ── Rule of Five ──────────────────────────────────────────────────────────
    //
    // We have a non-trivial destructor, so we must define all five.
    // Failing to do so would leave copy/move operations defaulted to
    // bitwise copies of storage_ — which would duplicate objects without
    // calling their constructors (silent undefined behaviour).

    // 1. Destructor
    ~ResultImpl() {
        if (ok_) as_value()->~T();   // Call T's destructor.
        else     as_error()->~E();   // Call E's destructor.
        // No memory to free — storage_ is part of this object, not heap memory.
    }

    // 2. Copy constructor
    ResultImpl(const ResultImpl& other) : ok_(other.ok_) {
        if (ok_) new (&storage_) T(*other.as_value());  // Copy-construct T.
        else     new (&storage_) E(*other.as_error());  // Copy-construct E.
    }

    // 3. Move constructor
    ResultImpl(ResultImpl&& other) : ok_(other.ok_) {
        if (ok_) new (&storage_) T(std::move(*other.as_value()));  // Move T.
        else     new (&storage_) E(std::move(*other.as_error()));  // Move E.
        // Note: other.storage_ still contains a "moved-from" T/E.
        // other's destructor will still be called and must be valid —
        // std::move leaves the source in a valid but unspecified state.
    }

    // 4. Copy assignment
    ResultImpl& operator=(const ResultImpl& other) {
        if (this == &other) return *this;
        // Destroy current contents before overwriting.
        if (ok_) as_value()->~T();
        else     as_error()->~E();
        // Copy-construct new contents.
        ok_ = other.ok_;
        if (ok_) new (&storage_) T(*other.as_value());
        else     new (&storage_) E(*other.as_error());
        return *this;
    }

    // 5. Move assignment
    ResultImpl& operator=(ResultImpl&& other) {
        if (this == &other) return *this;
        if (ok_) as_value()->~T();
        else     as_error()->~E();
        ok_ = other.ok_;
        if (ok_) new (&storage_) T(std::move(*other.as_value()));
        else     new (&storage_) E(std::move(*other.as_error()));
        return *this;
    }

    // ── Observers ─────────────────────────────────────────────────────────────

    bool has_value() const { return ok_; }

    // Contextual bool: if (result) { ... use *result ... }
    explicit operator bool() const { return ok_; }

    // Dereference — undefined behaviour if !has_value().
    // Fast path: no branch, no exception. The caller checked has_value() first.
    T&       operator*()       { return *as_value(); }
    const T& operator*() const { return *as_value(); }

    // .value() — safe: throws std::runtime_error if this holds an error.
    // Use this in contexts where you "know" it succeeded (e.g. in tests or
    // after a successful REQUIRE check) but want a safety net.
    T& value() {
        if (!ok_) throw std::runtime_error("Result::value() called on error");
        return *as_value();
    }
    const T& value() const {
        if (!ok_) throw std::runtime_error("Result::value() called on error");
        return *as_value();
    }

    // .error() — undefined behaviour if has_value().
    E&       error()       { return *as_error(); }
    const E& error() const { return *as_error(); }

    // -> operator support for value.
    T* operator->()       { return as_value(); }
    const T* operator->() const { return as_value(); }
};


// ── Public aliases ────────────────────────────────────────────────────────────
//
// The error type is always fry::Error — callers write Result<VectorId>,
// not ResultImpl<VectorId, Error>. The alias bakes in E = Error.
//
// C++11 alias template: Result<T> is just ResultImpl<T, Error>.
// Every call site writes Result<VectorId>, Result<SearchResult>, etc.
// The Error type is invisible to callers — they only see success/failure.
template<typename T>
using Result = ResultImpl<T, Error>;


// ── Helper factories ──────────────────────────────────────────────────────────
//
// These let call sites read naturally without spelling out ResultImpl:
//
//   return make_ok(next_id_++);           // Result<VectorId> inferred
//   return make_error<VectorId>(e);       // explicit T, E inferred as Error
//
// make_ok uses template argument deduction — T is inferred from the argument.
// make_error requires explicit T since the return type depends on it.

template<typename T>
auto make_ok(T value) -> Result<T> {
    return Result<T>::ok(std::move(value));
}

template<typename T>
auto make_error(Error err) -> Result<T> {
    return Result<T>::err(std::move(err));
}

} // namespace fry
