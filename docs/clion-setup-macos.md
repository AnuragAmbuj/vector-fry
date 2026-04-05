# CLion Setup for fry-vector

This project uses **CMakePresets.json** for build configuration and **vcpkg** (manifest mode)
for dependency management. CLion requires a small one-time local setup because its bundled
cmake process does not source your shell profile (`.zshrc` / `.bashrc`), so environment
variables like `VCPKG_ROOT` are not available to it automatically.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| CLion | 2022.3 | Required for CMakePresets.json support |
| CMake | 3.25 | CLion ships its own; see note below |
| Ninja | any | `brew install ninja` |
| vcpkg | — | Clone from https://github.com/microsoft/vcpkg |

> **CMake version note** — CLion uses its own bundled cmake
> (`CLion.app/Contents/bin/cmake/…`). This is fine; the setup below targets
> that binary explicitly.

---

## 1 — Install vcpkg

If you have not already cloned vcpkg, pick a permanent home for it (the path
must **not** change after packages are cached):

```bash
git clone https://github.com/microsoft/vcpkg ~/Tools/vcpkg
~/Tools/vcpkg/bootstrap-vcpkg.sh -disableMetrics
```

Add the following two lines to your `~/.zshrc` (or `~/.bashrc`):

```bash
export VCPKG_ROOT="$HOME/Tools/vcpkg"
export PATH="$VCPKG_ROOT:$PATH"
```

Reload your shell: `source ~/.zshrc`

---

## 2 — Create your local CMakeUserPresets.json

`CMakeUserPresets.json` is a **machine-local** file (it is gitignored).
It tells CLion's bundled cmake exactly where vcpkg lives on your machine,
without hardcoding anything in the shared `CMakePresets.json`.

Create the file at the **project root** (next to `CMakePresets.json`):

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },

  "configurePresets": [
    {
      "name": "clion-base",
      "hidden": true,
      "generator": "Ninja",
      "toolchainFile": "/YOUR/PATH/TO/vcpkg/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "VCPKG_MANIFEST_MODE": "ON",
        "VCPKG_TARGET_TRIPLET": "YOUR_TRIPLET"
      }
    },
    {
      "name": "clion-debug",
      "displayName": "Debug (CLion)",
      "inherits": "clion-base",
      "binaryDir": "${sourceDir}/cmake-build-debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "BUILD_TESTS": "ON"
      }
    },
    {
      "name": "clion-release",
      "displayName": "Release (CLion)",
      "inherits": "clion-base",
      "binaryDir": "${sourceDir}/cmake-build-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_TESTS": "OFF"
      }
    },
    {
      "name": "clion-asan",
      "displayName": "ASan (CLion)",
      "inherits": "clion-base",
      "binaryDir": "${sourceDir}/cmake-build-asan",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "ENABLE_ASAN": "ON",
        "BUILD_TESTS": "ON"
      }
    }
  ],

  "buildPresets": [
    { "name": "clion-debug",   "configurePreset": "clion-debug",   "jobs": 0 },
    { "name": "clion-release", "configurePreset": "clion-release",  "jobs": 0 },
    { "name": "clion-asan",    "configurePreset": "clion-asan",     "jobs": 0 }
  ],

  "testPresets": [
    {
      "name": "clion-debug",
      "configurePreset": "clion-debug",
      "output": { "outputOnFailure": true }
    },
    {
      "name": "clion-asan",
      "configurePreset": "clion-asan",
      "output": { "outputOnFailure": true },
      "environment": {
        "ASAN_OPTIONS":  "detect_leaks=1:abort_on_error=1",
        "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"
      }
    }
  ]
}
```

### Fill in your values

**`toolchainFile`** — replace `/YOUR/PATH/TO/vcpkg` with the absolute path
where you cloned vcpkg, e.g. `/Users/alice/Tools/vcpkg`.

**`VCPKG_TARGET_TRIPLET`** — pick the triplet for your machine:

| Machine | Triplet |
|---------|---------|
| Apple Silicon Mac (M1/M2/M3/M4) | `arm64-osx` |
| Intel Mac | `x64-osx` |
| Linux x86-64 | `x64-linux` |
| Windows x64 | `x64-windows` |

---

## 3 — Enable the presets in CLion

CLion detects `CMakeUserPresets.json` automatically when you open or reload
the project. You need to activate the new profiles and retire the old
hand-generated "Debug" one:

1. Open **Settings → Build, Execution, Deployment → CMake**
   (macOS shortcut: `⌘,` then search "cmake").

2. Find the profile named **"Debug"** (the plain one, not from a preset) and
   **uncheck** or **delete** it.

3. You should now see **Debug (CLion)**, **Release (CLion)**, and
   **ASan (CLion)** listed. **Check all three** to enable them.

4. Click **OK** and let CLion reload the project.

On the first load vcpkg will download and build all dependencies into
`cmake-build-debug/vcpkg_installed/` (takes ~30–60 s; subsequent reloads
use the binary cache and finish in under a second).

---

## 4 — Verify

After the CMake reload completes you should see the three profiles in the
build configuration drop-down (top-right toolbar):

```
Debug (CLion)    ← day-to-day development
Release (CLion)  ← performance testing
ASan (CLion)     ← memory / UB error hunting
```

Run a build with **⌘F9** or the hammer icon. A successful output looks like:

```
-- Running vcpkg install - done
-- Found nlohmann_json: …/vcpkg_installed/arm64-osx/…
-- Configuring done
-- Generating done
-- Build files have been written to: …/cmake-build-debug
```

---

## Why this two-file approach?

| File | Committed? | Purpose |
|------|-----------|---------|
| `CMakePresets.json` | ✅ yes | Shared, portable presets. Uses `$env{VCPKG_ROOT}` — works in any terminal that has the env var set (CI, command line). |
| `CMakeUserPresets.json` | ❌ gitignored | Machine-local overrides. Hardcodes the absolute vcpkg path so CLion's bundled cmake (which never sees `.zshrc`) can find the toolchain. |

This is the pattern [recommended by the vcpkg team](https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration)
for IDE environments that do not inherit the login shell.

---

## Command-line usage (outside CLion)

The shared presets work from any terminal where `VCPKG_ROOT` is exported:

```bash
# Configure
cmake --preset debug       # or: release, asan

# Build
cmake --build --preset debug

# Test
ctest --preset debug
ctest --preset asan        # runs with ASan + UBSan env options
```
