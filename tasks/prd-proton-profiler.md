# PRD: Integrate Proton Profiler Backend in vLLM

## Introduction

Add Triton's built-in Proton profiler as a new profiler backend in vLLM, complementing the existing `torch` and `cuda` profiler backends. Proton provides GPU kernel-level profiling with Triton-native insights. This is a minimal viable integration that adds Proton as a git submodule and implements the `WorkerProfiler` interface, reusing vLLM's existing profiler lifecycle (start/stop/annotate).

## Goals

- Add Triton as a git submodule to access the latest Proton features
- Implement `ProtonProfilerWrapper` extending the existing `WorkerProfiler` base class
- Register `"proton"` as a new `ProfilerKind` option
- Unify Proton annotations with vLLM's existing `annotate_context_manager()` pattern
- Output profiling data in Proton's native `.hatchet` format
- Ensure zero impact on existing `torch` and `cuda` profiler workflows

## User Stories

### US-001: Add Triton as a git submodule
**Description:** As a developer, I need Triton available as a submodule so I can install the latest Proton profiler with up-to-date features.

**Acceptance Criteria:**
- [ ] Triton added as git submodule under `submodules/triton` pointing to `https://github.com/triton-lang/triton.git`
- [ ] `.gitmodules` updated with the submodule entry
- [ ] Proton is importable after running `uv pip install -e submodules/triton`
- [ ] `import triton.profiler as proton` works in the vLLM environment
- [ ] Existing vLLM build process is not broken

### US-002: Add "proton" to ProfilerKind
**Description:** As a developer, I need `"proton"` recognized as a valid profiler kind so the config system can route to the Proton backend.

**Acceptance Criteria:**
- [ ] `ProfilerKind` literal in `vllm/config/profiler.py` includes `"proton"`
- [ ] `ProfilerConfig` accepts `profiler="proton"` without validation errors
- [ ] Existing `"torch"` and `"cuda"` options continue to work unchanged
- [ ] Typecheck passes

### US-003: Implement ProtonProfilerWrapper
**Description:** As a developer, I need a `ProtonProfilerWrapper` class that implements the `WorkerProfiler` interface so Proton profiling integrates with vLLM's profiler lifecycle.

**Acceptance Criteria:**
- [ ] `ProtonProfilerWrapper` extends `WorkerProfiler` in `vllm/profiler/wrapper.py`
- [ ] `_start()` initializes a Proton profiling session (`proton.start()`)
- [ ] `_stop()` finalizes the Proton session and writes output (`proton.finalize()`)
- [ ] Output is written in Proton's native `.hatchet` format to the configured output directory
- [ ] Multi-rank support: output files include rank suffix (consistent with existing profilers)
- [ ] Profiler is instantiated correctly from `ProfilerConfig` when `profiler="proton"`
- [ ] Typecheck passes

### US-004: Integrate Proton annotations with existing annotation system
**Description:** As a user, I want existing annotated regions in vLLM to appear in Proton traces so I get meaningful profiling data without code changes.

**Acceptance Criteria:**
- [ ] `annotate_context_manager()` in `vllm/profiler/wrapper.py` supports Proton's scope/annotation API
- [ ] When Proton profiler is active, `proton.scope()` or equivalent is used for region marking
- [ ] Existing annotation call sites work with Proton without modification
- [ ] Typecheck passes

### US-005: Wire ProtonProfilerWrapper into worker instantiation
**Description:** As a user, I want to select Proton profiler via config and have it work end-to-end through the existing start/stop profile API.

**Acceptance Criteria:**
- [ ] `create_worker_profiler()` or equivalent factory in `vllm/profiler/wrapper.py` handles `"proton"` kind
- [ ] Proton profiler can be started/stopped via `vllm serve` + `/start_profile` and `/stop_profile` API endpoints
- [ ] Proton profiler can be started/stopped via `LLM.start_profile()` / `LLM.stop_profile()` in offline mode
- [ ] Profiling output file is created in the configured directory
- [ ] Typecheck passes

### US-006: Write tests for Proton profiler integration
**Description:** As a developer, I need tests to verify the Proton profiler backend works correctly.

**Acceptance Criteria:**
- [ ] Unit test: `ProtonProfilerWrapper` correctly implements start/stop lifecycle
- [ ] Unit test: Proton annotation context manager works
- [ ] Integration test: end-to-end profiling with `profiler="proton"` produces output file
- [ ] Test that `ProfilerConfig(profiler="proton")` is valid
- [ ] Tests skip gracefully if Proton is not installed (conditional skip)
- [ ] All existing profiler tests continue to pass

## Functional Requirements

- FR-1: Add Triton as a git submodule at `submodules/triton` from `https://github.com/triton-lang/triton.git`
- FR-2: Extend `ProfilerKind` literal type to include `"proton"` alongside `"torch"` and `"cuda"`
- FR-3: Implement `ProtonProfilerWrapper(WorkerProfiler)` with `_start()` and `_stop()` methods that call Proton's session API
- FR-4: Output profiling data in Proton's native `.hatchet` format to the directory specified by config
- FR-5: Integrate Proton's annotation API (`proton.scope()`) into vLLM's `annotate_context_manager()` so existing annotated regions appear in Proton traces
- FR-6: Route `profiler="proton"` in the profiler factory to instantiate `ProtonProfilerWrapper`
- FR-7: Ensure Proton profiler works through all existing trigger paths: Python API (`start_profile`/`stop_profile`), HTTP API endpoints, and CLI

## Non-Goals

- No Proton-specific configuration options beyond what `ProfilerConfig` already supports (output dir, delay/max iterations)
- No conversion of Proton output to Chrome trace / Perfetto format
- No custom Proton metrics or advanced scope hierarchies beyond basic region annotation
- No changes to existing `torch` or `cuda` profiler behavior
- No Proton-specific analysis tooling (like the Nsight tools in `tools/profiler/`)
- No CI/CD integration for continuous Proton profiling
- Not replacing or deprecating any existing profiler backends

## Technical Considerations

- **Dependency:** Proton is part of Triton (`triton.profiler`). It is available wherever Triton is installed, which includes standard PyTorch installations. The submodule provides access to the latest version.
- **Lazy import:** Proton should be imported lazily (only when `profiler="proton"` is selected) to avoid import errors when Triton's profiler module is not available.
- **Existing architecture:** The `WorkerProfiler` base class in `vllm/profiler/wrapper.py` provides scheduling logic (delay, max iterations, warmup). `ProtonProfilerWrapper` inherits this for free.
- **Proton source & docs:** After adding Triton as a submodule, Proton's source code and documentation are located at `submodules/triton/third_party/proton`. Refer to this for API details and usage examples.
- **Key Proton API:** `proton.start(name, hook)`, `proton.finalize()`, `proton.scope(name)`, `proton.record()`. Refer to Proton documentation at `submodules/triton/third_party/proton`.
- **Files to modify:**
  - `vllm/config/profiler.py` — add `"proton"` to `ProfilerKind`
  - `vllm/profiler/wrapper.py` — add `ProtonProfilerWrapper` class and update factory logic
  - `.gitmodules` — add Triton submodule entry
- **Previous branches:** The user has prior work on branches `gpt-oss-profile`, `llmprof`, `064proton`, `proton-profile-sync`, `v0.10.1.1-llmprof`, etc. These can be referenced for implementation patterns but should not be merged directly.

## Success Metrics

- `profiler="proton"` config option works end-to-end without errors
- Proton `.hatchet` output file is generated with valid profiling data
- Annotated regions in vLLM appear correctly in Proton traces
- Zero regression in existing `torch` and `cuda` profiler tests
- All new tests pass
- **End-to-end validation:** The following command (replacing `torch` with `proton`) produces valid profiling output:
  ```bash
  vllm serve openai/gpt-oss-20b \
      --profiler-config '{"profiler": "proton", "torch_profiler_dir": "${VLLM_TORCH_PROFILER_DIR}"}'
  ```
  Then trigger profiling via `/start_profile` and `/stop_profile` endpoints and verify `.hatchet` output is written

## Development Guidelines

These project-wide conventions apply to all implementation work:

- **Simple design:** Make clear and simple code design for ease of understanding. Good design is simple yet effective design.
- **Documentation:** Write docstring and inline comments for every method.
- **Testing:** Write comprehensive tests for every function: unit tests, integration tests, smoke tests.
- **Minimal code:** Only write absolutely necessary code, avoid redundant code. Regularly look back to see whether we can remove unnecessary code — keep the codebase clean.
- **No fallbacks:** Avoid try-catch patterns. Fail early to discover issues at the shallow surface.
- **Environment:** Python environment is managed by `uv`. Run `. .venv/bin/activate` to activate and use `uv pip install ...` to manage packages.
- **Commits:** Do not include "Co-Authored-By: Claude" in commit messages. Do not include US-00X references in commit messages. Follow vLLM's commit message style (e.g., `[Profiler] Add Proton profiler backend`).

## Open Questions

- Which Triton commit/tag should the submodule pin to? (latest main vs. a stable release)
- Should Proton output directory reuse `torch_profiler_dir` config or get its own config field (e.g., `proton_profiler_dir`)?
- Does Proton's `proton.start()` API support the same schedule-based profiling (wait/warmup/active phases) as torch profiler, or should scheduling be handled entirely by the `WorkerProfiler` base class?
