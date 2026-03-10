"""Tests for the profiling sanity check script and helpers.

Uses source-level verification to validate script structure without
requiring GPU or model weights.
"""

import os
import subprocess
import textwrap

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts/profiling")
HELPERS_DIR = os.path.join(SCRIPT_DIR, "helpers")


# --- sanity_check.sh tests ---


class TestSanityCheckScript:
    """Verify sanity_check.sh structure and content."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        """Load the sanity check script source."""
        script_path = os.path.join(SCRIPT_DIR, "sanity_check.sh")
        with open(script_path) as f:
            self.source = f.read()

    def test_script_exists(self):
        """Script file exists at expected path."""
        assert os.path.isfile(os.path.join(SCRIPT_DIR, "sanity_check.sh"))

    def test_script_is_executable(self):
        """Script has executable permission."""
        path = os.path.join(SCRIPT_DIR, "sanity_check.sh")
        assert os.access(path, os.X_OK)

    def test_shebang(self):
        """Script starts with bash shebang."""
        assert self.source.startswith("#!/bin/bash")

    def test_activates_venv(self):
        """Script activates virtual environment."""
        assert ". \"$REPO_ROOT/.venv/bin/activate\"" in self.source

    def test_default_model_qwen3_06b(self):
        """Default model is Qwen3-0.6B."""
        assert "Qwen/Qwen3-0.6B" in self.source

    def test_model_configurable(self):
        """Model is configurable via first argument."""
        assert '${1:-' in self.source

    def test_shadow_tree_config(self):
        """Uses shadow+tree profiling configuration."""
        assert 'CONTEXT="shadow"' in self.source
        assert 'DATA="tree"' in self.source

    def test_calls_nano_vllm_helper(self):
        """Calls the nano-vllm offline profiling helper."""
        assert "nano_vllm_offline_run.py" in self.source

    def test_calls_vllm_helper(self):
        """Calls the vLLM offline profiling helper."""
        assert "vllm_offline_run.py" in self.source

    def test_validates_hatchet_files(self):
        """Validates .hatchet output files."""
        assert ".hatchet" in self.source

    def test_runs_proton_viewer(self):
        """Runs proton-viewer for validation."""
        assert "proton-viewer" in self.source

    def test_prints_pass_fail_summary(self):
        """Prints pass/fail summary for each framework."""
        assert "PASS" in self.source
        assert "FAIL" in self.source
        assert "SUMMARY" in self.source

    def test_nano_vllm_output_dir(self):
        """nano-vllm output goes to correct subdirectory."""
        assert 'NANO_OUTPUT="$OUTPUT_BASE/nano_vllm"' in self.source

    def test_vllm_output_dir(self):
        """vLLM output goes to correct subdirectory."""
        assert 'VLLM_OUTPUT="$OUTPUT_BASE/vllm"' in self.source

    def test_exit_code_on_success(self):
        """Script exits 0 on success."""
        assert "exit 0" in self.source

    def test_exit_code_on_failure(self):
        """Script exits 1 on failure."""
        assert "exit 1" in self.source


# --- nano_vllm_offline_run.py tests ---


class TestNanoVllmHelper:
    """Verify nano-vllm offline profiling helper structure."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Load the helper script source."""
        path = os.path.join(HELPERS_DIR, "nano_vllm_offline_run.py")
        with open(path) as f:
            self.source = f.read()

    def test_file_exists(self):
        """Helper script exists."""
        assert os.path.isfile(
            os.path.join(HELPERS_DIR, "nano_vllm_offline_run.py")
        )

    def test_imports_nanovllm(self):
        """Imports nano-vllm LLM and SamplingParams."""
        assert "from nanovllm import LLM, SamplingParams" in self.source

    def test_imports_profiler(self):
        """Imports ProtonProfiler from nanovllm.profiler."""
        assert "from nanovllm.profiler import ProtonProfiler" in self.source

    def test_has_model_arg(self):
        """Has --model argument."""
        assert "--model" in self.source

    def test_has_output_dir_arg(self):
        """Has --output-dir argument."""
        assert "--output-dir" in self.source

    def test_has_context_arg(self):
        """Has --context argument."""
        assert "--context" in self.source

    def test_has_data_arg(self):
        """Has --data argument."""
        assert "--data" in self.source

    def test_has_hook_arg(self):
        """Has --hook argument."""
        assert "--hook" in self.source

    def test_creates_profiler(self):
        """Creates ProtonProfiler with config args."""
        assert "ProtonProfiler(" in self.source

    def test_starts_profiler(self):
        """Calls profiler.start()."""
        assert "profiler.start()" in self.source

    def test_stops_profiler(self):
        """Calls profiler.stop()."""
        assert "profiler.stop()" in self.source

    def test_shuts_down_profiler(self):
        """Calls profiler.shutdown()."""
        assert "profiler.shutdown()" in self.source

    def test_runs_warmup(self):
        """Runs warmup before profiled run."""
        assert "warmup" in self.source.lower()

    def test_uses_enforce_eager(self):
        """Uses enforce_eager for small model."""
        assert "enforce_eager=True" in self.source


# --- vllm_offline_run.py tests ---


class TestVllmHelper:
    """Verify vLLM offline profiling helper structure."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Load the helper script source."""
        path = os.path.join(HELPERS_DIR, "vllm_offline_run.py")
        with open(path) as f:
            self.source = f.read()

    def test_file_exists(self):
        """Helper script exists."""
        assert os.path.isfile(
            os.path.join(HELPERS_DIR, "vllm_offline_run.py")
        )

    def test_imports_vllm(self):
        """Imports vLLM LLM and SamplingParams."""
        assert "from vllm import LLM, SamplingParams" in self.source

    def test_imports_profiler_config(self):
        """Imports ProfilerConfig."""
        assert "ProfilerConfig" in self.source

    def test_has_model_arg(self):
        """Has --model argument."""
        assert "--model" in self.source

    def test_has_output_dir_arg(self):
        """Has --output-dir argument."""
        assert "--output-dir" in self.source

    def test_has_context_arg(self):
        """Has --context argument."""
        assert "--context" in self.source

    def test_has_data_arg(self):
        """Has --data argument."""
        assert "--data" in self.source

    def test_creates_profiler_config(self):
        """Creates ProfilerConfig with proton settings."""
        assert 'profiler="proton"' in self.source

    def test_passes_profiler_config_to_llm(self):
        """Passes profiler_config to LLM constructor."""
        assert "profiler_config=profiler_config" in self.source

    def test_calls_start_profile(self):
        """Calls llm.start_profile()."""
        assert "start_profile()" in self.source

    def test_calls_stop_profile(self):
        """Calls llm.stop_profile()."""
        assert "stop_profile()" in self.source

    def test_runs_warmup(self):
        """Runs warmup before profiled run."""
        assert "warmup" in self.source.lower()

    def test_uses_enforce_eager(self):
        """Uses enforce_eager for small model."""
        assert "enforce_eager=True" in self.source


# --- Integration: shell script syntax check ---


class TestScriptSyntax:
    """Verify shell script has valid syntax."""

    def test_sanity_check_syntax(self):
        """sanity_check.sh passes bash syntax check."""
        script_path = os.path.join(SCRIPT_DIR, "sanity_check.sh")
        result = subprocess.run(
            ["bash", "-n", script_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"
