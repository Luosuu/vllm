"""Tests for the vLLM offline batch profiling script.

Uses source-level verification to validate script structure without
requiring GPU or model weights.
"""

import os
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts/profiling")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "vllm_offline.sh")


class TestVllmOfflineScript:
    """Verify vllm_offline.sh structure and content."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        """Load the script source."""
        with open(SCRIPT_PATH) as f:
            self.source = f.read()

    def test_script_exists(self):
        """Script file exists at expected path."""
        assert os.path.isfile(SCRIPT_PATH)

    def test_script_is_executable(self):
        """Script has executable permission."""
        assert os.access(SCRIPT_PATH, os.X_OK)

    def test_shebang(self):
        """Script starts with bash shebang."""
        assert self.source.startswith("#!/bin/bash")

    def test_activates_venv(self):
        """Script activates virtual environment."""
        assert '. "$REPO_ROOT/.venv/bin/activate"' in self.source

    def test_default_model_qwen3_32b(self):
        """Default model is Qwen3-32B."""
        assert "Qwen/Qwen3-32B" in self.source

    # --- Configurable workload parameters ---

    def test_batch_size_configurable(self):
        """Batch size is configurable via --batch-size."""
        assert "--batch-size" in self.source
        assert "BATCH_SIZE" in self.source

    def test_input_len_configurable(self):
        """Input length is configurable via --input-len."""
        assert "--input-len" in self.source
        assert "INPUT_LEN" in self.source

    def test_output_len_configurable(self):
        """Output length is configurable via --output-len."""
        assert "--output-len" in self.source
        assert "OUTPUT_LEN" in self.source

    def test_seed_configurable(self):
        """Random seed is configurable via --seed."""
        assert "--seed" in self.source
        assert "SEED" in self.source

    def test_model_configurable(self):
        """Model is configurable via --model."""
        assert "--model" in self.source

    # --- Matching workload defaults to nano-vllm ---

    def test_default_batch_size_matches_nano(self):
        """Default batch size matches nano-vllm (4)."""
        assert 'BATCH_SIZE=4' in self.source

    def test_default_input_len_matches_nano(self):
        """Default input length matches nano-vllm (2048)."""
        assert 'INPUT_LEN=2048' in self.source

    def test_default_output_len_matches_nano(self):
        """Default output length matches nano-vllm (128)."""
        assert 'OUTPUT_LEN=128' in self.source

    def test_default_seed_matches_nano(self):
        """Default random seed matches nano-vllm (42)."""
        assert 'SEED=42' in self.source

    # --- 4 profiling configurations ---

    def test_shadow_tree_config(self):
        """Includes shadow+tree profiling configuration."""
        assert "shadow:tree" in self.source

    def test_shadow_trace_config(self):
        """Includes shadow+trace profiling configuration."""
        assert "shadow:trace" in self.source

    def test_python_tree_config(self):
        """Includes python+tree profiling configuration."""
        assert "python:tree" in self.source

    def test_python_trace_config(self):
        """Includes python+trace profiling configuration."""
        assert "python:trace" in self.source

    def test_four_configs_total(self):
        """Exactly 4 profiling configurations defined."""
        config_entries = [
            line.strip()
            for line in self.source.splitlines()
            if line.strip().startswith('"') and ':' in line.strip()
            and line.strip().endswith('"')
            and any(
                ctx in line for ctx in ["shadow:tree", "shadow:trace",
                                        "python:tree", "python:trace"]
            )
        ]
        assert len(config_entries) == 4

    # --- Output directories ---

    def test_output_base_dir(self):
        """Output base is profiling_output/vllm/offline/."""
        assert "profiling_output/vllm/offline" in self.source

    def test_separate_subdirectories(self):
        """Each config outputs to a separate subdirectory."""
        assert "${CONTEXT}_${DATA}" in self.source

    # --- Calls vLLM helper script ---

    def test_calls_vllm_helper(self):
        """Calls the vllm_offline_run.py helper."""
        assert "vllm_offline_run.py" in self.source

    def test_passes_context_to_helper(self):
        """Passes --context to the helper script."""
        assert '--context "$CONTEXT"' in self.source

    def test_passes_data_to_helper(self):
        """Passes --data to the helper script."""
        assert '--data "$DATA"' in self.source

    def test_passes_hook_to_helper(self):
        """Passes --hook to the helper script with triton value."""
        assert '--hook "triton"' in self.source

    def test_passes_model_to_helper(self):
        """Passes --model to the helper script."""
        assert '--model "$MODEL"' in self.source

    def test_passes_num_prompts_to_helper(self):
        """Passes batch size as --num-prompts to the helper."""
        assert '--num-prompts "$BATCH_SIZE"' in self.source

    def test_passes_max_tokens_to_helper(self):
        """Passes output length as --max-tokens to the helper."""
        assert '--max-tokens "$OUTPUT_LEN"' in self.source

    def test_passes_max_model_len_to_helper(self):
        """Passes input length as --max-model-len to the helper."""
        assert '--max-model-len "$INPUT_LEN"' in self.source

    # --- Uses vLLM's profiler config (via helper) ---

    def test_uses_vllm_profiler_config(self):
        """Uses vLLM's ProfilerConfig via the vllm_offline_run.py helper."""
        # The helper uses ProfilerConfig - verify the script calls the right helper
        assert "vllm_offline_run.py" in self.source

    # --- Validation ---

    def test_validates_hatchet_files(self):
        """Validates .hatchet output files with proton-viewer."""
        assert ".hatchet" in self.source
        assert "proton-viewer" in self.source

    # --- Summary ---

    def test_prints_output_file_summary(self):
        """Prints summary of output files on completion."""
        assert "OUTPUT FILE SUMMARY" in self.source

    def test_prints_pass_fail(self):
        """Prints pass/fail information."""
        assert "Passed" in self.source
        assert "Failed" in self.source

    def test_exit_code_success(self):
        """Exits 0 on success."""
        assert "exit 0" in self.source

    def test_exit_code_failure(self):
        """Exits 1 on failure."""
        assert "exit 1" in self.source


class TestVllmOfflineHelperScript:
    """Verify vllm_offline_run.py uses vLLM's ProfilerConfig."""

    @pytest.fixture(autouse=True)
    def _load_helper(self):
        """Load the helper script source."""
        helper_path = os.path.join(SCRIPT_DIR, "helpers/vllm_offline_run.py")
        with open(helper_path) as f:
            self.source = f.read()

    def test_imports_profiler_config(self):
        """Helper imports ProfilerConfig from vllm."""
        assert "ProfilerConfig" in self.source

    def test_uses_proton_profiler(self):
        """Helper sets profiler='proton' in ProfilerConfig."""
        assert 'profiler="proton"' in self.source

    def test_uses_proton_profiler_dir(self):
        """Helper passes proton_profiler_dir to ProfilerConfig."""
        assert "proton_profiler_dir" in self.source

    def test_uses_proton_context(self):
        """Helper passes proton_context to ProfilerConfig."""
        assert "proton_context" in self.source

    def test_uses_proton_data(self):
        """Helper passes proton_data to ProfilerConfig."""
        assert "proton_data" in self.source

    def test_uses_proton_hook(self):
        """Helper passes proton_hook to ProfilerConfig."""
        assert "proton_hook" in self.source

    def test_uses_llm_class(self):
        """Helper uses vLLM's LLM class for offline inference."""
        assert "from vllm import LLM" in self.source

    def test_calls_start_profile(self):
        """Helper calls llm.start_profile() for profiled run."""
        assert "start_profile()" in self.source

    def test_calls_stop_profile(self):
        """Helper calls llm.stop_profile() after profiled run."""
        assert "stop_profile()" in self.source


class TestVllmOfflineSyntax:
    """Verify shell script has valid syntax."""

    def test_bash_syntax(self):
        """vllm_offline.sh passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"
