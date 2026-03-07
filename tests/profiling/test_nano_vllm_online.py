"""Tests for the nano-vllm online profiling script and helper.

Uses source-level verification to validate script structure without
requiring GPU or model weights.
"""

import ast
import os
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts/profiling")
HELPERS_DIR = os.path.join(SCRIPT_DIR, "helpers")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "nano_vllm_online.sh")
HELPER_PATH = os.path.join(HELPERS_DIR, "nano_vllm_online_run.py")


# --- nano_vllm_online.sh tests ---


class TestNanoVllmOnlineScript:
    """Verify nano_vllm_online.sh structure and content."""

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

    def test_num_prompts_configurable(self):
        """Number of prompts is configurable via --num-prompts."""
        assert "--num-prompts" in self.source
        assert "NUM_PROMPTS" in self.source

    def test_warmup_prompts_configurable(self):
        """Number of warmup prompts is configurable via --warmup-prompts."""
        assert "--warmup-prompts" in self.source
        assert "WARMUP_PROMPTS" in self.source

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

    # --- Output directories ---

    def test_output_base_dir(self):
        """Output base is profiling_output/nano_vllm/online/."""
        assert "profiling_output/nano_vllm/online" in self.source

    def test_separate_subdirectory(self):
        """Output goes to a context_data subdirectory."""
        assert "${CONTEXT}_${DATA}" in self.source

    # --- Periodic flushing ---

    def test_calls_online_helper(self):
        """Calls the nano_vllm_online_run.py helper."""
        assert "nano_vllm_online_run.py" in self.source

    def test_passes_context_to_helper(self):
        """Passes --context to the helper script."""
        assert '--context "$CONTEXT"' in self.source

    def test_passes_data_to_helper(self):
        """Passes --data to the helper script."""
        assert '--data "$DATA"' in self.source

    def test_passes_hook_to_helper(self):
        """Passes --hook to the helper script."""
        assert '--hook "$HOOK"' in self.source

    def test_passes_model_to_helper(self):
        """Passes --model to the helper script."""
        assert '--model "$MODEL"' in self.source

    def test_passes_num_prompts_to_helper(self):
        """Passes num prompts to the helper."""
        assert '--num-prompts "$NUM_PROMPTS"' in self.source

    def test_passes_warmup_prompts_to_helper(self):
        """Passes warmup prompts to the helper."""
        assert '--warmup-prompts "$WARMUP_PROMPTS"' in self.source

    def test_passes_max_tokens_to_helper(self):
        """Passes output length as --max-tokens to the helper."""
        assert '--max-tokens "$OUTPUT_LEN"' in self.source

    def test_passes_max_model_len_to_helper(self):
        """Passes input length as --max-model-len to the helper."""
        assert '--max-model-len "$INPUT_LEN"' in self.source

    def test_passes_seed_to_helper(self):
        """Passes seed to the helper."""
        assert '--seed "$SEED"' in self.source

    # --- Validation ---

    def test_validates_hatchet_files(self):
        """Validates .hatchet output files with proton-viewer."""
        assert ".hatchet" in self.source
        assert "proton-viewer" in self.source

    def test_checks_part_files(self):
        """Checks for per-phase .part_N files."""
        assert ".part_" in self.source
        assert "part_count" in self.source

    # --- Summary ---

    def test_prints_output_file_summary(self):
        """Prints summary of output files on completion."""
        assert "OUTPUT FILE SUMMARY" in self.source or "SUMMARY" in self.source

    def test_prints_per_phase_count(self):
        """Prints count of per-phase files."""
        assert "Per-phase files" in self.source

    def test_exit_code_success(self):
        """Exits 0 on success."""
        assert "exit 0" in self.source

    def test_exit_code_failure(self):
        """Exits 1 on failure."""
        assert "exit 1" in self.source


class TestNanoVllmOnlineSyntax:
    """Verify shell script has valid syntax."""

    def test_bash_syntax(self):
        """nano_vllm_online.sh passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


# --- nano_vllm_online_run.py helper tests ---


class TestNanoVllmOnlineHelper:
    """Verify nano_vllm_online_run.py helper structure."""

    @pytest.fixture(autouse=True)
    def _load_helper(self):
        """Load the helper script source."""
        with open(HELPER_PATH) as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def test_helper_exists(self):
        """Helper file exists at expected path."""
        assert os.path.isfile(HELPER_PATH)

    def test_helper_is_valid_python(self):
        """Helper is valid Python syntax."""
        ast.parse(self.source)

    def test_imports_nanovllm(self):
        """Helper imports from nanovllm."""
        assert "from nanovllm" in self.source

    def test_imports_proton_profiler(self):
        """Helper imports ProtonProfiler."""
        assert "ProtonProfiler" in self.source

    def test_imports_set_profiler(self):
        """Helper imports set_profiler."""
        assert "set_profiler" in self.source

    def test_uses_periodic_flushing_mode(self):
        """Helper uses periodic_flushing mode."""
        assert "periodic_flushing" in self.source

    def test_has_main_function(self):
        """Helper defines a main() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "main" in func_names

    def test_has_run_phase_function(self):
        """Helper defines a run_phase() function for simulating online requests."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "run_phase" in func_names

    def test_has_generate_prompts_function(self):
        """Helper defines a generate_prompts() function for varying prompt lengths."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "generate_prompts" in func_names

    def test_uses_add_request(self):
        """Helper uses add_request for sequential request submission."""
        assert "add_request" in self.source

    def test_uses_step(self):
        """Helper uses step() for iterative processing."""
        assert ".step()" in self.source

    def test_uses_is_finished(self):
        """Helper checks is_finished() for completion."""
        assert "is_finished" in self.source

    def test_profiler_start_stop_cycle(self):
        """Helper calls profiler.start() and profiler.stop() for each phase."""
        assert "profiler.start()" in self.source
        assert "profiler.stop()" in self.source

    def test_profiler_shutdown(self):
        """Helper calls profiler.shutdown() at the end."""
        assert "profiler.shutdown()" in self.source

    def test_two_phases(self):
        """Helper captures at least two phases (warmup and steady-state)."""
        # Count distinct profiler.start()/stop() pairs
        start_count = self.source.count("profiler.start()")
        stop_count = self.source.count("profiler.stop()")
        assert start_count >= 2, f"Expected at least 2 start() calls, found {start_count}"
        assert stop_count >= 2, f"Expected at least 2 stop() calls, found {stop_count}"

    def test_warmup_phase(self):
        """Helper has a warmup phase."""
        assert "warmup" in self.source.lower()

    def test_steady_state_phase(self):
        """Helper has a steady-state phase."""
        assert "steady" in self.source.lower()

    def test_varying_prompt_lengths(self):
        """Helper uses prompts with varying lengths."""
        assert "PROMPT_TEMPLATES" in self.source

    def test_argparse_model(self):
        """Helper accepts --model argument."""
        assert "--model" in self.source

    def test_argparse_output_dir(self):
        """Helper accepts --output-dir argument."""
        assert "--output-dir" in self.source

    def test_argparse_context(self):
        """Helper accepts --context argument."""
        assert "--context" in self.source

    def test_argparse_data(self):
        """Helper accepts --data argument."""
        assert "--data" in self.source

    def test_argparse_seed(self):
        """Helper accepts --seed argument."""
        assert "--seed" in self.source

    def test_argparse_warmup_prompts(self):
        """Helper accepts --warmup-prompts argument."""
        assert "--warmup-prompts" in self.source

    def test_enforce_eager(self):
        """Helper uses enforce_eager=True to avoid CUDA graph OOM."""
        assert "enforce_eager=True" in self.source

    def test_prints_status(self):
        """Helper prints profiler status at the end."""
        assert "get_status" in self.source

    def test_if_name_main(self):
        """Helper has if __name__ == '__main__' guard."""
        assert '__name__' in self.source
        assert '__main__' in self.source
