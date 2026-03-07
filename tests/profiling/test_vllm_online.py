"""Tests for the vLLM online profiling script and helper.

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
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "vllm_online.sh")
HELPER_PATH = os.path.join(HELPERS_DIR, "vllm_online_run.py")


# --- vllm_online.sh tests ---


class TestVllmOnlineScript:
    """Verify vllm_online.sh structure and content."""

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

    def test_port_configurable(self):
        """Server port is configurable via --port."""
        assert "--port" in self.source
        assert "PORT" in self.source

    # --- Output directories ---

    def test_output_base_dir(self):
        """Output base is profiling_output/vllm/online/."""
        assert "profiling_output/vllm/online" in self.source

    def test_separate_subdirectory(self):
        """Output goes to a context_data subdirectory."""
        assert "${CONTEXT}_${DATA}" in self.source

    # --- Periodic flushing ---

    def test_calls_online_helper(self):
        """Calls the vllm_online_run.py helper."""
        assert "vllm_online_run.py" in self.source

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

    def test_passes_port_to_helper(self):
        """Passes port to the helper."""
        assert '--port "$PORT"' in self.source

    # --- Matching workload to nano-vllm ---

    def test_default_num_prompts_matches_nano(self):
        """Default num prompts matches nano-vllm online (4)."""
        assert 'NUM_PROMPTS=4' in self.source

    def test_default_warmup_prompts_matches_nano(self):
        """Default warmup prompts matches nano-vllm online (2)."""
        assert 'WARMUP_PROMPTS=2' in self.source

    def test_default_seed_matches_nano(self):
        """Default seed matches nano-vllm online (42)."""
        assert 'SEED=42' in self.source

    def test_default_input_len_matches_nano(self):
        """Default input length matches nano-vllm online (2048)."""
        assert 'INPUT_LEN=2048' in self.source

    def test_default_output_len_matches_nano(self):
        """Default output length matches nano-vllm online (128)."""
        assert 'OUTPUT_LEN=128' in self.source

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


class TestVllmOnlineSyntax:
    """Verify shell script has valid syntax."""

    def test_bash_syntax(self):
        """vllm_online.sh passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


# --- vllm_online_run.py helper tests ---


class TestVllmOnlineHelper:
    """Verify vllm_online_run.py helper structure."""

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

    def test_imports_requests(self):
        """Helper imports requests for REST API calls."""
        assert "import requests" in self.source

    def test_imports_subprocess(self):
        """Helper imports subprocess for server management."""
        assert "import subprocess" in self.source

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

    def test_has_generate_prompts_function(self):
        """Helper defines a generate_prompts() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "generate_prompts" in func_names

    def test_has_wait_for_server_function(self):
        """Helper defines a wait_for_server() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "wait_for_server" in func_names

    def test_has_start_profile_function(self):
        """Helper defines a start_profile() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "start_profile" in func_names

    def test_has_stop_profile_function(self):
        """Helper defines a stop_profile() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "stop_profile" in func_names

    def test_has_get_profile_status_function(self):
        """Helper defines a get_profile_status() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "get_profile_status" in func_names

    def test_has_send_chat_completions_function(self):
        """Helper defines a send_chat_completions() function."""
        func_names = [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "send_chat_completions" in func_names

    # --- REST API endpoints ---

    def test_uses_start_profile_endpoint(self):
        """Helper calls /start_profile endpoint."""
        assert "/start_profile" in self.source

    def test_uses_stop_profile_endpoint(self):
        """Helper calls /stop_profile endpoint."""
        assert "/stop_profile" in self.source

    def test_uses_profile_status_endpoint(self):
        """Helper calls /profile_status endpoint."""
        assert "/profile_status" in self.source

    def test_uses_chat_completions_endpoint(self):
        """Helper uses /v1/chat/completions endpoint."""
        assert "/v1/chat/completions" in self.source

    def test_uses_health_endpoint(self):
        """Helper checks /health endpoint for server readiness."""
        assert "/health" in self.source

    # --- Profiling phases ---

    def test_two_profiling_phases(self):
        """Helper has at least two start/stop profile cycles."""
        start_count = self.source.count("start_profile(")
        stop_count = self.source.count("stop_profile(")
        # Exclude function definitions (def start_profile, def stop_profile)
        def_start = self.source.count("def start_profile(")
        def_stop = self.source.count("def stop_profile(")
        assert start_count - def_start >= 2, \
            f"Expected at least 2 start_profile calls, found {start_count - def_start}"
        assert stop_count - def_stop >= 2, \
            f"Expected at least 2 stop_profile calls, found {stop_count - def_stop}"

    def test_warmup_phase(self):
        """Helper has a warmup phase."""
        assert "warmup" in self.source.lower()

    def test_steady_state_phase(self):
        """Helper has a steady-state phase."""
        assert "steady" in self.source.lower()

    # --- Matching workload to nano-vllm ---

    def test_matching_prompt_templates(self):
        """Helper uses same PROMPT_TEMPLATES as nano-vllm for fair comparison."""
        assert "PROMPT_TEMPLATES" in self.source

    def test_prompt_templates_same_as_nano(self):
        """Helper prompt templates match nano-vllm's templates."""
        # Verify key templates from nano-vllm are present
        assert "Hello, my name is" in self.source
        assert "The capital of France is" in self.source
        assert "theory of relativity" in self.source

    def test_uses_seeded_random(self):
        """Helper uses seeded random for reproducibility."""
        assert "random.Random(seed)" in self.source

    # --- Server management ---

    def test_starts_vllm_server(self):
        """Helper starts a vLLM server process."""
        assert "subprocess.Popen" in self.source

    def test_uses_api_server(self):
        """Helper launches vllm.entrypoints.openai.api_server."""
        assert "vllm.entrypoints.openai.api_server" in self.source

    def test_server_shutdown(self):
        """Helper shuts down the server after profiling."""
        assert "terminate()" in self.source

    def test_server_enforce_eager(self):
        """Helper uses --enforce-eager for the server."""
        assert "--enforce-eager" in self.source

    def test_profiler_config_json(self):
        """Helper passes --profiler-config JSON to the server."""
        assert "--profiler-config" in self.source

    # --- CLI arguments ---

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

    def test_argparse_port(self):
        """Helper accepts --port argument."""
        assert "--port" in self.source

    def test_prints_status(self):
        """Helper prints profiler status at the end."""
        assert "get_profile_status" in self.source

    def test_if_name_main(self):
        """Helper has if __name__ == '__main__' guard."""
        assert '__name__' in self.source
        assert '__main__' in self.source
