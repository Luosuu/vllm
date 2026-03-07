"""Tests for the master profiling script (run_all.sh) and README.md.

Uses source-level verification to validate script structure without
requiring GPU, model weights, or actual profiling output.
"""

import os
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts/profiling")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "run_all.sh")
README_PATH = os.path.join(SCRIPT_DIR, "README.md")


class TestRunAllScript:
    """Verify run_all.sh structure and content."""

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

    def test_set_euo_pipefail(self):
        """Script uses strict error handling."""
        assert "set -euo pipefail" in self.source

    # --- Sequence: sanity check first ---

    def test_runs_sanity_check_first(self):
        """Sanity check runs before offline and online profiling."""
        sanity_pos = self.source.find("sanity_check.sh")
        offline_pos = self.source.find("nano_vllm_offline.sh")
        online_pos = self.source.find("nano_vllm_online.sh")
        analysis_pos = self.source.find("analyze_profiles.sh")
        assert sanity_pos < offline_pos < online_pos < analysis_pos

    def test_sanity_check_aborts_on_failure(self):
        """Script aborts if sanity check fails."""
        assert "sanity check must pass" in self.source.lower() or \
               "Aborting" in self.source

    # --- Sub-script invocations ---

    def test_calls_sanity_check(self):
        """Script calls sanity_check.sh."""
        assert "sanity_check.sh" in self.source

    def test_calls_nano_vllm_offline(self):
        """Script calls nano_vllm_offline.sh."""
        assert "nano_vllm_offline.sh" in self.source

    def test_calls_vllm_offline(self):
        """Script calls vllm_offline.sh."""
        assert "vllm_offline.sh" in self.source

    def test_calls_nano_vllm_online(self):
        """Script calls nano_vllm_online.sh."""
        assert "nano_vllm_online.sh" in self.source

    def test_calls_vllm_online(self):
        """Script calls vllm_online.sh."""
        assert "vllm_online.sh" in self.source

    def test_calls_analyze_profiles(self):
        """Script calls analyze_profiles.sh."""
        assert "analyze_profiles.sh" in self.source

    # --- Argument parsing ---

    def test_model_arg(self):
        """Script accepts --model argument."""
        assert "--model" in self.source

    def test_sanity_model_arg(self):
        """Script accepts --sanity-model argument."""
        assert "--sanity-model" in self.source

    def test_batch_size_arg(self):
        """Script accepts --batch-size argument."""
        assert "--batch-size" in self.source

    def test_input_len_arg(self):
        """Script accepts --input-len argument."""
        assert "--input-len" in self.source

    def test_output_len_arg(self):
        """Script accepts --output-len argument."""
        assert "--output-len" in self.source

    def test_seed_arg(self):
        """Script accepts --seed argument."""
        assert "--seed" in self.source

    def test_num_prompts_arg(self):
        """Script accepts --num-prompts argument."""
        assert "--num-prompts" in self.source

    def test_warmup_prompts_arg(self):
        """Script accepts --warmup-prompts argument."""
        assert "--warmup-prompts" in self.source

    def test_port_arg(self):
        """Script accepts --port argument."""
        assert "--port" in self.source

    # --- Skip flags ---

    def test_skip_sanity_flag(self):
        """Script supports --skip-sanity flag."""
        assert "--skip-sanity" in self.source

    def test_skip_offline_flag(self):
        """Script supports --skip-offline flag."""
        assert "--skip-offline" in self.source

    def test_skip_online_flag(self):
        """Script supports --skip-online flag."""
        assert "--skip-online" in self.source

    def test_skip_analysis_flag(self):
        """Script supports --skip-analysis flag."""
        assert "--skip-analysis" in self.source

    # --- Default workload parameters ---

    def test_default_model(self):
        """Default model is Qwen3-32B."""
        assert 'MODEL="Qwen/Qwen3-32B"' in self.source

    def test_default_sanity_model(self):
        """Default sanity model is Qwen3-0.6B."""
        assert 'SANITY_MODEL="Qwen/Qwen3-0.6B"' in self.source

    def test_default_batch_size(self):
        """Default batch size is 4."""
        assert "BATCH_SIZE=4" in self.source

    def test_default_input_len(self):
        """Default input length is 2048."""
        assert "INPUT_LEN=2048" in self.source

    def test_default_output_len(self):
        """Default output length is 128."""
        assert "OUTPUT_LEN=128" in self.source

    def test_default_seed(self):
        """Default seed is 42."""
        assert "SEED=42" in self.source

    # --- Parameter forwarding ---

    def test_forwards_model_to_offline(self):
        """Script forwards --model to offline scripts."""
        assert '--model "$MODEL"' in self.source

    def test_forwards_batch_size_to_offline(self):
        """Script forwards --batch-size to offline scripts."""
        assert '--batch-size "$BATCH_SIZE"' in self.source

    def test_forwards_seed_to_offline(self):
        """Script forwards --seed to offline scripts."""
        assert '--seed "$SEED"' in self.source

    def test_forwards_num_prompts_to_online(self):
        """Script forwards --num-prompts to online scripts."""
        assert '--num-prompts "$NUM_PROMPTS"' in self.source

    def test_forwards_warmup_prompts_to_online(self):
        """Script forwards --warmup-prompts to online scripts."""
        assert '--warmup-prompts "$WARMUP_PROMPTS"' in self.source

    def test_forwards_port_to_vllm_online(self):
        """Script forwards --port to vLLM online script."""
        assert '--port "$PORT"' in self.source

    # --- Summary ---

    def test_prints_final_summary(self):
        """Script prints a final summary."""
        assert "FINAL SUMMARY" in self.source

    def test_tracks_passed_steps(self):
        """Script tracks passed steps."""
        assert "steps_passed" in self.source

    def test_tracks_failed_steps(self):
        """Script tracks failed steps."""
        assert "steps_failed" in self.source

    def test_exit_code_on_success(self):
        """Script exits with 0 on success."""
        assert "exit 0" in self.source

    def test_exit_code_on_failure(self):
        """Script exits with 1 on failure."""
        assert "exit 1" in self.source


class TestRunAllSyntax:
    """Verify run_all.sh has valid bash syntax."""

    def test_bash_syntax(self):
        """Script passes bash -n syntax check."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestReadme:
    """Verify README.md content meets acceptance criteria."""

    @pytest.fixture(autouse=True)
    def _load_readme(self):
        """Load the README source."""
        with open(README_PATH) as f:
            self.source = f.read()

    def test_readme_exists(self):
        """README.md exists at expected path."""
        assert os.path.isfile(README_PATH)

    # --- Required sections ---

    def test_hardware_setup_section(self):
        """README documents hardware setup."""
        assert "## Hardware Setup" in self.source

    def test_hardware_mentions_h100(self):
        """README mentions H100 GPU."""
        assert "H100" in self.source

    def test_hardware_mentions_80gb(self):
        """README mentions 80GB GPU memory."""
        assert "80GB" in self.source

    def test_model_section(self):
        """README documents model."""
        assert "## Model" in self.source

    def test_model_mentions_qwen3_32b(self):
        """README mentions Qwen3-32B."""
        assert "Qwen3-32B" in self.source

    def test_workload_parameters_section(self):
        """README documents workload parameters."""
        assert "## Workload Parameters" in self.source

    def test_workload_batch_size(self):
        """README documents batch size."""
        assert "Batch size" in self.source or "batch_size" in self.source

    def test_workload_input_length(self):
        """README documents input length."""
        assert "2048" in self.source

    def test_workload_output_length(self):
        """README documents output length."""
        assert "128" in self.source

    def test_workload_seed(self):
        """README documents random seed."""
        assert "42" in self.source

    def test_how_to_run_section(self):
        """README documents how to run."""
        assert "## How to Run" in self.source

    def test_how_to_run_mentions_run_all(self):
        """README mentions run_all.sh."""
        assert "run_all.sh" in self.source

    # --- Quantitative comparison table ---

    def test_quantitative_comparison(self):
        """README includes quantitative comparison table."""
        assert "Quantitative Comparison" in self.source

    def test_comparison_gpu_time(self):
        """README mentions GPU time metric."""
        assert "GPU time" in self.source

    def test_comparison_kernel_count(self):
        """README mentions kernel count metric."""
        assert "Kernel count" in self.source or "kernel count" in self.source

    def test_comparison_top_kernel(self):
        """README mentions top kernel time."""
        assert "Top kernel" in self.source or "top kernel" in self.source

    def test_comparison_scheduling_overhead(self):
        """README mentions scheduling overhead."""
        assert "Scheduling overhead" in self.source or "scheduling overhead" in self.source

    # --- Shadow vs python context ---

    def test_shadow_vs_python_section(self):
        """README includes shadow vs python context observations."""
        assert "Shadow vs Python" in self.source or "shadow" in self.source.lower()

    def test_shadow_context_description(self):
        """README describes shadow context behavior."""
        assert "Shadow" in self.source and "GPU kernel" in self.source

    def test_python_context_description(self):
        """README describes python context behavior."""
        assert "Python" in self.source and "call stack" in self.source

    # --- Tree vs trace format ---

    def test_tree_vs_trace_section(self):
        """README includes tree vs trace format observations."""
        assert "Tree vs Trace" in self.source or "tree" in self.source.lower()

    def test_tree_format_description(self):
        """README describes tree format."""
        assert ".hatchet" in self.source and "proton-viewer" in self.source

    def test_trace_format_description(self):
        """README describes trace format."""
        assert ".chrome_trace" in self.source and "Perfetto" in self.source

    # --- Periodic flushing comparison ---

    def test_periodic_flushing_section(self):
        """README includes periodic flushing warmup vs steady-state comparison."""
        assert "Warmup" in self.source and "Steady" in self.source

    def test_warmup_phase_description(self):
        """README describes warmup phase characteristics."""
        assert "warmup" in self.source.lower() and "part_0" in self.source

    def test_steady_state_description(self):
        """README describes steady-state phase characteristics."""
        assert "steady" in self.source.lower() and "part_1" in self.source

    # --- Output directory documentation ---

    def test_output_directories_section(self):
        """README documents output directories."""
        assert "## Output Directories" in self.source

    def test_output_dir_nano_vllm_offline(self):
        """README documents nano-vllm offline output directory."""
        assert "nano_vllm" in self.source and "offline" in self.source

    def test_output_dir_vllm_offline(self):
        """README documents vLLM offline output directory."""
        assert "vllm" in self.source and "offline" in self.source

    def test_output_dir_online(self):
        """README documents online output directory."""
        assert "online" in self.source

    def test_output_dir_analysis(self):
        """README documents analysis output directory."""
        assert "analysis" in self.source

    def test_file_naming_hatchet(self):
        """README documents .hatchet file naming convention."""
        assert ".hatchet" in self.source

    def test_file_naming_chrome_trace(self):
        """README documents .chrome_trace file naming convention."""
        assert ".chrome_trace" in self.source

    def test_file_naming_part_n(self):
        """README documents .part_N naming convention for phases."""
        assert ".part_" in self.source

    def test_file_naming_rank(self):
        """README documents proton_rank naming convention."""
        assert "proton_rank" in self.source

    # --- Script reference ---

    def test_script_reference(self):
        """README lists all profiling scripts."""
        for script in [
            "run_all.sh",
            "sanity_check.sh",
            "nano_vllm_offline.sh",
            "vllm_offline.sh",
            "nano_vllm_online.sh",
            "vllm_online.sh",
            "analyze_profiles.sh",
        ]:
            assert script in self.source, f"Missing script reference: {script}"
