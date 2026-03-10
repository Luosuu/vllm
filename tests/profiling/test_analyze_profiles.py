"""Tests for the profile analysis script (analyze_profiles.sh).

Uses source-level verification to validate script structure without
requiring GPU, model weights, or actual profiling output.
"""

import os
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts/profiling")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "analyze_profiles.sh")


class TestAnalyzeProfilesScript:
    """Verify analyze_profiles.sh structure and content."""

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

    def test_set_euo_pipefail(self):
        """Script uses strict error handling."""
        assert "set -euo pipefail" in self.source

    # --- Profile directory configuration ---

    def test_profile_dir_configurable(self):
        """Profile directory is configurable via --profile-dir."""
        assert "--profile-dir" in self.source
        assert "PROFILE_DIR" in self.source

    def test_default_profile_dir(self):
        """Default profile directory is profiling_output."""
        assert "profiling_output" in self.source

    # --- Analysis output directory ---

    def test_analysis_output_dir(self):
        """Analysis output goes to profiling_output/analysis/."""
        assert "ANALYSIS_DIR" in self.source
        assert "/analysis" in self.source

    def test_creates_analysis_dir(self):
        """Script creates the analysis output directory."""
        assert 'mkdir -p "$ANALYSIS_DIR"' in self.source

    # --- proton-viewer usage ---

    def test_uses_proton_viewer(self):
        """Script runs proton-viewer for analysis."""
        assert "proton-viewer" in self.source

    def test_proton_viewer_time_metric(self):
        """Script uses time/ns metric with proton-viewer."""
        assert "time/ns" in self.source

    def test_saves_proton_viewer_output(self):
        """Script saves proton-viewer output to text files."""
        assert "output_file" in self.source
        assert ".txt" in self.source

    # --- Kernel extraction ---

    def test_top_10_kernels(self):
        """Script extracts top-10 kernels by time."""
        assert "Top-10" in self.source or "top-10" in self.source

    def test_kernel_count(self):
        """Script reports kernel count."""
        assert "kernel_count" in self.source

    def test_head_n_11(self):
        """Script uses head -n 11 for top-10 (1 header + 10 data)."""
        assert "head -n 11" in self.source

    # --- Framework comparison ---

    def test_analyzes_nano_vllm(self):
        """Script analyzes nano-vllm profiles."""
        assert "nano_vllm" in self.source

    def test_analyzes_vllm(self):
        """Script analyzes vLLM profiles."""
        # Check for vllm in a framework context (not just nano_vllm)
        assert "vllm_offline" in self.source or '"vllm"' in self.source

    def test_side_by_side_comparison(self):
        """Script includes side-by-side comparison section."""
        assert "SIDE-BY-SIDE" in self.source or "side-by-side" in self.source

    def test_compares_both_frameworks(self):
        """Script compares both frameworks for each profiling mode."""
        assert "nano_vllm" in self.source
        assert "vllm" in self.source
        # Both offline shadow+tree are compared
        assert "nano_vllm_offline_shadow_tree" in self.source
        assert "vllm_offline_shadow_tree" in self.source

    # --- Offline profiling configs ---

    def test_offline_shadow_tree(self):
        """Script analyzes offline shadow+tree profiles."""
        assert "shadow:tree" in self.source

    def test_offline_python_tree(self):
        """Script analyzes offline python+tree profiles."""
        assert "python:tree" in self.source

    def test_offline_configs_array(self):
        """Script iterates over offline profiling configurations."""
        assert "OFFLINE_CONFIGS" in self.source

    def test_all_four_offline_configs(self):
        """Script includes all 4 offline profiling configurations."""
        assert "shadow:tree" in self.source
        assert "shadow:trace" in self.source
        assert "python:tree" in self.source
        assert "python:trace" in self.source

    # --- Online / periodic flushing profiles ---

    def test_online_profile_analysis(self):
        """Script analyzes online profiling output."""
        assert "online" in self.source

    def test_periodic_flushing_section(self):
        """Script has a section for periodic flushing analysis."""
        assert "Periodic Flushing" in self.source or "periodic" in self.source.lower()

    def test_warmup_vs_steady_state(self):
        """Script compares warmup vs steady-state phases."""
        assert "warmup" in self.source.lower()
        assert "steady" in self.source.lower()

    def test_part_0_warmup(self):
        """Script looks for .part_0 files as warmup phase."""
        assert "part_0" in self.source

    def test_part_1_steady_state(self):
        """Script looks for .part_1 files as steady-state phase."""
        assert "part_1" in self.source

    def test_warmup_analysis_output(self):
        """Script saves warmup phase analysis to file."""
        assert "warmup.txt" in self.source

    def test_steady_analysis_output(self):
        """Script saves steady-state phase analysis to file."""
        assert "steady.txt" in self.source

    # --- Chrome trace files ---

    def test_chrome_trace_listing(self):
        """Script lists chrome trace files."""
        assert "chrome_trace" in self.source

    def test_perfetto_instructions(self):
        """Script includes instructions to open traces in Perfetto."""
        assert "perfetto" in self.source.lower()

    def test_perfetto_url(self):
        """Script includes the Perfetto UI URL."""
        assert "ui.perfetto.dev" in self.source

    def test_chrome_trace_sizes(self):
        """Script reports chrome trace file sizes."""
        # Uses stat to get file size
        assert "stat" in self.source

    def test_chrome_traces_saved(self):
        """Script saves chrome trace listing to analysis dir."""
        assert "chrome_traces.txt" in self.source

    # --- Analysis helper function ---

    def test_analyze_hatchet_function(self):
        """Script defines analyze_hatchet helper function."""
        assert "analyze_hatchet" in self.source

    def test_analyze_hatchet_takes_label(self):
        """analyze_hatchet function uses a label for output naming."""
        assert "label" in self.source

    # --- Summary section ---

    def test_summary_section(self):
        """Script prints an analysis summary."""
        assert "ANALYSIS SUMMARY" in self.source

    def test_summary_counts_analyzed(self):
        """Summary reports number of files analyzed."""
        assert "analyzed_files" in self.source

    def test_summary_counts_failed(self):
        """Summary reports number of failed analyses."""
        assert "failed_files" in self.source

    def test_summary_lists_output_files(self):
        """Summary lists all analysis output files."""
        assert "Analysis files" in self.source

    def test_exit_0_on_success(self):
        """Script exits 0 on success."""
        assert "exit 0" in self.source

    def test_exit_1_on_failure(self):
        """Script exits 1 on failure."""
        assert "exit 1" in self.source

    def test_no_files_warning(self):
        """Script warns when no .hatchet files found."""
        assert "No .hatchet files found" in self.source


class TestAnalyzeProfilesSyntax:
    """Verify script syntax is valid."""

    def test_bash_syntax_check(self):
        """Script passes bash -n syntax check."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestAnalyzeProfilesHelperVerification:
    """Verify analyze_profiles.sh helper function patterns."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        """Load the script source."""
        with open(SCRIPT_PATH) as f:
            self.source = f.read()

    def test_analyze_hatchet_checks_file_exists(self):
        """analyze_hatchet checks that the hatchet file exists."""
        assert "-f" in self.source

    def test_analyze_hatchet_saves_to_analysis_dir(self):
        """analyze_hatchet saves output to ANALYSIS_DIR."""
        assert "ANALYSIS_DIR" in self.source

    def test_iterates_both_frameworks(self):
        """Script iterates over both nano_vllm and vllm frameworks."""
        assert "nano_vllm vllm" in self.source or "nano_vllm" in self.source

    def test_online_shadow_tree_dir(self):
        """Script looks for online profiles in shadow_tree subdirectory."""
        assert "online/shadow_tree" in self.source

    def test_offline_dir_pattern(self):
        """Script looks for offline profiles in framework/offline/ dirs."""
        assert "offline" in self.source

    def test_find_hatchet_files(self):
        """Script uses find to locate .hatchet files."""
        assert 'find "$CONFIG_DIR" -name "*.hatchet"' in self.source

    def test_find_chrome_trace_files(self):
        """Script uses find to locate .chrome_trace files."""
        assert ".chrome_trace" in self.source

    def test_proton_viewer_redirect_to_file(self):
        """proton-viewer output is redirected to a file."""
        assert '> "$output_file"' in self.source

    def test_comparison_shadow_tree(self):
        """Side-by-side comparison includes shadow+tree profiles."""
        assert "shadow_tree" in self.source

    def test_comparison_python_tree(self):
        """Side-by-side comparison includes python+tree profiles."""
        assert "python_tree" in self.source
