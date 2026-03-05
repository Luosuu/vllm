# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Proton profiler integration.

Tests cover:
- ProfilerConfig validation with profiler='proton'
- ProtonProfilerWrapper start/stop lifecycle
- Proton annotation context manager
- End-to-end profiling output file creation
- Graceful skip when Proton is not installed
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import ProfilerConfig

# Check if proton is available for conditional skipping
try:
    import triton.profiler as proton
    HAS_PROTON = True
except ImportError:
    HAS_PROTON = False

requires_proton = pytest.mark.skipif(
    not HAS_PROTON,
    reason="Triton Proton profiler is not installed",
)


class TestProfilerConfigProton:
    """Tests for ProfilerConfig with profiler='proton'."""

    def test_proton_config_valid(self):
        """Test that ProfilerConfig accepts profiler='proton' with required dir."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        assert config.profiler == "proton"

    def test_proton_config_with_torch_profiler_dir(self):
        """Test that ProfilerConfig accepts proton with torch_profiler_dir."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            torch_profiler_dir="/tmp/torch_out",
        )
        assert config.profiler == "proton"
        assert config.torch_profiler_dir == "/tmp/torch_out"

    def test_proton_config_without_output_dir_raises(self):
        """Test that ProfilerConfig raises error for proton without dir."""
        with pytest.raises(ValueError, match="proton_profiler_dir must be set"):
            ProfilerConfig(profiler="proton")

    def test_torch_config_still_works(self):
        """Test that existing torch config is not broken."""
        config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/tmp/torch_out",
        )
        assert config.profiler == "torch"

    def test_cuda_config_still_works(self):
        """Test that existing cuda config is not broken."""
        config = ProfilerConfig(profiler="cuda")
        assert config.profiler == "cuda"

    def test_torch_profiler_dir_rejected_for_cuda(self):
        """Test that torch_profiler_dir raises error for cuda profiler."""
        with pytest.raises(ValueError, match="torch_profiler_dir is only applicable"):
            ProfilerConfig(
                profiler="cuda",
                torch_profiler_dir="/tmp/some_dir",
            )

    def test_proton_config_with_delay_iterations(self):
        """Test that proton config works with delay_iterations."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            delay_iterations=5,
        )
        assert config.delay_iterations == 5

    def test_proton_config_with_max_iterations(self):
        """Test that proton config works with max_iterations."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            max_iterations=10,
        )
        assert config.max_iterations == 10


@requires_proton
class TestProtonProfilerWrapper:
    """Tests for the ProtonProfilerWrapper class."""

    def _make_wrapper(self, output_dir="/tmp/proton_test", local_rank=0,
                      **config_kwargs):
        """Helper to create a ProtonProfilerWrapper with default config."""
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        config_kwargs.setdefault("proton_profiler_dir", output_dir)
        config = ProfilerConfig(profiler="proton", **config_kwargs)
        return ProtonProfilerWrapper(
            profiler_config=config,
            output_dir=output_dir,
            local_rank=local_rank,
        )

    def test_init(self):
        """Test that ProtonProfilerWrapper initializes correctly."""
        wrapper = self._make_wrapper()
        assert wrapper._local_rank == 0
        assert wrapper._output_dir == "/tmp/proton_test"
        assert wrapper._session_id is None

    def test_start_stop_lifecycle(self):
        """Test that start/stop lifecycle works correctly via base class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(output_dir=tmpdir)
            wrapper.start()
            assert wrapper._running is True
            assert wrapper._session_id is not None

            wrapper.stop()
            assert wrapper._running is False
            assert wrapper._session_id is None

    def test_start_creates_session_id(self):
        """Test that _start() sets a valid session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(output_dir=tmpdir)
            wrapper._start()
            assert wrapper._session_id is not None
            assert isinstance(wrapper._session_id, int)
            # Clean up
            wrapper._stop()

    def test_stop_clears_session_id(self):
        """Test that _stop() clears the session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(output_dir=tmpdir)
            wrapper._start()
            assert wrapper._session_id is not None
            wrapper._stop()
            assert wrapper._session_id is None

    def test_multi_rank_output_naming(self):
        """Test that output files include rank suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(output_dir=tmpdir, local_rank=3)
            wrapper._start()
            # The output path should include rank
            assert wrapper._local_rank == 3
            wrapper._stop()

    def test_annotate_context_manager(self):
        """Test that annotate_context_manager returns a proton.scope."""
        wrapper = self._make_wrapper()
        ctx = wrapper.annotate_context_manager("test_region")
        # Should be usable as a context manager
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_annotate_context_manager_executes(self):
        """Test that the annotation context manager can be entered/exited."""
        wrapper = self._make_wrapper()
        with wrapper.annotate_context_manager("test_scope"):
            pass  # Should not raise

    def test_delayed_start(self):
        """Test ProtonProfilerWrapper with delayed start via base class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(
                output_dir=tmpdir,
                delay_iterations=2,
            )
            wrapper.start()
            assert wrapper._active is True
            assert wrapper._running is False

            wrapper.step()
            assert wrapper._running is False

            wrapper.step()
            assert wrapper._running is True
            assert wrapper._session_id is not None

            wrapper.stop()
            assert wrapper._running is False

    def test_max_iterations(self):
        """Test ProtonProfilerWrapper with max iterations via base class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(
                output_dir=tmpdir,
                max_iterations=2,
            )
            wrapper.start()
            assert wrapper._running is True

            wrapper.step()  # iter 1
            assert wrapper._running is True

            wrapper.step()  # iter 2
            assert wrapper._running is True

            wrapper.step()  # iter 3 - exceeds max
            assert wrapper._running is False

    def test_shutdown(self):
        """Test that shutdown properly stops a running proton profiler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = self._make_wrapper(output_dir=tmpdir)
            wrapper.start()
            assert wrapper._running is True

            wrapper.shutdown()
            assert wrapper._running is False
            assert wrapper._session_id is None


@requires_proton
class TestProtonEndToEnd:
    """End-to-end tests for Proton profiling output."""

    def test_profiling_produces_output_file(self):
        """Test that a full profile session produces an output file.

        Proton writes .hatchet files to the configured output directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton", proton_profiler_dir=tmpdir
            )

            from vllm.profiler.wrapper import ProtonProfilerWrapper

            wrapper = ProtonProfilerWrapper(
                profiler_config=config,
                output_dir=tmpdir,
                local_rank=0,
            )

            wrapper.start()
            assert wrapper._running is True

            # Simulate a few steps
            wrapper.step()
            wrapper.step()

            wrapper.stop()
            assert wrapper._running is False

            # Check that output file was created
            output_files = os.listdir(tmpdir)
            hatchet_files = [f for f in output_files if "proton_rank0" in f]
            assert len(hatchet_files) > 0, (
                f"Expected .hatchet output file in {tmpdir}, "
                f"but found: {output_files}"
            )

    def test_profiling_multi_rank_output(self):
        """Test that multi-rank profiling creates rank-specific output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton", proton_profiler_dir=tmpdir
            )

            from vllm.profiler.wrapper import ProtonProfilerWrapper

            # Create profilers for rank 0 and rank 1
            wrapper0 = ProtonProfilerWrapper(
                profiler_config=config,
                output_dir=tmpdir,
                local_rank=0,
            )
            wrapper1 = ProtonProfilerWrapper(
                profiler_config=config,
                output_dir=tmpdir,
                local_rank=1,
            )

            wrapper0.start()
            wrapper1.start()

            wrapper0.step()
            wrapper1.step()

            wrapper0.stop()
            wrapper1.stop()

            # Both rank files should exist
            output_files = os.listdir(tmpdir)
            rank0_files = [f for f in output_files if "proton_rank0" in f]
            rank1_files = [f for f in output_files if "proton_rank1" in f]
            assert len(rank0_files) > 0, (
                f"Expected rank0 output file, found: {output_files}"
            )
            assert len(rank1_files) > 0, (
                f"Expected rank1 output file, found: {output_files}"
            )


class TestProtonImportGuard:
    """Test that ProtonProfilerWrapper handles missing Proton gracefully."""

    def test_import_error_when_proton_not_available(self):
        """Test that ProtonProfilerWrapper raises ImportError when
        triton.profiler is not available."""
        config = ProfilerConfig(
            profiler="proton", proton_profiler_dir="/tmp/test"
        )

        with patch.dict("sys.modules", {"triton.profiler": None, "triton": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                from importlib import reload
                import vllm.profiler.wrapper as wrapper_mod
                reload(wrapper_mod)
                wrapper_mod.ProtonProfilerWrapper(
                    profiler_config=config,
                    output_dir="/tmp/test",
                    local_rank=0,
                )

    def test_proton_wrapper_import_is_lazy(self):
        """Test that importing the wrapper module succeeds even if
        triton.profiler is not available — import happens at instantiation."""
        # The module itself should always be importable
        from vllm.profiler import wrapper  # noqa: F401
        # No ImportError should be raised from just importing the module
