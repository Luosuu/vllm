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

    def test_invalid_proton_context_raises(self):
        """Test that invalid proton_context value raises validation error."""
        with pytest.raises(ValueError, match="proton_context must be one of"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_context="invalid",
            )

    def test_invalid_proton_data_raises(self):
        """Test that invalid proton_data value raises validation error."""
        with pytest.raises(ValueError, match="proton_data must be one of"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_data="invalid",
            )

    def test_invalid_proton_backend_raises(self):
        """Test that invalid proton_backend value raises validation error."""
        with pytest.raises(ValueError, match="proton_backend must be one of"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_backend="invalid",
            )

    def test_invalid_proton_hook_raises(self):
        """Test that invalid proton_hook value raises validation error."""
        with pytest.raises(ValueError, match="proton_hook must be one of"):
            ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_hook="invalid",
            )

    def test_proton_mode_accepts_arbitrary_strings(self):
        """Test that proton_mode accepts any string without validation error."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="any-arbitrary-string-123",
        )
        assert config.proton_mode == "any-arbitrary-string-123"

    def test_proton_config_valid_all_fields(self):
        """Test valid config with all Proton fields explicitly set."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_context="python",
            proton_data="trace",
            proton_backend="cupti",
            proton_mode="custom-mode",
            proton_hook="triton",
        )
        assert config.proton_context == "python"
        assert config.proton_data == "trace"
        assert config.proton_backend == "cupti"
        assert config.proton_mode == "custom-mode"
        assert config.proton_hook == "triton"

    def test_proton_profiler_dir_converted_to_absolute(self):
        """Test that relative proton_profiler_dir is converted to absolute."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="relative/path",
        )
        assert os.path.isabs(config.proton_profiler_dir)

    def test_proton_profiler_dir_uri_not_converted(self):
        """Test that URI paths for proton_profiler_dir are not converted."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="gs://bucket/proton_out",
        )
        assert config.proton_profiler_dir == "gs://bucket/proton_out"

    def test_proton_valid_backend_values(self):
        """Test all valid proton_backend values are accepted."""
        for backend in ("cupti", "roctracer", "instrumentation", None):
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_backend=backend,
            )
            assert config.proton_backend == backend

    def test_proton_valid_context_values(self):
        """Test all valid proton_context values are accepted."""
        for context in ("shadow", "python"):
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_context=context,
            )
            assert config.proton_context == context

    def test_proton_valid_data_values(self):
        """Test all valid proton_data values are accepted."""
        for data in ("tree", "trace"):
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_data=data,
            )
            assert config.proton_data == data

    def test_proton_valid_hook_values(self):
        """Test all valid proton_hook values are accepted."""
        for hook in ("triton", None):
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir="/tmp/proton_out",
                proton_hook=hook,
            )
            assert config.proton_hook == hook


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


@requires_proton
class TestProtonConfigWiring:
    """Integration tests that Proton config fields are correctly passed
    to proton.start() arguments via ProtonProfilerWrapper."""

    def _make_wrapper_with_mock(self, config, output_dir="/tmp/proton_out",
                                local_rank=0):
        """Create a ProtonProfilerWrapper then replace _proton with a mock.

        Returns (wrapper, mock_proton) tuple.
        """
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        wrapper = ProtonProfilerWrapper(
            profiler_config=config,
            output_dir=output_dir,
            local_rank=local_rank,
        )
        mock_proton = MagicMock()
        mock_proton.start.return_value = 42
        wrapper._proton = mock_proton
        return wrapper, mock_proton

    def test_config_fields_passed_to_proton_start(self):
        """Test that all config fields are correctly forwarded to proton.start().

        Replaces the _proton module reference with a mock to verify the exact
        arguments passed to proton.start() match the ProfilerConfig values.
        """
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_context="python",
            proton_data="trace",
            proton_backend="cupti",
            proton_mode="custom-mode",
            proton_hook="triton",
        )

        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        wrapper._start()

        # Verify proton.start() was called with all config fields
        mock_proton.start.assert_called_once()
        call_kwargs = mock_proton.start.call_args
        assert call_kwargs.kwargs["context"] == "python"
        assert call_kwargs.kwargs["data"] == "trace"
        assert call_kwargs.kwargs["backend"] == "cupti"
        assert call_kwargs.kwargs["mode"] == "custom-mode"
        assert call_kwargs.kwargs["hook"] == "triton"
        # name should include the output dir and rank
        assert "proton_rank0" in call_kwargs.kwargs["name"]

    def test_config_defaults_passed_to_proton_start(self):
        """Test that default config values are forwarded to proton.start()."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )

        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        wrapper._start()

        call_kwargs = mock_proton.start.call_args
        # Defaults: context=shadow, data=tree, backend=None, mode=None, hook=None
        assert call_kwargs.kwargs["context"] == "shadow"
        assert call_kwargs.kwargs["data"] == "tree"
        assert call_kwargs.kwargs["backend"] is None
        assert call_kwargs.kwargs["mode"] is None
        assert call_kwargs.kwargs["hook"] is None

    def test_stop_calls_finalize_with_session_id(self):
        """Test that _stop() calls proton.finalize() with the correct session ID."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )

        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        mock_proton.start.return_value = 99
        wrapper._start()
        assert wrapper._session_id == 99

        wrapper._stop()
        mock_proton.finalize.assert_called_once_with(session=99)
        assert wrapper._session_id is None


@requires_proton
class TestProtonPhaseTracking:
    """Tests for phase tracking and get_status() in ProtonProfilerWrapper."""

    def _make_wrapper_with_mock(self, config, output_dir="/tmp/proton_out",
                                local_rank=0):
        """Create a ProtonProfilerWrapper then replace _proton with a mock.

        Returns (wrapper, mock_proton) tuple.
        """
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        wrapper = ProtonProfilerWrapper(
            profiler_config=config,
            output_dir=output_dir,
            local_rank=local_rank,
        )
        mock_proton = MagicMock()
        mock_proton.start.return_value = 42
        wrapper._proton = mock_proton
        return wrapper, mock_proton

    def test_phase_increments_on_stop_periodic(self):
        """Test that phase number increments on each _stop() in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # First start/stop cycle
        wrapper._start()
        assert wrapper._current_phase == 0
        wrapper._stop()
        assert wrapper._current_phase == 1

        # Second start/stop cycle
        wrapper._start()
        wrapper._stop()
        assert wrapper._current_phase == 2

    def test_phase_does_not_increment_non_periodic(self):
        """Test that phase number stays at 0 in non-periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._stop()
        assert wrapper._current_phase == 0

    def test_advance_phase_called_in_periodic_mode(self):
        """Test that proton.data.advance_phase() is called on _stop() in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._stop()

        mock_proton.data.advance_phase.assert_called_once_with(session=42)

    def test_advance_phase_not_called_non_periodic(self):
        """Test that proton.data.advance_phase() is NOT called in non-periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._stop()

        mock_proton.data.advance_phase.assert_not_called()

    def test_get_status_initial(self):
        """Test get_status() returns correct initial state."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, _ = self._make_wrapper_with_mock(config)

        status = wrapper.get_status()
        assert status["active"] is False
        assert status["current_phase"] == 0
        assert status["output_dir"] == "/tmp/proton_out"
        assert status["output_files"] == []

    def test_get_status_while_running(self):
        """Test get_status() returns active=True while profiler is running."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, _ = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._running = True  # Simulate base class setting this

        status = wrapper.get_status()
        assert status["active"] is True
        assert status["current_phase"] == 0

    def test_get_status_after_periodic_cycles(self):
        """Test get_status() returns correct phase after multiple cycles."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, _ = self._make_wrapper_with_mock(config)

        # Run three start/stop cycles
        for _ in range(3):
            wrapper._start()
            wrapper._stop()

        status = wrapper.get_status()
        assert status["active"] is False
        assert status["current_phase"] == 3

    def test_get_status_output_files_periodic(self):
        """Test get_status() tracks output files in periodic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=tmpdir,
                proton_mode="periodic_flushing",
            )
            wrapper, _ = self._make_wrapper_with_mock(config, output_dir=tmpdir)

            # Create fake output files to simulate Proton writing them
            for i in range(2):
                open(os.path.join(tmpdir, f"proton_rank0.part_{i}.hatchet"), "w").close()

            wrapper._start()
            wrapper._stop()

            status = wrapper.get_status()
            assert len(status["output_files"]) == 2
            assert all("proton_rank0" in f for f in status["output_files"])

    def test_get_status_output_files_non_periodic(self):
        """Test get_status() tracks output files in non-periodic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=tmpdir,
            )
            wrapper, _ = self._make_wrapper_with_mock(config, output_dir=tmpdir)

            # Create fake output file
            open(os.path.join(tmpdir, "proton_rank0.hatchet"), "w").close()

            wrapper._start()
            wrapper._stop()

            status = wrapper.get_status()
            assert len(status["output_files"]) == 1

    def test_get_status_multi_rank_filters_by_rank(self):
        """Test that get_status() only includes files for this rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=tmpdir,
            )
            wrapper, _ = self._make_wrapper_with_mock(
                config, output_dir=tmpdir, local_rank=1
            )

            # Create files for both ranks
            open(os.path.join(tmpdir, "proton_rank0.hatchet"), "w").close()
            open(os.path.join(tmpdir, "proton_rank1.hatchet"), "w").close()

            wrapper._start()
            wrapper._stop()

            status = wrapper.get_status()
            assert len(status["output_files"]) == 1
            assert "proton_rank1" in status["output_files"][0]

    def test_session_reused_across_periodic_cycles(self):
        """Test that session is created once and reused in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # First cycle creates session
        wrapper._start()
        assert mock_proton.start.call_count == 1
        wrapper._stop()

        # Second cycle reuses session (activate, not start)
        wrapper._start()
        assert mock_proton.start.call_count == 1  # Still 1
        mock_proton.activate.assert_called_once_with(session=42)
        wrapper._stop()

    def test_shutdown_calls_finalize_periodic(self):
        """Test that shutdown() calls finalize() in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._running = True
        wrapper._active = True
        wrapper.shutdown()

        mock_proton.finalize.assert_called_once_with(session=42)


@requires_proton
class TestContiguousProfiling:
    """Tests for contiguous profiling with periodic flushing mode.

    Verifies the complete call sequences for both periodic and non-periodic
    modes across multiple start/stop cycles, ensuring:
    - Periodic mode uses activate/deactivate instead of start/finalize
    - Non-periodic mode uses start/finalize per cycle (no regression)
    - Phase number increments correctly
    - Session reuse in periodic mode
    - Shutdown behavior in both modes
    """

    def _make_wrapper_with_mock(self, config, output_dir="/tmp/proton_out",
                                local_rank=0):
        """Create a ProtonProfilerWrapper then replace _proton with a mock.

        Returns (wrapper, mock_proton) tuple.
        """
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        wrapper = ProtonProfilerWrapper(
            profiler_config=config,
            output_dir=output_dir,
            local_rank=local_rank,
        )
        mock_proton = MagicMock()
        mock_proton.start.return_value = 42
        wrapper._proton = mock_proton
        return wrapper, mock_proton

    def test_periodic_uses_activate_deactivate(self):
        """Test that periodic mode uses activate/deactivate, not start/finalize."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # First cycle: start creates session, stop deactivates
        wrapper._start()
        wrapper._stop()

        # start() called once (creates session), finalize() never called
        mock_proton.start.assert_called_once()
        mock_proton.finalize.assert_not_called()
        # deactivate called with flushing=True
        mock_proton.deactivate.assert_called_once_with(
            session=42, flushing=True
        )

        # Second cycle: activate (not start), deactivate
        wrapper._start()
        mock_proton.activate.assert_called_once_with(session=42)
        assert mock_proton.start.call_count == 1  # Still just 1

        wrapper._stop()
        assert mock_proton.deactivate.call_count == 2
        mock_proton.finalize.assert_not_called()  # Still never called

    def test_non_periodic_uses_start_finalize(self):
        """Test that non-periodic mode uses start/finalize per cycle (no regression)."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # First cycle
        wrapper._start()
        wrapper._stop()

        mock_proton.start.assert_called_once()
        mock_proton.finalize.assert_called_once_with(session=42)
        # activate/deactivate should NOT be called
        mock_proton.activate.assert_not_called()
        mock_proton.deactivate.assert_not_called()

        # Second cycle: start/finalize again
        wrapper._start()
        assert mock_proton.start.call_count == 2
        wrapper._stop()
        assert mock_proton.finalize.call_count == 2

    def test_periodic_full_sequence_three_cycles(self):
        """Test full call sequence across three periodic start/stop cycles."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Cycle 1: start creates session
        wrapper._start()
        wrapper._stop()
        assert wrapper._current_phase == 1

        # Cycle 2: activate reuses session
        wrapper._start()
        wrapper._stop()
        assert wrapper._current_phase == 2

        # Cycle 3: activate reuses session
        wrapper._start()
        wrapper._stop()
        assert wrapper._current_phase == 3

        # Verify call counts
        assert mock_proton.start.call_count == 1
        assert mock_proton.activate.call_count == 2
        assert mock_proton.deactivate.call_count == 3
        assert mock_proton.data.advance_phase.call_count == 3
        mock_proton.finalize.assert_not_called()

    def test_periodic_deactivate_flushing_true(self):
        """Test that deactivate is always called with flushing=True in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        for _ in range(3):
            wrapper._start()
            wrapper._stop()

        # All deactivate calls should have flushing=True
        for call in mock_proton.deactivate.call_args_list:
            assert call.kwargs["flushing"] is True

    def test_periodic_session_id_persists_across_cycles(self):
        """Test that session ID is created once and never cleared in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        assert wrapper._session_id == 42
        wrapper._stop()
        assert wrapper._session_id == 42  # NOT cleared

        wrapper._start()
        assert wrapper._session_id == 42
        wrapper._stop()
        assert wrapper._session_id == 42  # Still persists

    def test_non_periodic_session_id_cleared_each_cycle(self):
        """Test that session ID is cleared after each stop in non-periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        assert wrapper._session_id == 42
        wrapper._stop()
        assert wrapper._session_id is None  # Cleared

    def test_shutdown_calls_finalize_periodic(self):
        """Test that shutdown() calls finalize() in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._running = True
        wrapper._active = True
        wrapper.shutdown()

        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None

    def test_shutdown_non_periodic_stops_normally(self):
        """Test that shutdown() in non-periodic mode calls stop (which finalizes)."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._running = True
        wrapper._active = True
        wrapper.shutdown()

        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None
        assert wrapper._running is False

    def test_get_status_periodic_mode_accuracy(self):
        """Test get_status() returns accurate state across periodic cycles."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Initial status
        status = wrapper.get_status()
        assert status["active"] is False
        assert status["current_phase"] == 0

        # After first cycle
        wrapper._start()
        wrapper._running = True
        status = wrapper.get_status()
        assert status["active"] is True

        wrapper._stop()
        wrapper._running = False
        status = wrapper.get_status()
        assert status["active"] is False
        assert status["current_phase"] == 1

        # After second cycle
        wrapper._start()
        wrapper._stop()
        status = wrapper.get_status()
        assert status["current_phase"] == 2

    def test_get_status_output_files_tracked(self):
        """Test get_status() correctly tracks output files across cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="proton",
                proton_profiler_dir=tmpdir,
                proton_mode="periodic_flushing",
            )
            wrapper, mock_proton = self._make_wrapper_with_mock(
                config, output_dir=tmpdir
            )

            # Simulate Proton writing output files
            for i in range(3):
                open(os.path.join(
                    tmpdir, f"proton_rank0.part_{i}.hatchet"
                ), "w").close()

            wrapper._start()
            wrapper._stop()

            status = wrapper.get_status()
            assert len(status["output_files"]) == 3
            assert status["output_dir"] == tmpdir

    def test_periodic_flushing_with_format_suffix(self):
        """Test that periodic_flushing:format=... mode string is correctly detected."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing:format=hatchet_msgpack",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        assert wrapper._periodic_mode is True

        # Should use activate/deactivate pattern
        wrapper._start()
        wrapper._stop()
        wrapper._start()

        mock_proton.start.assert_called_once()
        mock_proton.activate.assert_called_once_with(session=42)
        mock_proton.deactivate.assert_called_once_with(
            session=42, flushing=True
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
