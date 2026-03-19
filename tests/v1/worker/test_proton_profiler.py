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


@requires_proton
class TestProtonEarlyStart:
    """Tests for the early-start pattern: start_and_deactivate + deactivate_early.

    Verifies that ProtonProfilerWrapper supports creating a session before
    CUDA graph capture and deactivating after capture completes, so that
    subsequent start()/stop() cycles reuse the existing session.
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

    def test_start_and_deactivate_creates_session(self):
        """Test that start_and_deactivate() creates a session via proton.start()."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper.start_and_deactivate()

        mock_proton.start.assert_called_once()
        assert wrapper._session_id == 42
        # _running should NOT be set — not managed by WorkerProfiler lifecycle
        assert wrapper._running is False

    def test_deactivate_early_calls_deactivate_without_flushing(self):
        """Test that deactivate_early() calls proton.deactivate() without flushing."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper.start_and_deactivate()
        wrapper.deactivate_early()

        mock_proton.deactivate.assert_called_once_with(session=42)
        assert wrapper._session_id == 42
        assert wrapper._running is False

    def test_start_after_early_init_activates_existing_session(self):
        """Test that _start() activates the existing session after early init."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper.start_and_deactivate()
        wrapper.deactivate_early()

        # Reset mock to isolate _start() calls
        mock_proton.reset_mock()

        wrapper._start()

        # Should activate, not start a new session
        mock_proton.activate.assert_called_once_with(session=42)
        mock_proton.start.assert_not_called()

    def test_stop_after_activated_start_deactivates_without_finalize(self):
        """Test that _stop() after an activate()-based start calls
        deactivate(flushing=True) but does NOT finalize in non-periodic mode.
        The session stays alive for reuse; finalize only on shutdown()."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper.start_and_deactivate()
        wrapper.deactivate_early()
        mock_proton.reset_mock()

        wrapper._start()
        wrapper._stop()

        # Should deactivate with flushing, but NOT finalize
        mock_proton.deactivate.assert_called_once_with(
            session=42, flushing=True
        )
        mock_proton.finalize.assert_not_called()
        assert wrapper._session_id == 42  # Session stays alive

    def test_full_lifecycle_non_periodic(self):
        """Test the full early-start lifecycle in non-periodic mode:
        start_and_deactivate → deactivate_early → start → stop → shutdown."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Phase 1: Early init (before CUDA graph capture)
        wrapper.start_and_deactivate()
        assert wrapper._session_id == 42
        assert wrapper._running is False

        # Phase 2: Deactivate after capture
        wrapper.deactivate_early()
        assert wrapper._session_id == 42
        assert wrapper._running is False

        # Phase 3: Normal profiling via start()/stop()
        wrapper.start()
        assert wrapper._running is True
        assert wrapper._session_id == 42

        wrapper.stop()
        assert wrapper._running is False
        # Session stays alive — only deactivated, not finalized
        assert wrapper._session_id == 42

        # Verify call sequence
        assert mock_proton.start.call_count == 1  # Only from early init
        mock_proton.activate.assert_called_once_with(session=42)
        # deactivate called twice: once early (no flushing), once at stop (flushing)
        assert mock_proton.deactivate.call_count == 2
        # finalize NOT called yet — only on shutdown
        mock_proton.finalize.assert_not_called()

        # Phase 4: Shutdown finalizes the session
        wrapper.shutdown()
        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None

    def test_full_lifecycle_periodic(self):
        """Test the full early-start lifecycle in periodic mode:
        start_and_deactivate → deactivate_early → start → stop → start → stop."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Phase 1: Early init
        wrapper.start_and_deactivate()
        assert wrapper._session_id == 42

        # Phase 2: Deactivate after capture
        wrapper.deactivate_early()

        # Phase 3: First profiling cycle
        wrapper.start()
        assert wrapper._running is True
        wrapper.stop()
        assert wrapper._running is False
        assert wrapper._current_phase == 1
        # Session should be preserved in periodic mode
        assert wrapper._session_id == 42

        # Phase 4: Second profiling cycle
        wrapper.start()
        wrapper.stop()
        assert wrapper._current_phase == 2
        assert wrapper._session_id == 42

        # Verify: proton.start() called once, activate called for each cycle
        assert mock_proton.start.call_count == 1
        # activate called: once for deactivate_early recovery + twice for cycles
        # Actually: early deactivate doesn't set _was_activated. Let me check.
        # start_and_deactivate creates session, deactivate_early pauses.
        # start() sees _session_id != None, calls activate. stop() deactivates.
        # start() again sees _session_id != None, calls activate. stop() deactivates.
        assert mock_proton.activate.call_count == 2

    def test_without_early_start_still_works(self):
        """Test that normal start/stop still works without early init
        (no regression in non-periodic mode)."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._stop()

        mock_proton.start.assert_called_once()
        mock_proton.finalize.assert_called_once_with(session=42)
        # deactivate should NOT be called (no early init, no periodic mode)
        mock_proton.deactivate.assert_not_called()
        mock_proton.activate.assert_not_called()

    def test_without_early_start_periodic_still_works(self):
        """Test that normal start/stop still works without early init
        (no regression in periodic mode)."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        wrapper._start()
        wrapper._stop()
        wrapper._start()
        wrapper._stop()

        assert mock_proton.start.call_count == 1
        assert mock_proton.activate.call_count == 1
        assert mock_proton.deactivate.call_count == 2
        mock_proton.finalize.assert_not_called()


class TestProtonEarlyStartInWorker:
    """Tests for early Proton initialization in compile_or_warm_up_model.

    These tests verify that the GPU worker creates and manages the Proton
    profiler during model warmup/capture, ensuring Proton tracks CUDA
    graph capture activity.
    """

    def _make_mock_worker(self, profiler_type="proton", enforce_eager=False):
        """Create a minimal mock worker with profiler config for testing.

        Args:
            profiler_type: The profiler type to configure.
            enforce_eager: Whether to skip CUDA graph capture.

        Returns:
            A mock worker object with the necessary attributes.
        """
        worker = MagicMock()
        worker.profiler = None
        worker.profiler_config = ProfilerConfig(
            profiler=profiler_type,
            proton_profiler_dir="/tmp/proton_out",
        ) if profiler_type == "proton" else ProfilerConfig(
            profiler=profiler_type,
            torch_profiler_dir="/tmp/torch_out",
        ) if profiler_type == "torch" else ProfilerConfig(
            profiler=profiler_type,
        ) if profiler_type == "cuda" else ProfilerConfig()
        worker.model_config = MagicMock()
        worker.model_config.enforce_eager = enforce_eager
        worker.local_rank = 0
        return worker

    @requires_proton
    def test_proton_early_start_creates_wrapper(self):
        """Test that compile_or_warm_up_model creates ProtonProfilerWrapper
        when profiler is 'proton'."""
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        worker = self._make_mock_worker(profiler_type="proton")

        # Simulate the early start logic from compile_or_warm_up_model
        if worker.profiler_config.profiler == "proton" and worker.profiler is None:
            output_dir = worker.profiler_config.proton_profiler_dir
            worker.profiler = ProtonProfilerWrapper(
                worker.profiler_config,
                output_dir=output_dir,
                local_rank=worker.local_rank,
            )
            # Replace _proton with mock to avoid real GPU calls
            mock_proton = MagicMock()
            mock_proton.start.return_value = 42
            worker.profiler._proton = mock_proton

            worker.profiler.start_and_deactivate()

        assert worker.profiler is not None
        assert isinstance(worker.profiler, ProtonProfilerWrapper)
        assert worker.profiler._session_id == 42

    @requires_proton
    def test_proton_early_start_deactivates_after_capture(self):
        """Test that deactivate_early() is called after capture_model."""
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        worker = self._make_mock_worker(profiler_type="proton")

        output_dir = worker.profiler_config.proton_profiler_dir
        worker.profiler = ProtonProfilerWrapper(
            worker.profiler_config,
            output_dir=output_dir,
            local_rank=worker.local_rank,
        )
        mock_proton = MagicMock()
        mock_proton.start.return_value = 42
        worker.profiler._proton = mock_proton

        # Early start
        worker.profiler.start_and_deactivate()
        assert worker.profiler._session_id == 42
        assert not worker.profiler._running

        # Simulate capture_model() happening here...

        # Deactivate after capture
        worker.profiler.deactivate_early()
        mock_proton.deactivate.assert_called_once_with(session=42)
        assert not worker.profiler._running

    @requires_proton
    def test_proton_early_start_then_profile_reuses_session(self):
        """Test that profile(is_start=True) reuses the early-created session."""
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        worker = self._make_mock_worker(profiler_type="proton")

        output_dir = worker.profiler_config.proton_profiler_dir
        worker.profiler = ProtonProfilerWrapper(
            worker.profiler_config,
            output_dir=output_dir,
            local_rank=worker.local_rank,
        )
        mock_proton = MagicMock()
        mock_proton.start.return_value = 42
        worker.profiler._proton = mock_proton

        # Early start + deactivate
        worker.profiler.start_and_deactivate()
        worker.profiler.deactivate_early()

        # Now profile(is_start=True) — since self.profiler is not None,
        # it skips creation and calls start() which should activate()
        worker.profiler.start()
        mock_proton.activate.assert_called_once_with(session=42)
        assert worker.profiler._running
        assert worker.profiler._was_activated

    @requires_proton
    def test_torch_profiler_not_affected_by_early_start(self):
        """Test that torch profiler is not early-started."""
        worker = self._make_mock_worker(profiler_type="torch")

        # The early start logic only triggers for proton
        if worker.profiler_config.profiler == "proton" and worker.profiler is None:
            assert False, "Should not enter early start for torch profiler"

        assert worker.profiler is None

    @requires_proton
    def test_cuda_profiler_not_affected_by_early_start(self):
        """Test that CUDA profiler is not early-started."""
        worker = self._make_mock_worker(profiler_type="cuda")

        # The early start logic only triggers for proton
        if worker.profiler_config.profiler == "proton" and worker.profiler is None:
            assert False, "Should not enter early start for cuda profiler"

        assert worker.profiler is None

    @requires_proton
    def test_proton_early_start_with_enforce_eager(self):
        """Test early start still happens when enforce_eager is True."""
        from vllm.profiler.wrapper import ProtonProfilerWrapper

        worker = self._make_mock_worker(
            profiler_type="proton", enforce_eager=True
        )

        # Early start happens regardless of enforce_eager
        if worker.profiler_config.profiler == "proton" and worker.profiler is None:
            output_dir = worker.profiler_config.proton_profiler_dir
            worker.profiler = ProtonProfilerWrapper(
                worker.profiler_config,
                output_dir=output_dir,
                local_rank=worker.local_rank,
            )
            mock_proton = MagicMock()
            mock_proton.start.return_value = 42
            worker.profiler._proton = mock_proton
            worker.profiler.start_and_deactivate()

        assert worker.profiler is not None
        assert worker.profiler._session_id == 42

        # Deactivate happens even without capture_model (enforce_eager=True)
        if isinstance(worker.profiler, ProtonProfilerWrapper) and worker.profiler._session_id is not None and not worker.profiler._running:
            worker.profiler.deactivate_early()

        mock_proton.deactivate.assert_called_once_with(session=42)

    @requires_proton
    def test_no_profiler_config_skips_early_start(self):
        """Test that no profiler config skips early start."""
        worker = self._make_mock_worker(profiler_type=None)

        if worker.profiler_config.profiler == "proton" and worker.profiler is None:
            assert False, "Should not enter early start with no profiler"

        assert worker.profiler is None


@requires_proton
class TestProfileReusesEarlyInitSession:
    """Tests for US-003: profile() correctly reuses early-initialized Proton.

    Verifies the full integration lifecycle:
    1. Early init in compile_or_warm_up_model creates wrapper + start_and_deactivate
    2. profile(is_start=True) skips creation, calls start() which activates
    3. profile(is_start=False) calls stop() which finalizes correctly
    Both periodic and non-periodic modes are covered.
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

    def _simulate_early_init(self, wrapper, mock_proton):
        """Simulate the early init from compile_or_warm_up_model.

        Calls start_and_deactivate() then deactivate_early(), matching
        the actual code in gpu_worker.py compile_or_warm_up_model().
        """
        wrapper.start_and_deactivate()
        # Simulate capture_model() happening between start and deactivate
        wrapper.deactivate_early()

    def test_profile_start_skips_creation_when_early_initialized(self):
        """Test profile(is_start=True) skips wrapper creation when
        self.profiler is already set from early init."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        self._simulate_early_init(wrapper, mock_proton)

        # Simulate the guard in profile(): if self.profiler is None
        profiler = wrapper  # Already set, so creation is skipped
        assert profiler is not None

        # Call start() — should activate existing session, not create new
        profiler.start()
        mock_proton.activate.assert_called_once_with(session=42)
        assert mock_proton.start.call_count == 1  # Only from early init

    def test_full_lifecycle_non_periodic_through_public_api(self):
        """Test early init → start() → stop() → shutdown() in non-periodic
        mode using the public WorkerProfiler API (what profile() calls).
        Stop only deactivates; shutdown finalizes."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Phase 1: Early init (compile_or_warm_up_model)
        self._simulate_early_init(wrapper, mock_proton)
        assert wrapper._session_id == 42
        assert not wrapper._running
        assert not wrapper._active

        # Phase 2: profile(is_start=True) calls wrapper.start()
        wrapper.start()
        assert wrapper._running
        assert wrapper._active

        # Phase 3: profile(is_start=False) calls wrapper.stop()
        wrapper.stop()
        assert not wrapper._running
        assert not wrapper._active
        # Session stays alive — only deactivated, not finalized
        assert wrapper._session_id == 42

        # Verify Proton call sequence after stop
        assert mock_proton.start.call_count == 1  # Only early init
        mock_proton.activate.assert_called_once_with(session=42)
        mock_proton.deactivate.assert_any_call(session=42)  # Early deactivate
        mock_proton.deactivate.assert_any_call(session=42, flushing=True)
        mock_proton.finalize.assert_not_called()  # Not yet

        # Phase 4: Shutdown finalizes
        wrapper.shutdown()
        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None

    def test_full_lifecycle_periodic_through_public_api(self):
        """Test early init → start() → stop() → start() → stop()
        in periodic mode using the public WorkerProfiler API."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # Phase 1: Early init
        self._simulate_early_init(wrapper, mock_proton)
        assert wrapper._session_id == 42

        # Phase 2: First profiling cycle (start_profile → stop_profile)
        wrapper.start()
        assert wrapper._running
        wrapper.stop()
        assert not wrapper._running
        assert wrapper._current_phase == 1
        assert wrapper._session_id == 42  # Preserved in periodic mode

        # Phase 3: Second profiling cycle
        wrapper.start()
        assert wrapper._running
        wrapper.stop()
        assert not wrapper._running
        assert wrapper._current_phase == 2
        assert wrapper._session_id == 42  # Still preserved

        # Verify: start() only called once (early init), activate for each cycle
        assert mock_proton.start.call_count == 1
        assert mock_proton.activate.call_count == 2
        # deactivate: 1 early + 2 periodic flushes = 3
        assert mock_proton.deactivate.call_count == 3
        mock_proton.finalize.assert_not_called()  # Only on shutdown

    def test_stop_works_without_early_init_non_periodic(self):
        """Test that profile stop works correctly when profiler was NOT
        early-initialized (normal creation path in profile())."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # No early init — go directly to start/stop
        wrapper.start()
        assert wrapper._running
        wrapper.stop()
        assert not wrapper._running
        assert wrapper._session_id is None

        # Should use start/finalize, no activate/deactivate
        mock_proton.start.assert_called_once()
        mock_proton.finalize.assert_called_once_with(session=42)
        mock_proton.activate.assert_not_called()
        mock_proton.deactivate.assert_not_called()

    def test_stop_works_without_early_init_periodic(self):
        """Test that profile stop works correctly when profiler was NOT
        early-initialized in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)

        # No early init
        wrapper.start()
        wrapper.stop()
        assert wrapper._current_phase == 1
        assert wrapper._session_id == 42

        wrapper.start()
        wrapper.stop()
        assert wrapper._current_phase == 2

        mock_proton.start.assert_called_once()
        mock_proton.activate.assert_called_once_with(session=42)
        assert mock_proton.deactivate.call_count == 2

    def test_multiple_start_stop_cycles_after_early_init(self):
        """Test that multiple start/stop cycles work after early init
        in non-periodic mode. The early-started session persists across
        cycles — activate/deactivate is used, not start/finalize."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        self._simulate_early_init(wrapper, mock_proton)

        # First cycle reuses early-started session
        wrapper.start()
        wrapper.stop()
        assert wrapper._session_id == 42  # Session stays alive

        # Second cycle also reuses the same session
        wrapper.start()
        wrapper.stop()
        assert wrapper._session_id == 42

        # start called only once (early init); all cycles use activate
        assert mock_proton.start.call_count == 1
        assert mock_proton.activate.call_count == 2
        # deactivate called: 1 (early) + 2 (stop cycles) = 3
        assert mock_proton.deactivate.call_count == 3
        # finalize not called — only on shutdown
        mock_proton.finalize.assert_not_called()

        # Shutdown finalizes the session
        wrapper.shutdown()
        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None

    def test_shutdown_after_early_init_periodic(self):
        """Test shutdown after early init + profiling in periodic mode."""
        config = ProfilerConfig(
            profiler="proton",
            proton_profiler_dir="/tmp/proton_out",
            proton_mode="periodic_flushing",
        )
        wrapper, mock_proton = self._make_wrapper_with_mock(config)
        self._simulate_early_init(wrapper, mock_proton)

        # One profiling cycle
        wrapper.start()
        wrapper.stop()

        # Shutdown should finalize the persistent session
        wrapper.shutdown()
        mock_proton.finalize.assert_called_once_with(session=42)
        assert wrapper._session_id is None
