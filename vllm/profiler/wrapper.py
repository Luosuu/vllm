# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Literal

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerProfiler(ABC):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        self._delay_iters = profiler_config.delay_iterations
        if self._delay_iters > 0:
            logger.info_once(
                "GPU profiling will start "
                f"{self._delay_iters} steps after start_profile."
            )

        self._max_iters = profiler_config.max_iterations
        if self._max_iters > 0:
            logger.info_once(
                "GPU profiling will stop "
                f"after {self._max_iters} worker steps, "
                "or when stop_profile is received."
            )

        # Track when the profiler gets triggered by start_profile
        self._active_iteration_count = 0
        self._active = False

        # Track when the profiler is actually running
        self._profiling_for_iters = 0
        self._running = False

    @abstractmethod
    def _start(self) -> None:
        """Start the profiler."""
        pass

    @abstractmethod
    def _stop(self) -> None:
        """Stop the profiler."""
        pass

    def _call_start(self) -> None:
        """Call _start with error handling but no safeguards."""
        try:
            self._start()
            self._running = True  # Only mark as running if start succeeds
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)

    def _call_stop(self) -> None:
        """Call _stop with error handling but no safeguards."""
        try:
            self._stop()
            logger.info_once("Profiler stopped successfully.", scope="local")
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
        self._running = False  # Always mark as not running, assume stop worked

    def start(self) -> None:
        """Attempt to start the profiler, accounting for delayed starts."""
        if self._active:
            logger.debug(
                "start_profile received when profiler is already active. "
                "Ignoring request."
            )
            return
        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def step(self) -> None:
        """Update the profiler state at each worker step,
        to handle delayed starts and max iteration limits."""
        if not self._active:
            return

        self._active_iteration_count += 1

        if (
            not self._running
            and self._delay_iters > 0
            and self._active_iteration_count == self._delay_iters
        ):
            logger.info_once("Starting profiler after delay...", scope="local")
            self._call_start()

        # Call profiler step for schedule-based profiling
        # Only count iterations where data is actually recorded (not warmup)
        if self._running and self._profiler_step():
            self._profiling_for_iters += 1

        if (
            self._max_iters > 0
            and self._running
            and self._profiling_for_iters > self._max_iters
        ):
            # Automatically stop the profiler after max iters
            # will be marked as not running, but leave as active so that stop
            # can clean up properly
            logger.info_once(
                "Max profiling iterations reached. Stopping profiler...", scope="local"
            )
            self._call_stop()
            return

    def _profiler_step(self) -> bool:
        """Called each step when profiler is running.
        Override in subclasses to handle schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        return True

    def stop(self) -> None:
        """Attempt to stop the profiler, accounting for overlapped calls."""
        if not self._active:
            logger.debug(
                "stop_profile received when profiler is not active. Ignoring request."
            )
            return
        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        if self._running:
            self._call_stop()

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        logger.info_once("Shutting down profiler", scope="local")
        if self._running:
            self.stop()

    def get_status(self) -> dict:
        """Return current profiling status.

        Base implementation returns minimal status. Subclasses may
        override to include additional fields (e.g., phase tracking).
        """
        return {
            "active": self._running,
            "current_phase": 0,
            "output_dir": "",
            "output_files": [],
        }

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}


class TorchProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
        on_trace_ready: Callable[[torch.profiler.profile], None] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        torch_profiler_trace_dir = profiler_config.torch_profiler_dir
        if local_rank in (None, 0):
            logger.info_once(
                "Torch profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
                scope="local",
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
            )

        # Determine trace handler: use custom handler if provided,
        # otherwise default to tensorboard trace handler
        if on_trace_ready is not None:
            trace_handler = on_trace_ready
        else:
            trace_handler = torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir,
                worker_name=worker_name,
                use_gzip=profiler_config.torch_profiler_use_gzip,
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1

        # Create profiler schedule if warmup or wait iterations are configured
        profiler_schedule = None
        if profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0:
            profiler_schedule = torch.profiler.schedule(
                skip_first=0,
                wait=profiler_config.wait_iterations,
                warmup=profiler_config.warmup_iterations,
                active=profiler_config.active_iterations,
                repeat=1,
            )
            if local_rank in (None, 0):
                logger.info_once(
                    "Profiler schedule configured: wait=%d, warmup=%d, active=%d",
                    profiler_config.wait_iterations,
                    profiler_config.warmup_iterations,
                    profiler_config.active_iterations,
                    scope="local",
                )

        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[activity] for activity in activities],
            schedule=profiler_schedule,
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=trace_handler,
        )

        # Track if we're using a schedule (need to call step())
        self._uses_schedule = profiler_schedule is not None
        self._warmup_iterations = profiler_config.warmup_iterations
        # Subtract 1 because profiler.start() already consumes step 0
        # (WAIT or WARMUP), so only wait + warmup - 1 non-active steps
        # remain to be advanced through via profiler.step() calls.
        self._warmup_steps_remaining = max(
            profiler_config.wait_iterations + profiler_config.warmup_iterations - 1,
            0,
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

        profiler_config = self.profiler_config
        rank = self.local_rank
        if profiler_config.torch_profiler_dump_cuda_time_total:
            profiler_dir = profiler_config.torch_profiler_dir
            sort_key = "self_cuda_time_total"
            table = self.profiler.key_averages().table(sort_by=sort_key)

            # Skip file write for URI paths (gs://, s3://, etc.)
            # as standard file I/O doesn't work with URI schemes
            if not _is_uri_path(profiler_dir):
                profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
                with open(profiler_out_file, "w") as f:
                    print(table, file=f)

            # only print profiler results on rank 0
            if rank == 0:
                print(table)
        if self.dump_cpu_time_total and rank == 0:
            logger.info(
                self.profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=50
                )
            )

    @override
    def _profiler_step(self) -> bool:
        """Call profiler.step() when using schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        if self._uses_schedule:
            self.profiler.step()
            # Track warmup steps - only count active steps toward max_iterations
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)


class ProtonProfilerWrapper(WorkerProfiler):
    """Wrapper for the Triton Proton profiler.

    Proton profiles Triton kernels and GPU activity, writing output
    in .hatchet format. The triton.profiler module is lazily imported
    to avoid errors when Triton is not installed.

    Supports two modes:
      - **Non-periodic (default):** If a session was early-started (for
        CUDA graph capture), it persists across start/stop cycles.
        _start() calls proton.activate(), _stop() calls
        proton.deactivate(flushing=True), and proton.finalize() is only
        called on shutdown(). Without an early-started session, each
        cycle creates a fresh session via proton.start() and finalizes
        it via proton.finalize().
      - **Periodic flushing:** Enabled when proton_mode starts with
        "periodic_flushing". A single session is created on the first
        _start() call and reused across subsequent cycles. _start() calls
        proton.activate(), _stop() calls proton.deactivate(flushing=True),
        and proton.finalize() is only called on shutdown().
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        output_dir: str,
        local_rank: int,
    ) -> None:
        """Initialize the Proton profiler wrapper.

        Reads all Proton-specific config fields from ProfilerConfig and
        stores them for use in _start(). Each field maps to a proton.start()
        parameter.

        If proton_mode starts with "periodic_flushing", the wrapper switches
        to persistent session lifecycle where activate/deactivate are used
        instead of start/finalize per cycle.

        Args:
            profiler_config: The profiler configuration.
            output_dir: Directory to write profiling output files.
            local_rank: The local rank of this worker, used for
                multi-rank output file naming.
        """
        super().__init__(profiler_config)

        # Lazy import to avoid errors when Triton is not installed
        import triton.profiler as proton

        self._proton = proton
        self._output_dir = output_dir
        self._local_rank = local_rank
        self._session_id: int | None = None

        # Proton config fields — map to proton.start() parameters:
        #   proton_context  -> context ("shadow" or "python")
        #   proton_data     -> data ("tree" or "trace")
        #   proton_backend  -> backend ("cupti", "roctracer", "instrumentation", None)
        #   proton_mode     -> mode (free-form backend-specific string)
        #   proton_hook     -> hook ("triton" or None)
        self._context = profiler_config.proton_context
        self._data = profiler_config.proton_data
        self._backend = profiler_config.proton_backend
        self._mode = profiler_config.proton_mode
        self._hook = profiler_config.proton_hook

        # Detect periodic flushing mode from proton_mode prefix.
        # proton_mode may be "periodic_flushing" or
        # "periodic_flushing:format=hatchet_msgpack" etc.
        self._periodic_mode = (
            self._mode is not None
            and self._mode.startswith("periodic_flushing")
        )

        # Normalize bare "periodic_flushing" to include explicit format.
        # Proton's C++ code segfaults on bare "periodic_flushing" without
        # a format suffix (out-of-bounds access in setPeriodicFlushingMode).
        if self._mode == "periodic_flushing":
            self._mode = "periodic_flushing:format=hatchet"

        # Phase tracking for periodic mode status reporting.
        # _current_phase starts at 0, increments on each _stop() in periodic mode.
        # _output_files tracks paths of generated output files.
        self._current_phase: int = 0
        self._output_files: list[str] = []

        # Tracks whether the current cycle was started via activate()
        # (from an early-started session) rather than proton.start().
        # Used in non-periodic _stop() to call deactivate before finalize.
        self._was_activated: bool = False

        if local_rank in (None, 0):
            mode_label = "periodic" if self._periodic_mode else "non-periodic"
            logger.info_once(
                "Proton profiling enabled (%s mode). "
                "Output will be saved to: %s",
                mode_label,
                output_dir,
                scope="local",
            )

    def _create_session(self) -> int:
        """Create a new Proton profiling session via proton.start().

        Returns:
            The session ID from proton.start().
        """
        import os

        # Output file is written to proton_profiler_dir with rank suffix
        output_path = os.path.join(
            self._output_dir, f"proton_rank{self._local_rank}"
        )
        return self._proton.start(
            name=output_path,
            context=self._context,
            data=self._data,
            backend=self._backend,
            mode=self._mode,
            hook=self._hook,
        )

    def start_and_deactivate(self) -> None:
        """Create a Proton session for early initialization.

        Creates the session via proton.start() so that Proton tracks
        GPU activity (e.g., CUDA graph capture) from this point forward.
        The caller must later call deactivate_early() to pause the session,
        and the normal start()/stop() API to resume profiling.

        This method does NOT set _running — the session is active at the
        Proton level but not yet managed by the WorkerProfiler lifecycle.
        """
        self._session_id = self._create_session()

    def deactivate_early(self) -> None:
        """Pause the early-started Proton session without flushing data.

        Calls proton.deactivate() without flushing so the session can be
        reactivated later via the normal start() API. After this call,
        _session_id is set but _running is False.
        """
        self._proton.deactivate(session=self._session_id)

    @override
    def _start(self) -> None:
        """Start or reactivate a Proton profiling session.

        If a session already exists (from start_and_deactivate() or a
        previous periodic cycle), reactivates it via proton.activate().
        Otherwise creates a new session via proton.start().
        """
        if self._session_id is not None:
            # Reactivate existing session (early-started or periodic reuse)
            self._proton.activate(session=self._session_id)
            self._was_activated = True
        else:
            # Create a new session
            self._session_id = self._create_session()
            self._was_activated = False

    @override
    def _stop(self) -> None:
        """Stop or deactivate the Proton profiling session.

        Non-periodic mode: If the session was early-started (activated
        from a pre-existing session), only deactivates with flushing to
        keep the session alive for reuse. Fresh sessions (no early start)
        are finalized and destroyed.
        Periodic mode: Deactivates the session with flushing=True to
        write per-phase .part_N output files. The session is preserved
        for reuse; finalize() is only called on shutdown().
        """
        if self._periodic_mode:
            # Deactivate with flushing to write per-phase output
            self._proton.deactivate(
                session=self._session_id, flushing=True
            )
            # Advance phase counter and collect output file path
            self._proton.data.advance_phase(session=self._session_id)
            self._scan_output_files()
            self._current_phase += 1
            # Session is preserved — not cleared
        else:
            if self._was_activated:
                # Early-started session: just deactivate, keep session
                # alive for reuse. Finalize happens only on shutdown().
                self._proton.deactivate(
                    session=self._session_id, flushing=True
                )
                self._scan_output_files()
            else:
                # Fresh session: finalize and destroy
                self._proton.finalize(session=self._session_id)
                self._scan_output_files()
                self._session_id = None
            self._was_activated = False

    def _scan_output_files(self) -> None:
        """Scan proton_profiler_dir for output files matching this rank.

        Updates _output_files with any new files found. Files are
        identified by the "proton_rank{N}" prefix in the output directory.
        """
        import os

        if not os.path.isdir(self._output_dir):
            return
        prefix = f"proton_rank{self._local_rank}"
        current_files = sorted(
            os.path.join(self._output_dir, f)
            for f in os.listdir(self._output_dir)
            if f.startswith(prefix)
        )
        self._output_files = current_files

    def get_status(self) -> dict:
        """Return current profiling status for status endpoint reporting.

        Returns a dict with:
            - active: Whether the profiler is currently running.
            - current_phase: Current phase number (periodic mode).
            - output_dir: Directory where output files are written.
            - output_files: List of output file paths generated so far.
        """
        return {
            "active": self._running,
            "current_phase": self._current_phase,
            "output_dir": self._output_dir,
            "output_files": list(self._output_files),
        }

    @override
    def shutdown(self) -> None:
        """Shut down the profiler, finalizing any persistent session.

        Stops profiling if still running, then finalizes any session
        that is still alive (periodic sessions, or early-started
        non-periodic sessions that were only deactivated in _stop()).
        """
        logger.info_once("Shutting down profiler", scope="local")
        if self._running:
            if self._periodic_mode:
                self._active = False
                self._active_iteration_count = 0
                self._profiling_for_iters = 0
                self._call_stop()
            else:
                self.stop()
        # Finalize any persistent session (periodic or early-started)
        if self._session_id is not None:
            self._proton.finalize(session=self._session_id)
            self._scan_output_files()
            self._session_id = None

    @override
    def annotate_context_manager(self, name: str):
        """Return a Proton scope context manager for region annotation.

        Args:
            name: The name of the annotated region.

        Returns:
            A proton.scope context manager.
        """
        return self._proton.scope(name)


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        super().__init__(profiler_config)
        # Note: lazy import to avoid dependency issues if CUDA is not available.
        import torch.cuda.profiler as cuda_profiler

        self._cuda_profiler = cuda_profiler

    @override
    def _start(self) -> None:
        self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        return torch.cuda.nvtx.range(name)
