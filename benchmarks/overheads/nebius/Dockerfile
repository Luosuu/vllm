# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}

ARG NSIGHT_SYSTEMS_PACKAGE=cuda-nsight-systems-13-0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        "${NSIGHT_SYSTEMS_PACKAGE}" rsync \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system huggingface_hub matplotlib ninja
RUN command -v nsys \
    && nsys --version \
    && python3 -c 'import triton.profiler as p; assert all(hasattr(p, n) for n in ("start", "activate", "deactivate", "finalize"))'

COPY benchmarks/overheads /opt/vllm-benchmark/benchmarks/overheads
RUN chmod +x \
    /opt/vllm-benchmark/benchmarks/overheads/nebius/run_job.sh \
    /opt/vllm-benchmark/benchmarks/overheads/nebius/submit_job.sh

WORKDIR /opt/vllm-benchmark
ENTRYPOINT ["/opt/vllm-benchmark/benchmarks/overheads/nebius/run_job.sh"]
