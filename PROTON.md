# Integrate Proton Profiler in vLLM

Proton is a built-in profiler in triton and can be imported by `import triton.profiler as proton`.
Since Triton is integrated in PyTorch, as long as PyTorch is installed, proton is available.

The goal of this project is to add proton as a new profiler backend in vLLM's (based on its latest stable release).

There are several branches that I worked on for this for previous versions of vLLM, including
 - gpt-oss-profile
 - gpt-oss-profile-flashinfer
 - llmprof
 - 064proton
 - tianle/accel_hackathon
 - proton-profile-sync
 - v0.10.1.1-llmprof

most of them are similar just with different context. Now I am looking forward to a clean integration with latest stable release.

a inital plan is that
- checkout to a new branch of vllm based on upstream latest stable release
- add triton as submodule so that we can install latest triton for up-to-date proton features and its doc.
    - https://github.com/triton-lang/triton.git
- build vllm per its development doc through uv. 
- install latest triton by `uv pip install -e submodules/triton`
- check proton doc and vLLM profiling doc
- plan detailed integration
