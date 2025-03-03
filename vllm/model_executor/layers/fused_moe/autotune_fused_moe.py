"""Autotunable Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple
import triton.profiler as proton
import torch
import triton
import triton.language as tl
from typing import NamedTuple

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.fused_moe import grouped_topk, fused_topk

logger = init_logger(__name__)

def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)

    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         inplace=inplace,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a16=use_int8_w8a16,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale)


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None):
    if inplace:
        torch.ops.vllm.inplace_fused_experts_autotune(hidden_states, w1, w2,
                                             topk_weights, topk_ids,
                                             use_fp8_w8a8, use_int8_w8a16,
                                             w1_scale, w2_scale, a1_scale,
                                             a2_scale)
        return hidden_states
    else:
        return torch.ops.vllm.outplace_fused_experts_autotune(
            hidden_states, w1, w2, topk_weights, topk_ids, use_fp8_w8a8,
            use_int8_w8a16, w1_scale, w2_scale, a1_scale, a2_scale)

def inplace_fused_experts_autotune(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None) -> None:
    fused_experts_autotune_impl(hidden_states, w1, w2, topk_weights, topk_ids, True,
                       use_fp8_w8a8, use_int8_w8a16, w1_scale, w2_scale,
                       a1_scale, a2_scale)


def inplace_fused_experts_autotune_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts_autotune",
    op_func=inplace_fused_experts_autotune,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_autotune_fake,
)


def outplace_fused_experts_autotune(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    return fused_experts_autotune_impl(hidden_states, w1, w2, topk_weights, topk_ids,
                              False, use_fp8_w8a8, use_int8_w8a16, w1_scale,
                              w2_scale, a1_scale, a2_scale)


def outplace_fused_experts_autotune_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts_autotune",
    op_func=outplace_fused_experts_autotune,
    mutates_args=[],
    fake_impl=outplace_fused_experts_autotune_fake,
)

def fused_experts_autotune_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check constraints
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    # Allocate intermediate buffers
    intermediate_cache1 = torch.empty((num_tokens, topk_ids.shape[1], N),
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((num_tokens * topk_ids.shape[1], N // 2),
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((num_tokens, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    # First MLP + routing using autotuned dispatch
    dispatch_autotune_moe(
        tokens=hidden_states,
        expert_weights=w1,
        output=intermediate_cache1,
        topk_ids=topk_ids,
        expert_scores=topk_weights,
        mul_routed_weight=False,
        top_k=topk_ids.shape[1]
    )

    # Activation function
    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    # Second MLP using autotuned dispatch
    dispatch_autotune_moe(
        tokens=intermediate_cache2,
        expert_weights=w2,
        output = intermediate_cache3,
        topk_ids=topk_ids,
        expert_scores=topk_weights,
        mul_routed_weight=True,
        top_k=1
    )

    ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
        out_hidden_states)

    return out_hidden_states


# used by autotune moe implementation
def sort_tokens_by_expert(
    tokens: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_scores: torch.Tensor
):
    num_tokens, top_k = topk_ids.shape
    device = tokens.device

    # Track original token indices and expert positions
    original_indices = torch.arange(num_tokens, device=device)[:, None].repeat(1, top_k).flatten()
    expert_pos = torch.arange(top_k, device=device).repeat(num_tokens)

    # Flatten and sort
    flat_expert_ids = topk_ids.flatten()
    sort_order = torch.argsort(flat_expert_ids)

    return (
        tokens[original_indices[sort_order]],  # [num_tokens*top_k, hidden_dim]
        flat_expert_ids[sort_order],            # [num_tokens*top_k]
        expert_scores.flatten()[sort_order],     # [num_tokens*top_k]
        original_indices[sort_order],           # [num_tokens*top_k]
        expert_pos[sort_order]                  # [num_tokens*top_k]
    )

def dispatch_autotune_moe(
    tokens: torch.Tensor, # [num_tokens, hidden_dim]
    expert_weights: torch.Tensor, # [num_exp, in_dim (hidden_dim), out_dim]
    output: torch.Tensor, # Output tensor [num_tokens, topk, dim]
    topk_ids: torch.Tensor, # shape: [num_tokens, top_k expert ids]
    expert_scores: torch.Tensor, # [num_tokens, top_k]
    mul_routed_weight: bool, 
    top_k: int,
    # compute_type: tl.dtype,
    # use_fp8_w8a8: bool, 
    # use_int8_w8a16: bool
):
    # Sort tokens and get metadata
    sorted_tokens, sorted_expert_ids, sorted_scores, original_indices, expert_pos = sort_tokens_by_expert(
        tokens, topk_ids, expert_scores
    )

    num_experts = expert_weights.shape[0]
    # Group tokens by expert
    for expert_id in range(num_experts):
        expert_mask = (sorted_expert_ids == expert_id)
        if not expert_mask.any():
            continue

        # Get indices where this expert is used
        token_start_idx = torch.where(expert_mask)[0][0].item() # need to be Python INT
        token_end_idx = torch.where(expert_mask)[0][-1].item()  + 1 # need to be Python INT
        num_tokens_this_expert = expert_mask.sum()
        expert_dim = expert_weights.shape[2]

        # expert_dim = expert_weights.shape[2]
        # tokens_this_expert = sorted_tokens[expert_mask] # TODO: this is time-consuming, we should remove it and mask the tokens in the kernel instead
        # scores_this_expert = sorted_scores[expert_mask]
        # token_indices = original_indices[expert_mask]
        # pos_indices = expert_pos[expert_mask]
        # # TODO: check whether tokens_this_expert is empty
        # if tokens_this_expert.size(0) == 0:
        #     continue
        # # M, N = tokens.shape
        # # K, N = expert_weights.shape
        # # 1D launch grid
        # num_tokens = tokens_this_expert.shape[0]
        # expert_dim = expert_weights.shape[1]
        # output_indices = torch.where(expert_mask)[0]
        # TODO: change to lambda expression to address 
        # https://docs.astral.sh/ruff/rules/function-uses-loop-variable/
        def grid(meta):
            return (triton.cdiv(num_tokens_this_expert, meta["BLOCK_SIZE_M"]) *
                   triton.cdiv(expert_dim, meta["BLOCK_SIZE_N"]),)

        # Create temporary output buffer for this expert's results
        # output_this_expert = torch.zeros(
        #     num_tokens_this_expert, expert_weights.size(1),
        #     dtype=output.dtype, device=output.device
        # )
        autotune_moe_kernel[grid](
            tokens_ptr=sorted_tokens,  # give ptr to entire input tokens
            token_start_idx= token_start_idx, # speficy the location of tokens for this expert
            token_end_idx=token_end_idx,
            expert_weight_ptr=expert_weights, 
            expert_id = expert_id,
            scores_ptr=sorted_scores,
            output_ptr=output,
            # output_ptr=output_this_expert,
            # sorted_expert_ids_ptr=sorted_expert_ids, # maybe useless
            # scores_ptr=sorted_scores, 
            num_tokens=num_tokens_this_expert, 
            expert_dim=expert_dim, 
            hidden_dim=tokens.shape[1],
            num_experts=num_experts, # maybe useless
            stride_token=tokens.stride(0), 
            stride_input_feature_dim =tokens.stride(1),
            stride_experts=expert_weights.stride(0),
            stride_in_dim_weight=expert_weights.stride(2),
            stride_out_dim_weight=expert_weights.stride(1),
            stride_output_token=output.stride(1),
            stride_output_feature_dim=output.stride(2),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
        )

        # Get the indices for this expert
        # mask_indices = torch.where(expert_mask)[0]
        # token_indices = original_indices[mask_indices]
        # pos_indices = expert_pos[mask_indices]
        # # Scatter back to 3D output using both indices
        # output[token_indices, pos_indices] += output_this_expert

    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
    ],
    key=['num_tokens', 'hidden_dim', 'expert_dim',],
)
@triton.jit
def autotune_moe_kernel(
        tokens_ptr, # Input tensor [num_tokens, dim]
        expert_weight_ptr,  # Expert weight [dim, expert_capacity]
        expert_id, # which expert to use
        scores_ptr, # scores for this expert
        output_ptr, # Output tensor [num_tokens, dim]
        # a_scale_ptr, # used for quantization
        # b_scale_ptr, # used for quantization
        # sorted_expert_ids_ptr,  
        num_tokens, 
        token_start_idx, # token start idx for one kernel run
        token_end_idx, # token end idx for one kernel run
        expert_dim, # N, # Output feature dimension
        hidden_dim, # K, # Input feature dimension
        num_experts,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A (input tokens) has M rows).
        stride_token, # Stride for moving in token dimension of input
        stride_input_feature_dim,  # Stride for moving in feature dimension of input
        stride_experts,  # Stride between experts in weight tensor
        stride_in_dim_weight,  # Stride in input feature dimension of weights
        stride_out_dim_weight,  # Stride in output feature dimension of weights
        stride_output_token,  # Stride in token dimension of output
        stride_output_feature_dim, # Stride in feature dimension of output
        # stride_bse, # used for quantization
        # stride_bsn, # used for quantization
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        # GROUP_SIZE_M: tl.constexpr, # not useful
        # other configs
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        # compute_type: tl.constexpr, # skip quantization for now
        # use_fp8_w8a8: tl.constexpr, # skip quantization for now
        # use_int8_w8a16: tl.constexpr # skip quantization for now
        ):
    
    # Map program ids to block coordinates
    pid = tl.program_id(0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(expert_dim, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Create block offsets
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `token_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `expert_weight_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

    offs_m = token_start_idx + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # boundary handling
    token_mask = (offs_m < token_end_idx) & (offs_m >= token_start_idx)
    # input pointers
    block_token_ptrs = tokens_ptr + (offs_m[:, None] // top_k * stride_token + 
        offs_k[None, :] * stride_input_feature_dim)
    # init accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # blocked processing
    for k in range(0, tl.cdiv(hidden_dim, BLOCK_SIZE_K)):
        # block_token_ptrs = ( ## problematic
        #     tokens_ptr +
        #     offs_m[:, None] // top_k * stride_token + 
        #     (k + offs_k[None, :]) * stride_input_feature_dim
        # ) # advance pointers for input tokens
        tokens = tl.load(
            block_token_ptrs,
            mask=token_mask[:, None] & 
            (offs_k[None, :] < hidden_dim - k * BLOCK_SIZE_K),
            other=0.0
        )
        # Load corresponding expert weight block
        weight_ptrs = (
            expert_weight_ptr + expert_id * stride_experts + # go to that expert
            (k + offs_k[:, None]) * stride_in_dim_weight + # block k dim
            offs_n[None, :] * stride_out_dim_weight # block n dim
        ) # advance pointers for weight block
        weights = tl.load(
            weight_ptrs,
            mask=(k + offs_k[:, None] < hidden_dim) & (offs_n[None, :] < expert_dim),
            other=0.0
        )
        # Accumulate
        acc += tl.dot(tokens, weights, allow_tf32=True)
        # Advance pointers
        block_token_ptrs += BLOCK_SIZE_K * stride_input_feature_dim

    # Apply expert scores if needed
    if MUL_ROUTED_WEIGHT:
        scores = tl.load(
            scores_ptr + offs_m, 
            mask=token_mask,
            other=0.0
        )
        acc = acc * scores[:, None]
    # Convert back to fp16
    acc.to(tl.float16)


    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr + 
        offs_m[:, None] * stride_output_token + 
        offs_cn[None, :] * stride_output_feature_dim
    )
    output_mask = token_mask[:, None] & (offs_cn[None, :] < expert_dim)
    tl.store(
        output_ptrs,
        acc.to(tl.float16), # invalid address access, out of boundary
        mask=output_mask # share the same range mask of input
    )

def calculate_moe_kernel_input_bytes(
    A,                    # Input tensor (tokens)
    B,                    # Weight tensor (expert weights)
    C,                    # Output tensor
    A_scale,             # Optional scale for A
    B_scale,             # Optional scale for B
    topk_weights,        # Top-k weights
    sorted_token_ids,    # Sorted token indices
    expert_ids,          # Expert indices
    use_int8_w8a16=False # Flag for int8 mode
):
    total_bytes = 0

    # 1. Essential memory reads (cannot be cached in L2/SRAM)
    # Input tokens - each token is read once
    total_bytes += A.shape[0] * A.shape[1] * A.element_size()

    # Expert weights - each expert's weights are read only when needed
    # We only count the actual experts used, not the entire weight tensor
    num_unique_experts = torch.unique(expert_ids).numel()
    expert_weight_size = B.shape[1] * B.shape[2] * B.element_size()
    total_bytes += num_unique_experts * expert_weight_size

    # 2. Routing information (small, might be cached)
    total_bytes += topk_weights.numel() * topk_weights.element_size()
    total_bytes += sorted_token_ids.numel() * sorted_token_ids.element_size()
    total_bytes += expert_ids.numel() * expert_ids.element_size()

    # 3. Scales for quantization (small, likely cached)
    if A_scale is not None:
        total_bytes += A_scale.numel() * A_scale.element_size()
    if B_scale is not None and use_int8_w8a16:
        # Only count scales for actually used experts
        total_bytes += num_unique_experts * B_scale.shape[1] * B_scale.element_size()

    # 4. Output writes
    # Each token produces one output
    total_bytes += C.shape[0] * C.shape[1] * C.element_size()

    return total_bytes