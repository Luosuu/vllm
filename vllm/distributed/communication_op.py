from typing import Any, Dict, Optional, Union
import triton.profiler as proton
import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    with proton.scope("tensor_model_parallel_all_reduce", metrics={
        "bytes(exc)": calculate_allreduce_bytes(input_)
    }):
        output = get_tp_group().all_reduce(input_)
    return output


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    with proton.scope("tensor_model_parallel_all_gather", metrics={
        "bytes(exc)": calculate_allgather_bytes(input_)
    }):
        output = get_tp_group().all_gather(input_, dim)
    return output


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    with proton.scope("tensor_model_parallel_gather", metrics={
        "bytes(exc)": calculate_gather_bytes(input_, dst)
    }):
        output = get_tp_group().gather(input_, dst, dim)
    return output


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    with proton.scope("broadcast_tensor_dict", metrics={
        "bytes(exc)": calculate_broadcast_dict_bytes(tensor_dict, src)
    }):
        output = get_tp_group().broadcast_tensor_dict(tensor_dict, src)
    return output



def calculate_allreduce_bytes(input_: torch.Tensor) -> int:
    # Get world size from the group coordinator
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    # Total bytes = (send + receive) * (world_size - 1) * tensor_size
    return 2 * (world_size - 1) * input_.numel() * input_.element_size()

def calculate_allgather_bytes(input_: torch.Tensor) -> int:
    # Get world size from the group coordinator
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    # Total bytes = send to all + receive from all
    return world_size * input_.numel() * input_.element_size()

def calculate_gather_bytes(input_: torch.Tensor, dst: int) -> int:
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    rank = tp_group.rank if hasattr(tp_group, 'rank') else tp_group.get_rank()

    if rank == dst:
        # Receive from all other ranks
        return (world_size - 1) * input_.numel() * input_.element_size()
    else:
        # Send to dst rank only
        return input_.numel() * input_.element_size()

def calculate_broadcast_dict_bytes(tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]], 
                                 src: int) -> int:
    if tensor_dict is None:
        return 0

    tp_group = get_tp_group()
    world_size = tp_group.world_size
    rank = tp_group.rank if hasattr(tp_group, 'rank') else tp_group.get_rank()

    total_bytes = 0
    for v in tensor_dict.values():
        if isinstance(v, torch.Tensor):
            if rank == src:
                # Source sends to all other ranks
                total_bytes += (world_size - 1) * v.numel() * v.element_size()
            else:
                # Other ranks receive from source
                total_bytes += v.numel() * v.element_size()
    return total_bytes