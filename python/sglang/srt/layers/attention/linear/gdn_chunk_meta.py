"""CPU-side precomputation of GDN chunked-prefill metadata.

Motivation: the fla `prepare_chunk_indices` / `prepare_chunk_offsets` helpers
call `.tolist()` on a GPU/NPU tensor inside every layer's forward, which
introduces a D2H sync on the hot path. For GDN non-spec extend on Ascend NPU,
all inputs (`extend_seq_lens_cpu`) are already known on the host at
`init_forward_metadata` time, so we can build these indices once per step on
the CPU and asynchronously H2D-copy them into pinned memory, making them free
to consume later in every GDN layer.

Scope: this module covers the metadata that SGLang can derive from already
known host sequence lengths: `chunk_indices`, `chunk_offsets`, and host shape
values used by Ascend NPU kernels to avoid device-to-host scalar reads.
Other precomputed tensors (e.g. for large-block triu / cumulative block
indices in vllm-ascend's PR) depend on the consuming kernel's exact signature
and are intentionally left for a follow-up once the matching `sgl_kernel_npu`
side lands.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class GDNChunkedPrefillMetadata:
    """Precomputed chunk metadata for GDN chunked prefill.

    `chunk_indices` and `chunk_offsets` mirror the return values of
    `sglang.srt.layers.attention.fla.index.prepare_chunk_indices` and
    `prepare_chunk_offsets`, respectively. The host fields let external
    kernels build launch shapes without reading scalar values from device
    tensors.
    """

    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    max_T: int
    cu_seq_len: int
    query_start_loc_cpu: List[int]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def build_gdn_chunked_prefill_meta(
    extend_seq_lens_cpu: List[int],
    chunk_size: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.int32,
    use_pinned_memory: bool = True,
) -> Optional[GDNChunkedPrefillMetadata]:
    """Build chunk metadata and host shape values from per-sequence lengths.

    Returns None when there is nothing to build (empty batch), letting callers
    fall back to the legacy lazy path.

    The CPU construction exactly matches `prepare_chunk_indices` /
    `prepare_chunk_offsets` over a `cu_seqlens` whose diffs equal
    `extend_seq_lens_cpu`; see `test_gdn_chunk_meta.py`.
    """
    if not extend_seq_lens_cpu:
        return None

    # prepare_chunk_indices: for each seq of length L, produce
    # [(seq_idx, 0), (seq_idx, 1), ..., (seq_idx, ceil(L/chunk)-1)], concatenated.
    chunk_indices_rows: List[List[int]] = []
    # prepare_chunk_offsets: cumsum of chunks-per-seq, prefixed with 0.
    chunk_offsets_list: List[int] = [0]
    query_start_loc_cpu: List[int] = [0]
    running = 0
    cu_seq_len = 0
    max_T = 0
    for seq_idx, seq_len_raw in enumerate(extend_seq_lens_cpu):
        seq_len = int(seq_len_raw)
        n_chunks = _cdiv(seq_len, chunk_size)
        for k in range(n_chunks):
            chunk_indices_rows.append([seq_idx, k])
        running += n_chunks
        chunk_offsets_list.append(running)
        cu_seq_len += seq_len
        max_T = max(max_T, seq_len)
        query_start_loc_cpu.append(cu_seq_len)

    # Allocate CPU tensors (pinned when possible so the H2D copy is async).
    # Fall back to regular CPU tensors when pinning isn't supported (e.g. in
    # environments without a pinned-memory allocator).
    if chunk_indices_rows:
        chunk_indices_cpu = torch.tensor(
            chunk_indices_rows, dtype=dtype, pin_memory=False
        )
    else:
        chunk_indices_cpu = torch.empty((0, 2), dtype=dtype)
    chunk_offsets_cpu = torch.tensor(chunk_offsets_list, dtype=dtype)

    if use_pinned_memory:
        try:
            chunk_indices_cpu = chunk_indices_cpu.pin_memory()
            chunk_offsets_cpu = chunk_offsets_cpu.pin_memory()
        except (RuntimeError, NotImplementedError):
            # Pinning not supported on this backend; fall back silently.
            pass

    chunk_indices = chunk_indices_cpu.to(device, non_blocking=True)
    chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)

    return GDNChunkedPrefillMetadata(
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        max_T=max_T,
        cu_seq_len=cu_seq_len,
        query_start_loc_cpu=query_start_loc_cpu,
    )
