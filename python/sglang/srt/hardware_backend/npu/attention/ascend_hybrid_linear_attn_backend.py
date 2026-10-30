import logging
from typing import Optional, Union

import torch
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache,
)

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


class AscendMambaAttnBackendBase(MambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.state_indices_list_gdn = []

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert max_num_tokens % max_bs == 0, (
            f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        )
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.state_indices_list_gdn.append(
                torch.full(
                    ((i + 1) * draft_token_num,),
                    self.pad_slot_id,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
            ssm_state_indices = torch.arange(
                mamba_indices.shape[0] * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1][
                : len(mamba_indices) * spec_info.draft_token_num
            ].copy_(ssm_state_indices)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
        ragged_verify_layout=None,
        *args,
        **kwargs,
    ):
        if ragged_verify_layout is not None:
            raise NotImplementedError(
                "Ascend hybrid linear attention does not support ragged verify"
            )
        # out_graph passes seq_lens_cpu=None at capture; mirror the base guard.
        if seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = torch.count_nonzero(
                seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
            )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = 0
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = mamba_track_indices
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            ssm_state_indices = torch.arange(
                bs * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1].copy_(ssm_state_indices)
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                mamba_track_indices=track_buf,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0  # Mamba attn does not use seq lens to index kv cache


class AscendMamba2AttnBackend(AscendMambaAttnBackendBase):
    pass


class AscendHybridLinearAttnBackend(HybridLinearAttnBackend):
    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AscendMambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)
        # Deferred post-verify mamba commit: run on a side stream to overlap
        # the draft window; next round's target-side consumers fence behind it
        # in init_forward_metadata[_out_graph]. Kill switch:
        # SGLANG_NPU_DEFER_MAMBA_COMMIT=0.
        self._defer_mamba_commit = envs.SGLANG_NPU_DEFER_MAMBA_COMMIT.get()
        self._mamba_commit_stream = torch.npu.Stream()
        self._mamba_commit_pending = False
        # Persistent per-role buffers holding the commit's int64 index inputs,
        # written on the caller stream so the commit stream reads no
        # allocator-managed tensor (see _commit_inputs).
        self._commit_bufs = {}
        self._commit_arange_buf = None

    def _fence_deferred_mamba_commit(self):
        # Order this round's target-side mamba state access behind the
        # deferred commit. By the time a consumer reaches here the side
        # stream has long drained, so the wait is free in practice.
        if self._mamba_commit_pending:
            torch.npu.current_stream().wait_stream(self._mamba_commit_stream)
            self._mamba_commit_pending = False

    def _commit_buf(self, key: str, shape, dtype, device) -> torch.Tensor:
        buf = self._commit_bufs.get(key)
        if buf is None or buf.shape != torch.Size(shape) or buf.device != device:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._commit_bufs[key] = buf
        return buf

    def _commit_arange(self, n: int, device: torch.device) -> torch.Tensor:
        # Slicing a cached buffer replaces the per-round allocate-and-fill.
        buf = self._commit_arange_buf
        if buf is None or buf.numel() < n or buf.device != device:
            buf = torch.arange(max(n, 64), device=device, dtype=torch.int64)
            self._commit_arange_buf = buf
        return buf[:n]

    def _commit_inputs(
        self,
        state_indices: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
    ):
        """Build the commit's index inputs as int64 in persistent buffers on
        the current stream — the copies double as the int64 casts, so the
        commit stream runs no cast/where kernels. Returns
        (dst, last, track, steps, track_srcs); the track* tensors are None
        when there is nothing to track."""
        n = last_correct_step_indices.shape[0]
        device = last_correct_step_indices.device
        dst = self._commit_buf("dst", (n,), torch.int64, device)
        dst.copy_(state_indices[:n])
        last = self._commit_buf("last", (n,), torch.int64, device)
        last.copy_(last_correct_step_indices)
        # Pre-warm the arange here so it never grows on the commit stream.
        self._commit_arange(n, device)
        if mamba_track_indices is None:
            return dst, last, None, None, None
        track = self._commit_buf(
            "track", mamba_track_indices.shape, torch.int64, device
        )
        track.copy_(mamba_track_indices)
        steps = self._commit_buf(
            "steps", mamba_steps_to_track.shape, torch.int64, device
        )
        steps.copy_(mamba_steps_to_track)
        srcs = torch.where(steps >= 0, dst, track)
        track_srcs = self._commit_buf("track_srcs", srcs.shape, torch.int64, device)
        track_srcs.copy_(srcs)
        return dst, last, track, steps, track_srcs

    def init_forward_metadata(self, forward_batch):
        self._fence_deferred_mamba_commit()
        super().init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(self, *args, **kwargs):
        # Graph-decode rounds go through here every round (metadata glue
        # calls it eagerly), covering the verify-replay consumer.
        self._fence_deferred_mamba_commit()
        super().init_forward_metadata_out_graph(*args, **kwargs)

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        dst, last, track, steps, track_srcs = self._commit_inputs(
            self.linear_attn_backend.forward_metadata.mamba_cache_indices,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        if not self._defer_mamba_commit:
            self._run_mamba_commit(dst, last, track, steps, track_srcs)
            return
        # Defer the commit onto the side stream so it overlaps the draft
        # window. Its only consumer is next round's target-side mamba state
        # access, which fences at the consumer (init_forward_metadata /
        # init_forward_metadata_out_graph) before touching the state.
        self._mamba_commit_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._mamba_commit_stream):
            self._run_mamba_commit(dst, last, track, steps, track_srcs)
        self._mamba_commit_pending = True

    def _run_mamba_commit(
        self,
        dst: torch.Tensor,
        last: torch.Tensor,
        track: Optional[torch.Tensor],
        steps: Optional[torch.Tensor],
        track_srcs: Optional[torch.Tensor],
    ):
        """Move mamba states for the accepted tokens: roll the intermediate
        ssm/conv states back to the last accepted step per request, keeping
        tracked rows whose rollback window was already consumed. All index
        inputs are int64 tensors built by _commit_inputs."""
        mamba_caches = self.linear_attn_backend.state_pool.get_speculative_mamba2_params_all_layers()

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        src = self._commit_arange(dst.shape[0], dst.device)

        move_intermediate_cache(ssm_states, intermediate_state_cache, dst, src, last)

        draft_token_num = intermediate_state_cache.shape[2]
        if track is not None:
            move_intermediate_cache(
                ssm_states, intermediate_state_cache, track, src, steps
            )

            # Track conv state from the verify-time window before rolling
            # back the working slot; NPU does not keep per-step conv
            # intermediates. Gather into a persistent buffer (no MB-level
            # temporary on the commit stream); scatter via index_put —
            # aclnnInplaceIndexCopy has no mid-dim fast path and falls back
            # to a transpose round-trip.
            gather = self._commit_buf(
                "track_gather",
                (conv_states.shape[0], track.shape[0], *conv_states.shape[2:]),
                conv_states.dtype,
                conv_states.device,
            )
            torch.index_select(conv_states, 1, track_srcs, out=gather)
            conv_states[:, track] = gather

        if dst.numel() > 0:
            conv_state_rollback(conv_states, dst, last, draft_token_num)

        if track is not None and track.numel() > 0:
            conv_state_rollback(conv_states, track, steps, draft_token_num)

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass
