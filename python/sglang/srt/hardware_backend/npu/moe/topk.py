from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.utils.custom_op import register_custom_op


@register_custom_op(mutates_args=["topk_weights", "topk_ids"])
@register_split_op()
def fused_topk_npu_compute(
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    num_fused_shared_experts: int,
    correction_bias: Optional[torch.Tensor] = None,
    num_expert_group: int = 0,
    topk_group: int = 0,
    routed_scaling_factor: float = 1.0,
) -> None:
    if not use_grouped_topk and correction_bias is None:
        tw, ti, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=top_k,
        )
        if renormalize:
            w = tw if num_fused_shared_experts == 0 else tw[:, :-1]
            w = l1_norm(w)
            if num_fused_shared_experts == 0:
                tw = w
            else:
                tw = torch.cat([w, tw[:, -1:]], dim=-1)
        tw = tw.to(torch.float32)
        topk_weights.copy_(tw)
        topk_ids.copy_(ti)
    elif use_grouped_topk and correction_bias is not None:
        tw, ti, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=top_k,
            bias=correction_bias.to(torch.float32),
            k_group=topk_group,
            group_count=num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=(
                1 if renormalize else routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        topk_weights.copy_(tw)
        topk_ids.copy_(ti)
    elif correction_bias is not None:
        tw, ti, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=top_k,
            bias=correction_bias.to(torch.float32),
            renorm=0,
            norm_type=1,
            routed_scaling_factor=(
                1 if renormalize else routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        topk_weights.copy_(tw)
        topk_ids.copy_(ti)
    else:
        raise NotImplementedError(
            "fused_topk_npu_compute does not support custom_routing_function "
            "or torch_native fallback"
        )


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    if (
        topk_config.custom_routing_function is not None
        or (num_token_non_padded is None and correction_bias is not None and not use_grouped_topk)
    ):
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    num_tokens = router_logits.shape[0]
    top_k = topk_config.top_k
    out_topk_weights = torch.empty(
        num_tokens, top_k, dtype=torch.float32, device=router_logits.device
    )
    out_topk_ids = torch.empty(
        num_tokens, top_k, dtype=torch.int64, device=router_logits.device
    )

    torch.ops.sglang.fused_topk_npu_compute(
        router_logits,
        out_topk_weights,
        out_topk_ids,
        top_k,
        use_grouped_topk,
        renormalize,
        topk_config.num_fused_shared_experts,
        correction_bias,
        topk_config.num_expert_group if topk_config.num_expert_group is not None else 0,
        topk_config.topk_group if topk_config.topk_group is not None else 0,
        topk_config.routed_scaling_factor if topk_config.routed_scaling_factor is not None else 1.0,
    )

    if expert_location_dispatch_info is not None:
        out_topk_ids = topk_ids_logical_to_physical(out_topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=out_topk_ids)
    get_global_experts_capturer().capture(
        layer_id=layer_id,
        topk_ids=out_topk_ids,
    )

    return StandardTopKOutput(out_topk_weights, out_topk_ids, router_logits)
