import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
# W4A8_MXFP block (group) size — fixed at 32 by the msmodelslim export format.
MXFP4_BLOCK_SIZE = 32


def _dequant_e4m3fn_to_float32(u8: torch.Tensor) -> torch.Tensor:
    """Decode float8_e4m3fn payload bytes to FP32 by explicit bit layout.

    Direct ``.to(torch.float32)`` on NPU fp8 tensors fails on some CANN builds
    (aclnnInplaceCopy error 561103), so decode manually. Only used at
    weight-load time (and the defensive pre-quantized-input path), where the
    handful of extra elementwise ops is irrelevant.
    """
    bits = u8.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 0x80, -1.0, 1.0)
    e = ((bits >> 3) & 0xF).to(torch.float32)
    m = (bits & 0x7).to(torch.float32)
    mag = torch.where(
        e == 0.0,
        m * (1.0 / 512.0),  # subnormal: m * 2^-9
        (1.0 + m / 8.0) * torch.pow(2.0, e - 7.0),
    )
    # e4m3fn reserves S.1111.111 for NaN (no infinities); max finite is 448.
    mag = torch.where((e == 15.0) & (m == 7.0), float("nan"), mag)
    return sign * mag


# NPU ops are reached via torch.ops.npu.* (registered when torch_npu is imported
# by the runtime), so this module needs no top-level `import torch_npu` and stays
# importable on CUDA/CPU/AMD/XPU CI.
def _get_float8_e8m0fnu_dtype():
    # Resolve lazily rather than as a module-level constant: this module is
    # imported early (during quant-scheme registration), so reading the dtype at
    # call time keeps it correct regardless of import order / platform.
    return getattr(torch, "float8_e8m0fnu", None)


def _get_float4_e2m1fn_x2_dtype():
    # The packed-FP4 dtype MUST come from torch_npu (an int enum, e.g. 296), not
    # from torch. The NPU ops that consume it -- npu_dynamic_mx_quant(dst_type=),
    # npu_quant_matmul(x2_dtype=), npu_format_cast(input_dtype=) -- REJECT the
    # torch dtype object torch.float4_e2m1fn_x2 in op-plugin on recent torch_npu
    # builds (it raises, or with None gives "output y must be same shape as input
    # x"), even though torch.float4_e2m1fn_x2 exists. This is fp4-specific: fp8 /
    # float8_e8m0fnu is accepted from torch either way. Verified on A5 /
    # torch_npu 2.10.0.post2.dev20260704 (see llm/probe_fp4_w4a8_chain.py: dst=296
    # passes the full quant->format_cast->matmul chain, dst=torch dtype fails).
    #
    # Lazy import so this NPU-only path keeps the module importable on
    # CUDA/CPU/AMD/XPU CI (no top-level torch_npu; see AGENTS.md known pitfalls).
    from sglang.srt.utils import is_npu

    if is_npu():
        import torch_npu

        npu_dtype = getattr(torch_npu, "float4_e2m1fn_x2", None)
        if npu_dtype is not None:
            return npu_dtype
    return getattr(torch, "float4_e2m1fn_x2", None)


class _NPULinearMethodBase(LinearMethodBase):
    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch.ops.npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch.ops.npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
        else:
            original_dtype = x.dtype
            quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP8LinearMethod(_NPULinearMethodBase):
    """NPU MXFP8 linear method for LLM (SRT) models.

    Shared kernel for both the online config path (``--quantization mxfp8``) and
    the offline ModelSlimMXFP8Scheme (which delegates to this as ``self.kernel``).
    process_weights_after_loading branches on weight dtype: FP16/BF16 weights are
    quantised to MXFP8 at load time (online); pre-quantised float8_e4m3fn weights
    are only re-laid-out (offline). Inference: dynamic MXFP8 activation quant +
    MXFP8 matmul (block_size=32).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise later in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype == torch.float8_e4m3fn:
            # Offline (ModelSlim) path: weight is already MXFP8-quantised and
            # layer.weight_scale holds the uint8 block scales [out, in/32]. Only
            # re-layout to [in, out] / [in//64, out, 2] strided views below.
            n_dim, k_dim = layer.weight_scale.data.shape
            scale = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
            layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                scale.transpose(0, 1), requires_grad=False
            )
            # weight_scale is now folded into weight_scale_inv (which keeps the
            # underlying storage alive via its view); drop the stale parameter so
            # it doesn't linger in named_parameters() / state_dict().
            del layer.weight_scale
        else:
            # Online path: quantise FP16/BF16 weights to MXFP8 at load time.
            if weight.dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "NPUMXFP8LinearMethod: weight dtype %s is not float16/bfloat16; "
                    "casting to bfloat16 before MXFP8 quantisation.",
                    weight.dtype,
                )
                weight = weight.to(torch.bfloat16)
            # Move weight to NPU if needed (cpu offload may move it back to CPU).
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            # Online MXFP8 quantisation of weights (block_size=32).
            # qw: [out, in] float8_e4m3fn, w_scale: [out, in//64, 2] uint8.
            qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn
            )
            layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                w_scale.transpose(0, 1), requires_grad=False
            )

        # Both paths produce weight [in, out] and weight_scale_inv [in//64, out,
        # 2] as strided transpose views — DO NOT call .contiguous(). The matmul
        # reduction loop scans the in-dim per output column; the [out, in]
        # row-major source gives stride-1 access for that scan via the transpose
        # view, matching msmodelslim's offline layout. Calling .contiguous()
        # physically reorders to [in, out] row-major, making the inner-loop stride
        # equal to out and tanking HBM bandwidth.

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # MXFP8 matmul (weight & scale already transposed at load time)
        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        if envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            layer.weight.data = layer.weight.data.view(torch.int32).contiguous()
        else:
            layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
                layer.weight.data.to(torch.int32)
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP4W4A8LinearMethod(_NPULinearMethodBase):
    """NPU W4A8 online quantization: MXFP4 weights + MXFP8 activations.

    This is a *true* W4(weight) A8(activation) path: it mirrors the offline
    ``W4A8_MXFP`` kernel (``NPUMXFP4W4A8OfflineLinearMethod``) exactly — the only
    difference is that the FP4 weights are produced online from BF16/FP16
    (round-to-nearest, no calibration) instead of being loaded from a msmodelslim
    checkpoint. An earlier version of this method ran a *dual-level* scheme that
    also compressed the activation to FP4 (W4A4 compute via
    ``npu_dual_level_quant_matmul``); that was a large accuracy regression — 4-bit
    activations — so it was replaced with the single-level FP8-activation path
    below, aligned with the offline W4A8 implementation.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2) → packed FP4
        + UE8M0 block scale → npu_format_cast to FRACTAL_NZ → transpose [in//2, out]

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, FP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    Hardware: A5 NPU + a recent torch_npu with the FP4 npu_quant_matmul
    (same requirement as the offline W4A8 path — see that class's docstring).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register an unquantized (``params_dtype``) weight placeholder.

        Online quantization needs its own ``create_weights`` because the
        checkpoint still holds full-precision BF16/FP16 weights: the loader
        fills this buffer, then ``process_weights_after_loading`` quantizes it to
        MXFP4 in place. This differs from the offline/int8 methods, whose weights
        are created by the scheme's own ``create_weights`` to match the
        already-quantized (FP8 / uint8-packed) layout the checkpoint provides.
        """
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise to MXFP4 in
        # process_weights_after_loading.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Online single-level MXFP4 weight quant, then lay the weight out exactly
        # like the offline W4A8 path so the same npu_quant_matmul(x2_dtype=fp4)
        # kernel accepts it. All NPU ops go through torch.ops.npu.* (no torch_npu).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        # npu_dynamic_mx_quant returns the scale as [out, in//64, 2] (3D); older
        # builds may return [out, in//32] (2D) — handle both before the transpose.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )

        # weight: packed FP4 -> FRACTAL_NZ (float8_e4m3fn view) -> transpose
        # [in//2, out]. Mirror the offline path (no .contiguous() on the NZ view);
        # view as uint8 first because npu_format_cast only accepts int-dtype tensors.
        qw_nz = npu_format_cast(
            qw.view(torch.uint8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight = Parameter(qw_nz.transpose(-1, -2), requires_grad=False)

        # weight_scale -> [in//64, out, 2] to match npu_quant_matmul.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8 — FP8, not FP4).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        # True W4(weight)A8(activation) matmul, identical to the offline path.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUMXFP4W4A8OfflineLinearMethod(_NPULinearMethodBase):
    """NPU offline W4A8 (ModelSlim ``W4A8_MXFP``): packed-FP4 weights + MXFP8 activations.

    Kernel for the offline ModelSlimMXFP4W4A8Scheme (delegated as ``self.kernel``).
    The msmodelslim ``W4A8_MXFP`` checkpoint stores weights as *packed FP4*
    (``pack_fp4_to_uint8`` → ``uint8`` shape ``[out, in//2]``) plus UE8M0 block
    scales (``uint8`` shape ``[out, in//group_size]``):

      process_weights_after_loading:
        weight (uint8 packed FP4 [out, in//2]) → npu_format_cast(29,
            customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2) → FRACTAL_NZ
            → transpose [in//2, out]
        weight_scale [out, in/32] → reshape [out, in/64, 2] → transpose → [in/64, out, 2]

      apply:
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, MXFP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    The weight is cast to FRACTAL_NZ then transposed; ``npu_dynamic_mx_quant`` already
    returns a 3D ``[tokens, in//64, 2]`` block scale so the matmul needs no extra
    scale-layout normalization.

    ⚠️ REQUIRES a recent torch_npu build for the FP4 ``npu_quant_matmul``. On the
    A5 this device forces ``allow_internal_format=False`` (the NZ cast still produces
    a ``FRACTAL_NZ_C0_16`` tensor, which is fine). Older torch_npu (e.g.
    ``2.10.0.dev20260320``) had a broken FP4 matmul that rejected the NZ weight in
    *prefill* with ``x2 should be in ... nz format, but it is 2``;
    ``2.10.0.post1.dev20260624`` (and later) runs the FRACTAL_NZ path correctly.
    If you hit ``it is 2``, update torch_npu — do NOT "fix" it by
    switching the weight to ND.

    ⚠️ A ``atb::OperationSetup`` *segfault during decode* (not prefill) is a
    DIFFERENT, unrelated issue: it is the eager-decode ``ascend`` attention
    backend, NOT this matmul (verified by stage-sync bisection — qkv's matmul
    syncs clean, the fault surfaces at the entry-sync of the next layer, i.e. the
    decode attention between qkv and o_proj). Run with the NPU decode graph (do
    NOT pass ``--disable-cuda-graph``); graph mode is the NPU default. This
    attention issue is model-agnostic and out of scope for W4A8.

    This is a true W4(weight) A8(activation) single-level matmul. The *online*
    ``NPUMXFP4W4A8LinearMethod`` now uses this exact apply path — the only
    difference is that it quantizes BF16/FP16 weights to FP4 at load time instead
    of loading them from a msmodelslim checkpoint. ``group_size`` is fixed at 32
    by the ``W4A8_MXFP`` export format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Cast the packed-FP4 weight to FRACTAL_NZ then transpose. All NPU ops go
        # through torch.ops.npu.* (no torch_npu). A recent torch_npu build is
        # required (see class docstring): older builds reject the NZ weight
        # ("x2 ... it is 2").
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        # weight: packed-FP4 uint8 [out, in//2] -> FRACTAL_NZ (float8_e4m3fn view)
        # -> transpose to [in//2, out].
        layer.weight.data = npu_format_cast(
            layer.weight.data,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        # weight_scale: [out, in/32] uint8 -> [in/64, out, 2].
        n, k = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(
            n, k // 2, 2
        ).transpose(-3, -2)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        # W4(weight)A8(activation) matmul.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUSingleLevelMXFP4LinearMethod(_NPULinearMethodBase):
    """NPU W4A4 online quantization: single-level MXFP4.

    True W4(weight) A4(activation): both weights and activations are quantised to
    single-level MXFP4 (``float4_e2m1fn_x2``), unlike the W4A8 path which keeps FP8
    activations. All NPU ops go through ``torch.ops.npu.*`` (no top-level
    ``torch_npu``) and the fp4 dtype comes from ``_get_float4_e2m1fn_x2_dtype()``.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)
        → (packed FP4 [out, in//2], UE8M0 block scale) → transpose [in//2, out]

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)  (A4)
        → npu_quant_matmul(x1_dtype = x2_dtype = float4_e2m1fn_x2,
                           group_sizes=[1, 1, MXFP4_BLOCK_SIZE])

    Triggered by ``--quantization mxfp4`` on NPU. Hardware: A5 NPU with a recent
    torch_npu exposing ``float4_e2m1fn_x2``.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register an unquantized (``params_dtype``) weight placeholder.

        The checkpoint still holds full-precision BF16/FP16 weights: the loader
        fills this buffer, then ``process_weights_after_loading`` quantizes it to
        MXFP4 in place.
        """
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise to MXFP4 in
        # process_weights_after_loading.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Online single-level MXFP4 weight quant. All NPU ops go through
        # torch.ops.npu.* (no torch_npu); the fp4 dtype comes from the shared
        # _get_float4_e2m1fn_x2_dtype() helper (the torch_npu int enum).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )
        # Pre-transpose the weight to [in//2, out] for npu_quant_matmul; use
        # .data= to preserve the non-contiguous transpose view (npu_quant_matmul
        # reads strides directly — .contiguous() would reorder data and break
        # block-scale alignment).
        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight.data = layer.weight.data.transpose(0, 1)

        # weight_scale -> [in//64, out, 2] (3D), matching the offline W4A4 and
        # W4A8 paths. npu_dynamic_mx_quant
        # already returns the scale as [out, in//64, 2] (3D) on current builds;
        # older builds may return [out, in//32] (2D) — reshape those first so the
        # transpose always yields the 3D layout npu_quant_matmul requires.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic single-level MXFP4 activation quantisation (A4 — FP4).
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=fp4_dtype, round_mode="round"
        )

        # Single-level MXFP4 matmul (weight & scale already transposed at load
        # time): x1_dtype = x2_dtype = fp4, group_sizes=[1, 1, block].
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            x1_dtype=fp4_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[1, 1, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUSingleLevelMXFP4OfflineLinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """NPU offline W4A4 (ModelSlim ``W4A4_MXFP4``): packed FP4 weights.

    Kernel for the offline ``ModelSlimMXFP4Scheme`` (delegated as ``self.kernel``).
    The msmodelslim ``W4A4_MXFP4`` checkpoint stores weights as packed ``uint8``
    [out, in//2] (two FP4 values per byte) plus UE8M0 block scales (``uint8``
    [out, in//32]). The weight is transposed and the scale reshaped to 3D; it then
    shares the online :class:`NPUSingleLevelMXFP4LinearMethod` matmul (``apply``)
    exactly — only the weight source differs (msmodelslim checkpoint vs online RTN).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        # The checkpoint is already packed two-FP4-per-byte. Preserve the strided
        # transpose used by the online path.
        layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)

        weight_scale = layer.weight_scale.data
        if not weight_scale.is_npu:
            weight_scale = weight_scale.to(f"npu:{torch.npu.current_device()}")
        # npu_quant_matmul with float4_e2m1fn_x2 requires x2Scale to be 3D:
        # [out, in/32] -> [out, in/64, 2] -> transpose to [in/64, out, 2].
        n_dim, k_dim = weight_scale.shape
        layer.weight_scale = Parameter(
            weight_scale.reshape(n_dim, k_dim // 2, 2).transpose(0, 1),
            requires_grad=False,
        )


class NPUDualLevelMXFP4LinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """NPU W4A4 online quantization: dual-level MXFP4 (higher accuracy).

    This is the sole online ``--quantization mxfp4`` linear path. Instead of a single
    UE8M0 (power-of-2) block scale, dual-level MX quant produces a finer L0 (FP8 E4M3)
    block scale plus a coarser L1 scale, so per-block dynamic range is captured far
    more accurately — this fixed the online-RTN degradation that made single-level
    decoding loop (never emitting EOS) under greedy sampling. (The single-level
    :class:`NPUSingleLevelMXFP4LinearMethod` is retained only as the offline path's
    base — msmodelslim checkpoints ship single-level UE8M0 scales.)

    All NPU ops go through ``torch.ops.npu.*`` (no top-level ``torch_npu``). Only
    ``create_weights`` (the BF16/FP16 placeholder) is shared with the single-level
    base; weight post-processing and the matmul are fully dual-level.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_dual_level_mx_quant
        → (packed FP4 weight, L0 scale, L1 scale); weight cast to FRACTAL_NZ,
          L0 scale transposed to [in//l0_block, out].

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_dual_level_mx_quant  (A4, dual-level)
        → npu_dual_level_quant_matmul(act, weight, act_l0, w_l0, act_l1, w_l1)

    Reference: Diffusion ``NPUMXFP4DiffusionLinearMethod``. Hardware: A5 NPU
    only — the ``DualLevelQuantBatchMatmul`` op is unavailable on A2/A3.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Dual-level MXFP4 weight quant: packed FP4 weight + L0 (fine, FP8 E4M3)
        # and L1 (coarse) block scales.
        qw, w_l0_scale, w_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            weight_fp, smooth_scale=None
        )

        # npu_dual_level_quant_matmul requires the weight (x2) in FRACTAL_NZ.
        # View the packed FP4 as int8 first (npu_format_cast takes int dtypes).
        qw_nz = npu_format_cast(
            qw.view(torch.int8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.int8,
        )

        # L0 scale -> [in//l0_block, out] (op returns [out, in//l0_block, 1]).
        w_l0_scale = w_l0_scale.squeeze(-1).transpose(0, 1).contiguous()

        layer.weight = Parameter(qw_nz, requires_grad=False)
        layer.weight_l0_scale = Parameter(w_l0_scale, requires_grad=False)
        layer.weight_l1_scale = Parameter(w_l1_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for the quant operators.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic dual-level MXFP4 activation quant (A4): packed FP4 + L0/L1 scales.
        qx, act_l0_scale, act_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level matmul. Arg order (act, weight, act_l0, w_l0, act_l1, w_l1);
        # the weight is NOT transposed here (unlike the single-level path).
        output = torch.ops.npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_l0_scale,
            act_l1_scale,
            layer.weight_l1_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUBlockFP8LinearMethod(_NPULinearMethodBase):
    """NPU block-FP8 (e.g. 128x128) dense linear — dequant-to-BF16 route.

    On NPU the generic ``Fp8LinearMethod`` dispatches ``w8a8_block_fp8_linear``
    to the CUDA Triton tile GEMM, which cannot use the NPU Cube units and runs
    far slower than a native matmul (verified by profiling). Until a native W8A8
    block-scale GEMM lands, dequantise the FP8 weights once at load time
    (fp8 weight * expanded block scale -> BF16) and run a plain BF16 matmul.

    Trade-off: 2x weight memory for dense layers, in exchange for native Cube
    throughput. Weight creation is delegated to ``Fp8LinearMethod.create_fp8_weight_``
    so checkpoint loading (fp8 payload + ``weight_scale_inv`` [N//bn, K//bk]) is
    unchanged; only ``process_weights_after_loading`` differs.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__(quant_config)
        self.weight_block_size = quant_config.weight_block_size
        self.is_checkpoint_fp8_serialized = quant_config.is_checkpoint_fp8_serialized

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        Fp8LinearMethod.create_fp8_weight_(
            layer,
            block_quant=True,
            quant_config=self.quant_config,
            use_mxfp8=False,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            input_size=input_size,
            output_size=output_size,
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            params_dtype=params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype in (torch.float16, torch.bfloat16):
            # Unquantized checkpoint — keep as-is, apply() runs a plain matmul.
            return

        block_n, block_k = self.weight_block_size
        n, k = weight.shape
        dtype = getattr(layer, "orig_dtype", None) or torch.bfloat16
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        scale = layer.weight_scale_inv.data

        out = torch.empty(n, k, dtype=dtype, device=weight.device)
        # Row-chunked (block-aligned) dequant to bound the FP32 transient at
        # ~1k rows instead of materialising the whole FP32 weight at once.
        rows_per_chunk = block_n * max(1, 1024 // block_n)
        for r0 in range(0, n, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, n)
            s = scale[r0 // block_n : (r1 + block_n - 1) // block_n]
            s = s.repeat_interleave(block_n, dim=0)[: r1 - r0].repeat_interleave(
                block_k, dim=1
            )[:, :k]
            # Explicit bit decode: direct .to(float32) on fp8 NPU tensors fails
            # on some CANN builds (aclnnInplaceCopy error 561103).
            out[r0:r1] = (_dequant_e4m3fn_to_float32(weight[r0:r1]) * s).to(dtype)

        layer.weight = Parameter(out, requires_grad=False)
        # Keep weight_scale_inv registered (tiny) so hot reload / any external
        # reader of the parameter still finds it.

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            # Pre-quantized (fp8 payload, per-group scale[, orig_dtype]) input
            # from a fused norm+quant kernel: dequantise back before the BF16
            # GEMM. Scale layout is row-major (M, cdiv(K, group)).
            qx, x_scale = x[0], x[1]
            orig_dtype = x[2] if len(x) > 2 else torch.bfloat16
            group = self.weight_block_size[1]
            x = (
                _dequant_e4m3fn_to_float32(qx)
                * x_scale.repeat_interleave(group, dim=1)[..., : qx.shape[-1]]
            ).to(orig_dtype)
        return torch.nn.functional.linear(x, layer.weight, bias)


class NPUBlockFP8RequantMXFP8LinearMethod(_NPULinearMethodBase):
    """NPU block-FP8 (128x128) dense linear — route B1: requantise to MXFP8.

    Alternative to ``NPUBlockFP8LinearMethod`` (route A, dequant-to-BF16) that
    keeps the weight at 1 byte: at load time, dequantise the checkpoint's
    block-FP8 weight (fp8 payload * 128x128 fp32 scale) to BF16, then requantise
    to MXFP8 (fp8 payload + 1x32 UE8M0 block scale via npu_dynamic_mx_quant).
    Inference then runs the exact production MXFP8 chain
    (NPUMXFP8LinearMethod.apply): dynamic MXFP8 activation quant +
    npu_quant_matmul(group_sizes=[1, 1, 32]).

    Rationale (probed on A5, see llm/probe_blockfp8_w8a8_dense.py):
    npu_quant_matmul rejects 128-group scales (aclnnQuantMatmulV5 error
    161002) while the gs=[1,1,32] MX convention runs with rel_err ~2e-3, so a
    native W8A8 block-FP8 GEMM does not exist; requantising to MXFP8 is the
    only native quantised path.

    Trade-off vs route A: weight memory back to ~1/2 (1 byte + tiny scale), at
    the cost of double weight quantisation (128x128 fp32 scale -> 1x32 e8m0)
    and e8m0 activation quant — needs a GSM8K regression.

    Opt-in via SGLANG_NPU_BLOCK_FP8_REQUANT_MXFP8=1; route A stays the default.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__(quant_config)
        self.weight_block_size = quant_config.weight_block_size
        self.is_checkpoint_fp8_serialized = quant_config.is_checkpoint_fp8_serialized

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        # Same checkpoint layout as route A (fp8 payload + 128x128 fp32 scale);
        # only process_weights_after_loading differs.
        Fp8LinearMethod.create_fp8_weight_(
            layer,
            block_quant=True,
            quant_config=self.quant_config,
            use_mxfp8=False,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            input_size=input_size,
            output_size=output_size,
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            params_dtype=params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype in (torch.float16, torch.bfloat16):
            # Unquantized checkpoint: quantise straight to MXFP8.
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            bf16 = (
                weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)
            )
        else:
            # Step 1: dequantise block-FP8 -> BF16 (same chunked logic as
            # route A; duplicated deliberately so route A stays untouched).
            block_n, block_k = self.weight_block_size
            n, k = weight.shape
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            scale = layer.weight_scale_inv.data

            bf16 = torch.empty(n, k, dtype=torch.bfloat16, device=weight.device)
            rows_per_chunk = block_n * max(1, 1024 // block_n)
            for r0 in range(0, n, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, n)
                s = scale[r0 // block_n : (r1 + block_n - 1) // block_n]
                s = s.repeat_interleave(block_n, dim=0)[: r1 - r0].repeat_interleave(
                    block_k, dim=1
                )[:, :k]
                # Explicit bit decode: direct .to(float32) on fp8 NPU tensors
                # fails on some CANN builds (aclnnInplaceCopy error 561103).
                bf16[r0:r1] = (_dequant_e4m3fn_to_float32(weight[r0:r1]) * s).to(
                    torch.bfloat16
                )

        # Step 2: requantise BF16 -> MXFP8 (1x32 e8m0 block scale).
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            bf16, dst_type=torch.float8_e4m3fn
        )

        # Layout mirrors NPUMXFP8LinearMethod: weight [in, out] and scale
        # [in//64, out, 2] as strided transpose views — DO NOT call
        # .contiguous() (see that class for the bandwidth rationale).
        layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
        if w_scale.dim() == 2:
            # Older torch_npu builds return [out, in//32]; reshape to 3D.
            n_s, k_s = w_scale.shape
            w_scale = w_scale.reshape(n_s, k_s // 2, 2)
        layer.weight_scale_inv = Parameter(w_scale.transpose(0, 1), requires_grad=False)

        # Cache FP32 bias once (same as NPUMXFP8LinearMethod).
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            # Pre-quantized (fp8 payload, per-group-128 scale[, orig_dtype])
            # input from a fused norm+quant kernel: dequantise back to BF16,
            # then requantise to MXFP8 below. Scale layout is row-major
            # (M, cdiv(K, group)).
            qx, x_scale = x[0], x[1]
            orig_dtype = x[2] if len(x) > 2 else torch.bfloat16
            group = self.weight_block_size[1]
            x = (
                _dequant_e4m3fn_to_float32(qx)
                * x_scale.repeat_interleave(group, dim=1)[..., : qx.shape[-1]]
            ).to(orig_dtype)

        # Inference chain: identical to NPUMXFP8LinearMethod.apply.
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
