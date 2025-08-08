# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from einops import rearrange
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
) 
import math
from flash_attn import flash_attn_varlen_func

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:

        # print(q.shape, k.shape, v.shape)
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # torch.cuda.synchronize()
        # start_event.record()
        
        if False:
            out = moba_attention(
                q=q,
                k=k,
                v=v
            )
        else:
            out = flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )

        # end_event.record()
        # torch.cuda.synchronize()

        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"############# Efficient time: {elapsed_time_ms:.3f} ms")
        return out
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out

def moba_attention(q,k,v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    moba_chunk_size = 1170
    moba_topk = 8

    seq_lens = [q_len]
    cu_seqlens_q = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))], dtype=torch.int32).to(q.device)

    seq_lens = [kv_len]
    cu_seqlens_k = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))], dtype=torch.int32).to(q.device)

    output = moba_attn_varlen(q = q.squeeze(0),
                        k = k.squeeze(0),
                        v = v.squeeze(0),
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        moba_chunk_size=moba_chunk_size,
                        max_seqlen_q = q_len,
                        max_seqlen_k = kv_len,
                        moba_topk=moba_topk
                        )
    output = output.unsqueeze(0)
    return output
    


class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen_q,
        self_attn_cu_seqlen_kv,
        moba_q,
        moba_k,
        moba_v,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        if k is not None:

            self_attn_out_sh, self_attn_lse_hs, _, _ = (
                _flash_attn_varlen_forward(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=self_attn_cu_seqlen_q,
                    cu_seqlens_k=self_attn_cu_seqlen_kv,
                    max_seqlen_q=self_attn_cu_seqlen_q[-1],
                    max_seqlen_k=self_attn_cu_seqlen_kv[-1],
                    softmax_scale=softmax_scale,
                    causal=False,
                    dropout_p=0.0,
                )
            )

            # moba attn
            moba_attn_out, _,_,_,_,moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
                q=moba_q,
                k=moba_k,
                v=moba_v,
                cu_seqlens_q=moba_cu_seqlen_q,
                cu_seqlens_k=moba_cu_seqlen_kv,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=moba_chunk_size,
                softmax_scale=softmax_scale,
                causal=False,
                dropout_p=0.0,
            )

            # convert lse shape hs -> sh ( follow the legacy mix attn logic )
            self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
            moba_attn_lse = moba_attn_lse_hs.t().contiguous()

            # output buffer [S, H, D], same shape as q
            output = torch.zeros(
                (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
            )

            # flatten vS & H for index ops
            output_2d = output.view(-1, q.shape[2])

            # calc mixed_lse
            # minus max lse to avoid exp explosion
            max_lse_1d = self_attn_lse_sh.view(-1)
            max_lse_1d = max_lse_1d.index_reduce(
                0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
            )
            self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
            moba_attn_lse = (
                moba_attn_lse.view(-1)
                .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
                .reshape_as(moba_attn_lse)
            )

            mixed_attn_se_sh = self_attn_lse_sh.exp()
            moba_attn_se = moba_attn_lse.exp()

            mixed_attn_se_sh.view(-1).index_add_(
                0, moba_q_sh_indices, moba_attn_se.view(-1)
            )
            mixed_attn_lse_sh = mixed_attn_se_sh.log()

            # add attn output
            factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
            self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
            output_2d += self_attn_out_sh.reshape_as(output_2d)

            # add moba output
            mixed_attn_lse = (
                mixed_attn_lse_sh.view(-1)
                .index_select(0, moba_q_sh_indices)
                .view_as(moba_attn_lse)
            )
            factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
            moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
            raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
            output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
            output = output.to(q.dtype)
            # add back max lse
            mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
            ctx.save_for_backward(
                output,
                mixed_attn_lse_sh,
                q,
                k,
                v,
                self_attn_cu_seqlen_q,
                moba_q,
                moba_k,
                moba_v,
                moba_cu_seqlen_q,
                moba_cu_seqlen_kv,
                moba_q_sh_indices,
            )
            return output

        else: #_, _, _, _,  moba_attn_out, _,_,_,_, moba_attn_lse_hs, _, _
            moba_attn_out, moba_attn_lse_hs, _, _  = _flash_attn_varlen_forward(
                q=moba_q,
                k=moba_k,
                v=moba_v,
                cu_seqlens_q=moba_cu_seqlen_q,
                cu_seqlens_k=moba_cu_seqlen_kv,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=moba_chunk_size,
                softmax_scale=softmax_scale,
                causal=False,
                dropout_p=0.0,
                # window_size=(-1,-1),
                # alibi_slopes=None,
                # return_softmax=False,
                # block_table=None,
            )

            moba_attn_lse = moba_attn_lse_hs.t().contiguous()

            max_lse_1d = torch.zeros_like(q).view(-1).to(moba_attn_lse.dtype)
            max_lse_1d = max_lse_1d.index_reduce(
                0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
            )

            moba_attn_lse = (
                moba_attn_lse.view(-1)
                .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
                .reshape_as(moba_attn_lse)
            )

            # flatten vS & H for index ops

            output_2d = torch.zeros_like(q).view(-1, q.shape[2]).to(moba_attn_lse.dtype)
            se =torch.zeros([q.shape[0], q.shape[1],1]).view(-1, 1).to(moba_attn_lse.dtype).to(moba_attn_lse.device)

            moba_attn_out = moba_attn_out * moba_attn_lse.exp().unsqueeze(-1)
            raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
            output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out.to(moba_attn_lse.device))
            se.index_add_(0, moba_q_sh_indices, moba_attn_lse.exp())
            output = output_2d.to(q.dtype)/(se.to(q.dtype))
            output = output.view(q.shape[0],q.shape[1],q.shape[2])
            ctx.save_for_backward(
                output,
                None,
                q,
                k,
                v,
                self_attn_cu_seqlen_q,
                moba_q,
                moba_k,
                moba_cu_seqlen_q,
                moba_cu_seqlen_kv,
                moba_q_sh_indices,
            )
            return output



    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv, _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None



def preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    moba_chunk_size: int,
    moba_topk: int,
):
    """ some basic variables """
    # qkv shape = [ S, H, D ]
    q_seqlen, num_head, head_dim = q.shape
    k_seqlen, num_head, head_dim = k.shape

    num_filtered_chunk = k_seqlen // moba_chunk_size
    num_block_in_q = math.ceil(max_seqlen_q / moba_chunk_size)

    need_moba_attn = False if moba_topk >= num_filtered_chunk else True
    need_moba_attn = False if num_block_in_q >= num_filtered_chunk else True


    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False
        )
    
    if max_seqlen_k % moba_chunk_size == 0:
        self_attn = False
        selected_num = num_block_in_q
        moba_topk = moba_topk
        self_q = q
        self_k = None
        self_v = None
        self_attn_cu_seqlen_q = None
        self_attn_cu_seqlen_kv = None
    else:
        raise

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        k
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
    )
    # q = q.type(torch.bfloat16)  # float logit on the fly for better gate logit perception
    # key_gate_weight = key_gate_weight.type(torch.bfloat16) 

    gate_mask = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    ).type_as(k)  # gate [ F_N_CHUNK, HEAD, SEQ ]
    gate_mask[-selected_num:,:,:] = float("inf")

    """ find moba q that needs moba attn """

    _, gate_top_k_idx = torch.topk(gate_mask, k=moba_topk, dim=0, largest=True, sorted=False)
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [ HS indices ] * N

    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.to(torch.int32).contiguous().sum(dim=-1).view(-1)

    # select all q that needs moba attn based on the moba_q_indices
    
    # q_re = q.permute(1, 0, 2).reshape(-1, q.shape[-1]).contiguous()  # [h, s, d] -> [h * s, d]
    # moba_q = torch.index_select(q_re, 0, moba_q_indices)

    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices)  # [ selected_S, D ]

    moba_q = moba_q.unsqueeze(1)
    
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % q_seqlen * num_head + moba_q_indices // q_seqlen


    """ prepare moba kv """
    
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)

    # [s, h, d] → [C_num, chunk_size, h,  d]
    moba_k = k.view(num_filtered_chunk, moba_chunk_size, num_head, head_dim).transpose(2, 1).contiguous()
    moba_v = v.view(num_filtered_chunk, moba_chunk_size, num_head, head_dim).transpose(2, 1).contiguous()

    moba_k = moba_k.flatten(start_dim=0, end_dim=1)
    moba_v = moba_v.flatten(start_dim=0, end_dim=1)
    if zero_expert_count > 0:
        moba_k = moba_k[
            valid_expert_mask
        ]  
        moba_v = moba_v[
            valid_expert_mask
        ]  

    moba_k = moba_k.flatten(start_dim=0, end_dim=1).unsqueeze(1)
    moba_v = moba_v.flatten(start_dim=0, end_dim=1).unsqueeze(1)
    
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    return (
        self_q,
        self_k,
        self_v,
        self_attn_cu_seqlen_q,
        self_attn_cu_seqlen_kv,
        moba_q,
        moba_k,
        moba_v,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
    )

pre_func = torch.compile(preprocess)

def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:

    output = pre_func(q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            moba_chunk_size,
            moba_topk)
    
    if not isinstance(output, tuple):
        return output

    self_q, self_k, self_v, self_attn_cu_seqlen_q, self_attn_cu_seqlen_kv, moba_q, moba_k, moba_v, moba_cu_seqlen_q, moba_cu_seqlen_kv, max_seqlen_q, moba_chunk_size, moba_q_sh_indices = output

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    output = MixedAttention.apply(
        self_q,
        self_k,
        self_v,
        self_attn_cu_seqlen_q,
        self_attn_cu_seqlen_kv,
        moba_q,
        moba_k,
        moba_v,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
    )    
    return output


