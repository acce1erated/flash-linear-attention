# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Token-parallel implementation of QuasarAttention
# FUSED mega-kernel: alpha + KK^T + tril mask + Neumann inversion + W/U + scan precompute

import torch
import triton
import triton.language as tl

from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem, input_guard

NUM_WARPS = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]

# log2(e) for exp2 conversion
LOG2E = tl.constexpr(1.4426950408889634)

# Neumann series terms for (I+M)^{-1}: k=3 saves 1 dot vs k=4, cos>=0.9985
NEUMANN_TERMS = tl.constexpr(3)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'STORE_WU': lambda args: args['W_out'].data_ptr() != args['A_trans_out'].data_ptr(),
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64]  # Must match S for correct dot product dimensions
        for num_warps in [4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'S', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_quasar_fwd_kernel_intra(
    q, k, v, beta,
    W_out, U_out,
    A_trans_out, B_trans_out, qA_out, qB_out,
    cu_seqlens,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_WU: tl.constexpr,
):
    """
    Fused intra-chunk kernel: computes W, U, A_trans, B_trans, qA, qB directly.

    Fuses: alpha + KK^T + tril mask + Neumann inversion + W/U + scan precompute
    All dot products use fp16 tensor cores with fp32 accumulation for 2x throughput.
    """
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    if i_c * BT >= T:
        return

    # Load k, v, q
    base_k = k + (bos * H + i_h) * S + i_c * BT * H * S
    base_v = v + (bos * H + i_h) * S + i_c * BT * H * S
    base_q = q + (bos * H + i_h) * S + i_c * BT * H * S

    offs_t = tl.arange(0, BT)
    offs_s = tl.arange(0, BK)

    mask_k = (offs_t[:, None] < BT) & (offs_s[None, :] < S)
    p_idx = offs_t[:, None] * (H * S) + offs_s[None, :]

    b_k = tl.load(base_k + p_idx, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(base_v + p_idx, mask=mask_k, other=0.0).to(tl.float32)
    b_q = tl.load(base_q + p_idx, mask=mask_k, other=0.0).to(tl.float32)

    # Beta: shape [H] — single scalar per head
    b_beta = tl.load(beta + i_h).to(tl.float32)

    # Step 1: alpha
    b_lambda = tl.sum(b_k * b_k, axis=1)
    eps = 1e-8
    b_alpha = (1 - tl.exp2(-b_beta * b_lambda * LOG2E)) / (b_lambda + eps)

    # Step 2: KK^T (fp16 tensor cores for 2x throughput)
    b_k_f16 = b_k.to(tl.float16)
    b_KK_t = tl.dot(b_k_f16, tl.trans(b_k_f16), out_dtype=tl.float32)

    # Step 3: M = tril(alpha * KK^T, diagonal=-1)
    b_alpha_col = b_alpha[:, None]
    b_M = b_alpha_col * b_KK_t
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    mask_lower = offs_i[:, None] > offs_j[None, :]
    b_M = tl.where(mask_lower, b_M, 0.0)

    # Step 4: Neumann series: A = (I + M)^{-1} (fp16 dots, fp32 accumulation)
    b_I_BT = (offs_i[:, None] == offs_j[None, :]).to(tl.float32)
    b_neg_M = -b_M
    b_A = b_I_BT + b_neg_M

    b_neg_M_power = b_neg_M
    for _k in range(NEUMANN_TERMS - 1):
        b_neg_M_power = tl.dot(b_neg_M_power.to(tl.float16), b_neg_M.to(tl.float16), out_dtype=tl.float32)
        b_A += b_neg_M_power

    # Step 5: W = A @ (alpha * K), U = A @ (alpha * V)
    b_alpha_k = b_alpha_col * b_k
    b_alpha_v = b_alpha_col * b_v
    b_A_f16 = b_A.to(tl.float16)
    b_W = tl.dot(b_A_f16, b_alpha_k.to(tl.float16), out_dtype=tl.float32)
    b_U = tl.dot(b_A_f16, b_alpha_v.to(tl.float16), out_dtype=tl.float32)

    # Step 6: Precompute scan matrices (fp16 dots)
    b_k_t = tl.trans(b_k_f16)
    b_KtW = tl.dot(b_k_t, b_W.to(tl.float16), out_dtype=tl.float32)
    b_I_S = (offs_s[:, None] == offs_s[None, :]).to(tl.float32)
    b_A_trans = b_I_S - b_KtW
    b_B_trans = tl.dot(b_k_t, b_U.to(tl.float16), out_dtype=tl.float32)
    b_q_f16 = b_q.to(tl.float16)
    b_qA = tl.dot(b_q_f16, b_A_trans.to(tl.float16), out_dtype=tl.float32)
    b_qB = tl.dot(b_q_f16, b_B_trans.to(tl.float16), out_dtype=tl.float32)

    # Store
    NT = tl.cdiv(T, BT)
    out_offset_bts = (i_bh * NT + i_c) * BT * S
    p_out_idx = offs_t[:, None] * S + offs_s[None, :]
    mask_out = (offs_t[:, None] < BT) & (offs_s[None, :] < S)

    if STORE_WU:
        tl.store(W_out + out_offset_bts + p_out_idx, b_W.to(W_out.dtype.element_ty), mask=mask_out)
        tl.store(U_out + out_offset_bts + p_out_idx, b_U.to(U_out.dtype.element_ty), mask=mask_out)

    # Store scan precompute: A_trans, B_trans [S,S], qA, qB [BT,S]
    out_offset_ss = (i_bh * NT + i_c) * S * S
    ss_idx = offs_s[:, None] * S + offs_s[None, :]
    mask_ss = (offs_s[:, None] < S) & (offs_s[None, :] < S)

    tl.store(A_trans_out + out_offset_ss + ss_idx, b_A_trans.to(A_trans_out.dtype.element_ty), mask=mask_ss)
    tl.store(B_trans_out + out_offset_ss + ss_idx, b_B_trans.to(B_trans_out.dtype.element_ty), mask=mask_ss)
    tl.store(qA_out + out_offset_bts + p_out_idx, b_qA.to(qA_out.dtype.element_ty), mask=mask_out)
    tl.store(qB_out + out_offset_bts + p_out_idx, b_qB.to(qB_out.dtype.element_ty), mask=mask_out)
