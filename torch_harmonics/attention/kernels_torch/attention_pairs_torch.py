# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

r"""
Neighbourhood attention gradients over an explicit pair list, differentiable to any order.

The backwards of the attention ops run compiled kernels or accumulate in place, and autograd
can differentiate neither. Under ``create_graph=True`` they call :func:`_attention_backward_pairs`
instead: the gradients of the ragged reference (see ``attention_ragged_torch``), vectorized over
every (output point, input point) pair of the neighbourhood and built from torch ops only.

The pairs come from the neighbourhood each op already saves for its backward, so this needs no
state of its own. Its cost is memory: the pairs times the channels, about the mean neighbourhood
size times the activations of the first-order backward, which is why it serves double backward
only and first order keeps the kernels.
"""

from typing import Optional, Tuple

import torch

from .._attention_utils import _reciprocal_or_zero


def _arc_pairs(segments: torch.Tensor, offsets: torch.Tensor, ring_base: torch.Tensor, ring_size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand an arc pattern into ``(row, column, ring)`` of every entry, rows as the pattern stores them.

    The device-side counterpart of :meth:`NeighborhoodArcsS2.to_csr`, without the chunking: the
    pairs are needed whole, on the device the arcs live on.
    """
    device = segments.device
    seg = segments.to(torch.int64)
    off = offsets.to(torch.int64)
    lengths = seg[:, 2]
    row_of_seg = torch.repeat_interleave(torch.arange(off.numel() - 1, device=device), off[1:] - off[:-1])
    seg_of_entry = torch.repeat_interleave(torch.arange(seg.shape[0], device=device), lengths)
    within = torch.arange(seg_of_entry.numel(), device=device) - (torch.cumsum(lengths, 0) - lengths)[seg_of_entry]
    ring = seg[seg_of_entry, 0]
    col = ring_base.to(device=device, dtype=torch.int64)[ring] + (seg[seg_of_entry, 1] + within) % ring_size.to(device=device, dtype=torch.int64)[ring]
    return row_of_seg[seg_of_entry], col, ring


def _unfold_longitude(rows: torch.Tensor, cols: torch.Tensor, points_per_row: int, shift: int, nlon_cols: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Undo the longitude fold of a product-grid pattern: row ``r`` holds the columns of the first
    point of ring ``r``, and point ``w`` of that ring takes them shifted by ``w * shift`` within
    their own ring of ``nlon_cols`` points.
    """
    w = torch.arange(points_per_row, device=rows.device)[:, None]
    base = cols - cols % nlon_cols
    cols = base + (cols % nlon_cols + w * shift) % nlon_cols
    rows = rows * points_per_row + w
    return rows.reshape(-1), cols.reshape(-1)


def _regular_pairs(rows: torch.Tensor, cols: torch.Tensor, nlon_in: int, nlon_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ``(output point, input point)`` pairs of a product-grid pattern stored folded, one row per ring.

    The direction follows the longitude ratio, as in the ops: a pattern whose input longitudes
    are a multiple of the output's is stored per output ring over input columns, an upsampling
    one per input ring over output columns.
    """
    if nlon_in % nlon_out == 0:
        return _unfold_longitude(rows, cols, nlon_out, nlon_in // nlon_out, nlon_in)
    inp, out = _unfold_longitude(rows, cols, nlon_in, nlon_out // nlon_in, nlon_out)
    return out, inp


def _attention_backward_pairs(
    kx: torch.Tensor,
    vx: torch.Tensor,
    qy: torch.Tensor,
    dy: torch.Tensor,
    pair_weights: torch.Tensor,
    out_idx: torch.Tensor,
    in_idx: torch.Tensor,
    needs_input_grad: Tuple[bool, bool, bool],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""
    Gradients of neighbourhood attention w.r.t. keys, values and queries, built from torch ops.

    With :math:`p_{oj} = \alpha_{oj} / A_o`, :math:`g_{oj} = \partial_y L|_o \cdot v_j` and
    :math:`I_o = \sum_j p_{oj} g_{oj}`, these are :math:`\partial L / \partial v_j = \sum_o p_{oj}
    \partial_y L|_o`, :math:`\partial L / \partial k_j = \sum_o p_{oj} (g_{oj} - I_o) q_o` and
    :math:`\partial L / \partial q_o = \sum_j p_{oj} (g_{oj} - I_o) k_j`, the formulas of the
    ragged reference, summed over the pairs at once rather than looped over output points.

    Parameters
    ----------
    kx, vx : torch.Tensor
        Keys ``(batch, channels, npoints_in)`` and values ``(batch, channels_v, npoints_in)``,
        channels first with the heads folded into the batch.
    qy, dy : torch.Tensor
        Queries ``(batch, channels, npoints_out)`` and the output gradient
        ``(batch, channels_v, npoints_out)``.
    pair_weights : torch.Tensor
        Quadrature weight of each pair's input point, ``(npairs,)``.
    out_idx, in_idx : torch.Tensor
        Output and input point of each pair, ``(npairs,)``.
    needs_input_grad : tuple of bool
        Whether the keys, values and queries need a gradient; the others are returned as ``None``.
    """
    batch, npoints_out = qy.shape[0], qy.shape[-1]
    w = pair_weights.to(qy.dtype)

    qdotk = torch.sum(qy[:, :, out_idx] * kx[:, :, in_idx], dim=1)
    # the row maximum only stabilizes the exponential and cancels in the ratio, so it carries no gradient
    qdotk_max = torch.full((batch, npoints_out), float("-inf"), dtype=qdotk.dtype, device=qdotk.device)
    qdotk_max = qdotk_max.scatter_reduce(1, out_idx.expand(batch, -1), qdotk.detach(), "amax")
    alpha = torch.exp(qdotk - qdotk_max[:, out_idx]) * w
    alpha_sum = torch.zeros_like(qdotk_max).index_add(1, out_idx, alpha)
    # an empty row's zero sum has a NaN derivative here, but no pair indexes that row, so it never propagates
    alpha_norm = alpha * _reciprocal_or_zero(alpha_sum)[:, out_idx]

    dkx = dvx = dqy = None
    if needs_input_grad[1]:
        dvx = torch.zeros_like(vx).index_add(2, in_idx, alpha_norm.unsqueeze(1) * dy[:, :, out_idx])
    if needs_input_grad[0] or needs_input_grad[2]:
        gdotv = torch.sum(dy[:, :, out_idx] * vx[:, :, in_idx], dim=1)
        integral = torch.zeros_like(alpha_sum).index_add(1, out_idx, alpha_norm * gdotv)
        dscore = alpha_norm * (gdotv - integral[:, out_idx])
        if needs_input_grad[0]:
            dkx = torch.zeros_like(kx).index_add(2, in_idx, dscore.unsqueeze(1) * qy[:, :, out_idx])
        if needs_input_grad[2]:
            dqy = torch.zeros_like(qy).index_add(2, out_idx, dscore.unsqueeze(1) * kx[:, :, in_idx])
    return dkx, dvx, dqy


def _nhwc_to_cf(t: torch.Tensor, nh: int) -> torch.Tensor:
    """``(batch, nlat, nlon, heads * channels)`` to ``(batch * heads, channels, nlat * nlon)``."""
    b, h, w, c = t.shape
    return t.reshape(b, h * w, nh, c // nh).permute(0, 2, 3, 1).reshape(b * nh, c // nh, h * w)


def _cf_to_nhwc(t: torch.Tensor, nh: int, nlat: int, nlon: int) -> torch.Tensor:
    """Inverse of :func:`_nhwc_to_cf`."""
    bh, c, _ = t.shape
    return t.reshape(bh // nh, nh, c, nlat, nlon).permute(0, 3, 4, 1, 2).reshape(bh // nh, nlat, nlon, nh * c)


def _regular_attention_backward_pairs(
    kw: torch.Tensor,
    vw: torch.Tensor,
    qw: torch.Tensor,
    grad_output: torch.Tensor,
    ring_weights: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    nh: int,
    nlon_in: int,
    nlat_out: int,
    nlon_out: int,
    needs_input_grad: Tuple[bool, bool, bool],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    :func:`_attention_backward_pairs` for the product-grid ops: channels-last ``(batch, nlat, nlon,
    heads * channels)`` in and out, the neighbourhood as the folded ``(rows, cols)`` they store,
    and the quadrature weight per input latitude.
    """
    nlat_in = kw.shape[1]
    out_idx, in_idx = _regular_pairs(rows, cols, nlon_in, nlon_out)
    compute_dtype = torch.promote_types(kw.dtype, torch.float32)
    kx, vx, qy, dy = (_nhwc_to_cf(t, nh).to(compute_dtype) for t in (kw, vw, qw, grad_output))
    dkx, dvx, dqy = _attention_backward_pairs(kx, vx, qy, dy, ring_weights[in_idx // nlon_in], out_idx, in_idx, needs_input_grad)
    dkw = None if dkx is None else _cf_to_nhwc(dkx, nh, nlat_in, nlon_in).to(kw.dtype)
    dvw = None if dvx is None else _cf_to_nhwc(dvx, nh, nlat_in, nlon_in).to(vw.dtype)
    dqw = None if dqy is None else _cf_to_nhwc(dqy, nh, nlat_out, nlon_out).to(qw.dtype)
    return dkw, dvw, dqw
