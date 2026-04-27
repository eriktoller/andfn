"""
CUDA implementation of build_head_matrix — CSR / Jagged-Array architecture.

Parallelisation strategy
------------------------
* blockIdx.x  = discharge element index  (j)
* threadIdx.x = integration-point index  (i)

Every thread computes the full omega contribution for ONE z-point across ALL
elements of its fracture.  Shared memory holds the per-thread omega values
and a parallel tree-reduction computes the mean.

CSR memory layout
-----------------
Three previously-padded 2-D arrays are replaced by flat 1-D arrays with
companion offset arrays.  This eliminates padding and dramatically cuts VRAM:

  z_int (integration points)
  ──────────────────────────
  Old: z0_re [N_DE, MAX_DISCHARGE_INT=400]   ~12.8 kB / element
  New: z0_re_flat [total_z_pts]              ~200 B / element  (discharge_int=25)
       de_z_offsets [N_DE] int64
       de_z_counts  [N_DE] int32

  frac_elements
  ─────────────
  Old: frac_elements [NFRAC, MAX_ELEMENTS_PER_FRAC=500]   2 kB / fracture
  New: frac_el_flat  [sum(frac_nelements)]                 no padding
       frac_el_offsets [NFRAC] int64

  el_coef
  ───────
  Old: el_coef_re/im [NEL, MAX_COEF=200]    1.6 kB / element
  New: el_coef_re/im_flat [NEL * coef_stride]  same total, but with a CSR
       el_coef_offsets [NEL] int64 handle.
       Future: per-element variable allocation once ncoef is stable.

VRAM savings for 300 k elements / 100 k fractures / discharge_int=25:
  z_int:        300 k × (400-25) × 32 B ≈ -3.6 GB saved
  frac_elements: 100 k × (500 - avg_ne) × 4 B ≈ -100-180 MB saved

Note on ``frac0`` vs ``frac1``
------------------------------
For an intersection discharge element:
  * frac0 call uses z0 (z-points on frac0) with positive coefs
  * frac1 call uses z1 (z-points on frac1) with negated result
The kernel passes frac_id_arr[fid0] for the frac0 comparison and
frac_id_arr[fid1] for the frac1 comparison, exactly matching the CPU.
"""

import math
import numpy as np
import numba as nb
from numba import cuda
import atexit

# ── compile-time constants ──────────────────────────────────────────────────

MAX_DISCHARGE_INT = 400  # upper bound on discharge_int (shared memory sizing)
MAX_COEF = 200  # upper bound on ncoef per element
MAX_ELEMENTS_PER_FRAC = 500  # upper bound on elements per fracture (shared memory)
WARP_SIZE = 32

# Shared omega array rounded up to warp multiple so padding threads always
# have a valid slot and all syncthreads calls are safe.
SHARED_SIZE = ((MAX_DISCHARGE_INT + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE  # 416


# ═══════════════════════════════════════════════════════════════════════════════
# Host-side JIT helpers for variable-length CSR coef packing
#
# These run on the CPU and replace the NumPy-based coef copy in
# _update_dynamic.  They avoid:
#   1. The ascontiguousarray copy of the full (nel, MAX_COEF) complex128 field
#   2. Copying inactive coef slots (beyond el_ncoef[el]) over PCIe
#
# For 105k elements with avg ncoef=15: packed size ≈ 105k×15×8×2 = 25 MB
# vs the old uniform-stride transfer of 105k×200×8×2 = 336 MB → ~13× less.
# ═══════════════════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def _compute_coef_offsets(ncoef_arr, offsets_out):
    """Serial exclusive-prefix-sum of ncoef_arr → offsets_out.
    Returns total number of active coef values."""
    pos = np.int64(0)
    for el in range(len(ncoef_arr)):
        offsets_out[el] = pos
        pos += ncoef_arr[el]
    return pos


@nb.njit(cache=True, parallel=True)
def _fill_coef_flat(coef_field, ncoef_arr, offsets, h_re_flat, h_im_flat):
    """Parallel pack of complex coef_field into separate real/imag flat CSR buffers.

    Parameters
    ----------
    coef_field : (nel, coef_stride) complex128 — structured-array 'coef' field
    ncoef_arr  : (nel,) int64 — active coef count per element
    offsets    : (nel,) int64 — exclusive prefix sum (from _compute_coef_offsets)
    h_re_flat  : (total_active,) float64 — pinned output, real parts
    h_im_flat  : (total_active,) float64 — pinned output, imag parts
    """
    for el in nb.prange(len(ncoef_arr)):
        off = offsets[el]
        nc = ncoef_arr[el]
        for c in range(nc):
            v = coef_field[el, c]
            h_re_flat[off + c] = v.real
            h_im_flat[off + c] = v.imag


@cuda.jit(device=True, inline=True)
def _csqrt(re, im):
    r = math.sqrt(re * re + im * im)
    real_part = math.sqrt(max(0.0, (r + re) * 0.5))
    imag_part = math.copysign(math.sqrt(max(0.0, (r - re) * 0.5)), im)
    return real_part, imag_part


@cuda.jit(device=True, inline=True)
def _clog(re, im):
    r = math.sqrt(re * re + im * im)
    if r < 1e-300:
        r = 1e-300
    return math.log(r), math.atan2(im, re)


@cuda.jit(device=True, inline=True)
def _cabs(re, im):
    return math.sqrt(re * re + im * im)


# ═══════════════════════════════════════════════════════════════════════════��═══
# Geometry device functions
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True, inline=True)
def _map_z_line_to_chi(z_re, z_im, ep_a_re, ep_a_im, ep_b_re, ep_b_im):
    """Joukowski map z → χ for a line element."""
    den_re = ep_b_re - ep_a_re
    den_im = ep_b_im - ep_a_im
    den_abs2 = den_re * den_re + den_im * den_im
    num_re = 2.0 * z_re - ep_a_re - ep_b_re
    num_im = 2.0 * z_im - ep_a_im - ep_b_im
    bZ_re = (num_re * den_re + num_im * den_im) / den_abs2
    bZ_im = (num_im * den_re - num_re * den_im) / den_abs2
    s1_re, s1_im = _csqrt(bZ_re - 1.0, bZ_im)
    s2_re, s2_im = _csqrt(bZ_re + 1.0, bZ_im)
    return (
        bZ_re + s1_re * s2_re - s1_im * s2_im,
        bZ_im + s1_re * s2_im + s1_im * s2_re,
    )


@cuda.jit(device=True, inline=True)
def _map_z_circle_to_chi(z_re, z_im, r, c_re, c_im):
    return (z_re - c_re) / r, (z_im - c_im) / r


# ═══════════════════════════════════════════════════════════════════════════════
# Math device functions (single chi-point)
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True)
def _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef):
    """Asymptotic expansion Σ aₙ/χⁿ (Horner form). Returns (re, im)."""
    tmp_re = 0.0
    tmp_im = 0.0
    chi_abs2 = chi_re * chi_re + chi_im * chi_im
    for n in range(ncoef - 1):
        tmp_re += coef_re[ncoef - n - 1]
        tmp_im += coef_im[ncoef - n - 1]
        new_re = (tmp_re * chi_re + tmp_im * chi_im) / chi_abs2
        new_im = (tmp_im * chi_re - tmp_re * chi_im) / chi_abs2
        tmp_re = new_re
        tmp_im = new_im
    return tmp_re + coef_re[0], tmp_im + coef_im[0]


@cuda.jit(device=True)
def _taylor_series_pt(chi_re, chi_im, coef_re, coef_im, ncoef):
    """Taylor series Σ aₙχⁿ (Horner form). Returns (re, im)."""
    tmp_re = 0.0
    tmp_im = 0.0
    for n in range(ncoef - 1):
        tmp_re += coef_re[ncoef - n - 1]
        tmp_im += coef_im[ncoef - n - 1]
        new_re = tmp_re * chi_re - tmp_im * chi_im
        new_im = tmp_re * chi_im + tmp_im * chi_re
        tmp_re = new_re
        tmp_im = new_im
    return tmp_re + coef_re[0], tmp_im + coef_im[0]


@cuda.jit(device=True)
def _well_chi_pt(chi_re, chi_im, q):
    log_re, log_im = _clog(chi_re, chi_im)
    f = q / (2.0 * math.pi)
    return f * log_re, f * log_im


# ═══════════════════════════════════════════════════════════════════════════════
# Per-element omega contributions (single z-point, shared-memory coef arrays)
# ═════════════════════════════════════════════════════════════════════════════���═


@cuda.jit(device=True)
def _omega_intersection_pt(
    z_re,
    z_im,
    frac_id,
    frac0,
    ep0a_re,
    ep0a_im,
    ep0b_re,
    ep0b_im,
    ep1a_re,
    ep1a_im,
    ep1b_re,
    ep1b_im,
    coef_re,
    coef_im,
    ncoef,
    q,
):
    if frac_id == frac0:
        chi_re, chi_im = _map_z_line_to_chi(
            z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im
        )
        ae_re, ae_im = _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)
        wc_re, wc_im = _well_chi_pt(chi_re, chi_im, q)
        return ae_re + wc_re, ae_im + wc_im
    else:
        chi_re, chi_im = _map_z_line_to_chi(
            z_re, z_im, ep1a_re, ep1a_im, ep1b_re, ep1b_im
        )
        ae_re, ae_im = _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)
        wc_re, wc_im = _well_chi_pt(chi_re, chi_im, q)
        return -(ae_re + wc_re), -(ae_im + wc_im)


@cuda.jit(device=True)
def _omega_bounding_circle_pt(z_re, z_im, radius, coef_re, coef_im, ncoef):
    chi_re, chi_im = _map_z_circle_to_chi(z_re, z_im, radius, 0.0, 0.0)
    return _taylor_series_pt(chi_re, chi_im, coef_re, coef_im, ncoef)


@cuda.jit(device=True)
def _omega_well_pt(z_re, z_im, radius, c_re, c_im, q):
    chi_re, chi_im = _map_z_circle_to_chi(z_re, z_im, radius, c_re, c_im)
    if _cabs(chi_re, chi_im) < 1.0 - 1e-12:
        return math.nan, math.nan
    return _well_chi_pt(chi_re, chi_im, q)


@cuda.jit(device=True)
def _omega_const_head_pt(
    z_re,
    z_im,
    ep0a_re,
    ep0a_im,
    ep0b_re,
    ep0b_im,
    coef_re,
    coef_im,
    ncoef,
    q,
):
    chi_re, chi_im = _map_z_line_to_chi(z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im)
    wc_re, wc_im = _well_chi_pt(chi_re, chi_im, q)
    ae_re, ae_im = _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)
    return wc_re + ae_re, wc_im + ae_im


@cuda.jit(device=True)
def _omega_imp_circle_pt(z_re, z_im, radius, c_re, c_im, coef_re, coef_im, ncoef):
    chi_re, chi_im = _map_z_circle_to_chi(z_re, z_im, radius, c_re, c_im)
    if _cabs(chi_re, chi_im) < 1.0 - 1e-10:
        return math.nan, math.nan
    return _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)


@cuda.jit(device=True)
def _omega_imp_line_pt(
    z_re,
    z_im,
    ep0a_re,
    ep0a_im,
    ep0b_re,
    ep0b_im,
    coef_re,
    coef_im,
    ncoef,
):
    chi_re, chi_im = _map_z_line_to_chi(z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im)
    return _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel tree reduction
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True, inline=True)
def _block_reduce_sum(sh_arr, tid, n):
    """In-place tree reduction of sh_arr[0:n]. Result in sh_arr[0].
    All threads (including tid >= n) must participate."""
    stride = 1
    while stride < n:
        stride <<= 1
    stride >>= 1
    while stride > 0:
        if tid < stride and tid + stride < n:
            sh_arr[tid] += sh_arr[tid + stride]
        cuda.syncthreads()
        stride >>= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Warp-per-DE kernel
#
#   Why the old "one block per DE" design hit a ceiling
#   ─────────────────────────────────────────────────────
#   The old kernel used one block per discharge element and loaded coefs into
#   shared memory cooperatively across all threads.  This requires
#   cuda.syncthreads() after EVERY element's coef load (and again after
#   compute).  With discharge_int=25 there is only one active warp per block,
#   so adding more warps (tpb=64/128) gave zero improvement — all warps wait
#   at every barrier.
#
#   New design
#   ──────────
#   blockIdx.x  → group of DES_PER_BLOCK discharge elements
#   warpId      → which DE within the block  (= threadIdx.x // 32)
#   laneId      → which z-point              (= threadIdx.x  % 32)
#
#   Each warp is completely independent.  There are NO cross-warp
#   cuda.syncthreads() calls — coefs are loaded by the warp's own lanes
#   with warp-shuffle reduction replacing the shared-memory reduction.
#
#   Each warp reads coefs directly from global memory (CSR flat array) into
#   registers.  The RTX A1000 L2 is 2 MB — for sorted DEs, consecutive
#   warps process DEs from the same fracture, so coef data stays L2-hot and
#   each warp's DRAM traffic is ~ncoef×2×8 B ≈ 160 B at ncoef=10.
#
#   Shared memory layout (per block of DES_PER_BLOCK warps)
#   ─────────────────────────────────────────────────────────
#   sh_omega0[DES_PER_BLOCK * WARP_SIZE]  — per-lane omega accumulator frac0
#   sh_omega1[DES_PER_BLOCK * WARP_SIZE]  — per-lane omega accumulator frac1
#   sh_el0   [DES_PER_BLOCK * MAX_ELEMENTS_PER_FRAC]  — frac0 element indices
#   sh_el1   [DES_PER_BLOCK * MAX_ELEMENTS_PER_FRAC]  — frac1 element indices
#
#   Frac element indices are loaded cooperatively across the block's lanes
#   (one syncthreads at the top, once per frac pass), so the DRAM cost of
#   reading frac_el_flat is paid once per block, not once per warp.
#
#   Occupancy improvement
#   ──────────────────────
#   DES_PER_BLOCK=4 → tpb=128.  RTX A1000: 2048 threads/SM ÷ 128 = 16 blocks/SM.
#   For n_de=35k and 20 SMs: 35k/20 = 1759 blocks/SM queue depth.
#   Active warps per SM = 16 blocks × 4 warps = 64 warps = 100% occupancy
#   at n_de ≥ 20 × 16 × 4 = 1280 (vs 1760 required before).
# ═══════════════════════════════════════════════════════════════════════════════

# Number of discharge elements processed per block.
# Must divide WARP_SIZE evenly; tpb = DES_PER_BLOCK * WARP_SIZE.
# 4 → tpb=128, 100% occupancy at n_de ≥ 1 280 on RTX A1000.
DES_PER_BLOCK = 4
_TPB_WARP_KERNEL = DES_PER_BLOCK * WARP_SIZE  # 128  (default)

# Maximum DES_PER_BLOCK ever used (for tpb sweep up to 256 = 8 warps).
# Shared memory is allocated at compile-time with these upper bounds.
_MAX_DES_PER_BLOCK = 8  # supports tpb up to 8*32=256

# Shared-memory sizes per block (compile-time upper bounds)
_SH_OMEGA_SIZE = _MAX_DES_PER_BLOCK * WARP_SIZE  # 256 floats × 8 B = 2 KB
_SH_EL_SIZE = _MAX_DES_PER_BLOCK * MAX_ELEMENTS_PER_FRAC  # 8 × 500 × 4 B = 16 KB


@cuda.jit(device=True, inline=True)
def _warp_reduce_sum(val, lane):
    """Warp-level reduction using shuffle-down.  Returns sum in lane 0.
    No shared memory or syncthreads needed — pure register communication.

    Uses cuda.activemask() so the mask exactly matches the set of threads
    converged at this point.  All 32 lanes of a warp always reach this
    function together (no intra-warp divergence at the call sites), so
    the mask will always be 0xFFFFFFFF in practice — but using activemask
    is the defensive/spec-compliant approach.
    """
    mask = cuda.activemask()
    val += cuda.shfl_down_sync(mask, val, 16)
    val += cuda.shfl_down_sync(mask, val, 8)
    val += cuda.shfl_down_sync(mask, val, 4)
    val += cuda.shfl_down_sync(mask, val, 2)
    val += cuda.shfl_down_sync(mask, val, 1)
    return val


@cuda.jit(cache=True)
def _build_head_matrix_kernel(
    # ── discharge elements ─────────────────────────────────────────────────
    de_frac0,  # [n_de] int64
    de_frac1,  # [n_de] int64
    de_type,  # [n_de] int64
    de_phi,  # [n_de] float64
    de_z_offsets,  # [n_de] int64 — start into flat z arrays
    de_z_counts,  # [n_de] int32 — actual z-point count per DE
    # ── fractures ─────────────────────────────────────────────────────────
    frac_constant,  # [nfrac] float64
    frac_t,  # [nfrac] float64
    frac_nelements,  # [nfrac] int64
    frac_el_offsets,  # [nfrac] int64 — start into frac_el_flat
    frac_el_flat,  # [total_frac_el] int32 — CSR element index list
    frac_id_arr,  # [nfrac] int64 — _id of each fracture
    # ── elements ──────────────────────────────────────────────────────────
    el_type,
    el_frac0,
    el_ep0a_re,
    el_ep0a_im,
    el_ep0b_re,
    el_ep0b_im,
    el_ep1a_re,
    el_ep1a_im,
    el_ep1b_re,
    el_ep1b_im,
    el_coef_re_flat,  # [total_coef] float64 — CSR flat real coefs
    el_coef_im_flat,  # [total_coef] float64 — CSR flat imag coefs
    el_coef_offsets,  # [nel] int64
    el_ncoef,  # [nel] int64
    el_q,  # [nel] float64
    el_radius,  # [nel] float64
    el_center_re,
    el_center_im,
    # ── integration points (flat CSR) ─────────────────────────────────────
    z0_re_flat,
    z0_im_flat,
    z1_re_flat,
    z1_im_flat,
    # ── output ────────────────────────────────────────────────────────────
    de_out_idx,  # [n_de] int64 — inverse sort permutation
    head_matrix,  # [n_de] float64
):
    """Warp-per-DE kernel.

    Each warp (32 lanes) handles one discharge element independently.
    No cross-warp synchronisation — coef reduction uses warp shuffle.
    Frac element indices are loaded cooperatively across the whole block
    once per fracture pass (one syncthreads per pass, not per element).
    """
    tid = cuda.threadIdx.x
    blk = cuda.blockDim.x
    # Derive des_per_blk from actual block size — works for any tpb.
    # DES_PER_BLOCK is only the *default*; _threads_per_block can override it.
    des_per_blk_rt = blk // WARP_SIZE
    warpId = tid // WARP_SIZE  # which DE within this block
    lane = tid % WARP_SIZE  # which z-point within the DE

    # Global discharge element index
    j = cuda.blockIdx.x * des_per_blk_rt + warpId
    n_de_total = de_frac0.shape[0]

    # ── shared memory ─────────────────────────────────────────────────────
    # Layout: [warpId * WARP_SIZE + lane]
    sh_omega0 = cuda.shared.array(_SH_OMEGA_SIZE, nb.float64)
    sh_omega1 = cuda.shared.array(_SH_OMEGA_SIZE, nb.float64)
    # sh_el0/sh_el1: [warpId * MAX_ELEMENTS_PER_FRAC + k]
    sh_el0 = cuda.shared.array(_SH_EL_SIZE, nb.int32)
    sh_el1 = cuda.shared.array(_SH_EL_SIZE, nb.int32)

    # Initialise omega slots (guards padding warps at end of grid)
    sh_omega0[tid] = 0.0
    sh_omega1[tid] = 0.0

    if j >= n_de_total:
        return

    # ── per-DE metadata ───────────────────────────────────────────────────
    z_count = de_z_counts[j]  # actual number of integration points (= discharge_int)
    z_off = de_z_offsets[j]
    fid0 = de_frac0[j]
    const0 = frac_constant[fid0]
    t0 = frac_t[fid0]
    ne0_raw = frac_nelements[fid0]
    # Clamp ne0 to the shared-memory bound.  If a fracture genuinely has
    # more than MAX_ELEMENTS_PER_FRAC elements, the extra ones are skipped
    # (rather than writing past the end of sh_el0 and corrupting the GPU).
    ne0 = ne0_raw if ne0_raw <= MAX_ELEMENTS_PER_FRAC else MAX_ELEMENTS_PER_FRAC
    fe0_off = frac_el_offsets[fid0]
    fid0_id = frac_id_arr[fid0]
    de_type_j = de_type[j]

    # frac1 metadata (only used for intersections, but must be defined)
    t1 = 1.0
    const1 = 0.0
    ne1 = nb.int64(0)
    fe1_off = nb.int64(0)
    fid1_id = nb.int64(0)
    if de_type_j == 0:
        fid1 = de_frac1[j]
        const1 = frac_constant[fid1]
        t1 = frac_t[fid1]
        ne1_raw = frac_nelements[fid1]
        ne1 = ne1_raw if ne1_raw <= MAX_ELEMENTS_PER_FRAC else MAX_ELEMENTS_PER_FRAC
        fe1_off = frac_el_offsets[fid1]
        fid1_id = frac_id_arr[fid1]

    # ── cooperative frac0 element-index load ──────────────────────────────
    # Each warp fills its own section sh_el0[warpId*MAX_ELEMENTS_PER_FRAC :].
    # Within a single warp lanes execute in lockstep on all current NVIDIA
    # hardware, so the warp-strided store is immediately visible to all lanes
    # in the subsequent read loop.  A cuda.syncwarp() guard is added for
    # spec-correctness on future independent-thread-scheduling hardware.
    el0_base = warpId * MAX_ELEMENTS_PER_FRAC
    for k in range(lane, ne0, WARP_SIZE):
        sh_el0[el0_base + k] = frac_el_flat[fe0_off + k]
    cuda.syncwarp(cuda.activemask())

    # ── frac0: multi-round z-point loop ───────────────────────────────────
    # Each round processes WARP_SIZE z-points (lanes 0..31).  This supports
    # discharge_int > WARP_SIZE (e.g. 50 → 2 rounds: z[0..31], z[32..49]).
    # The z-point load is hoisted outside the element loop per round so each
    # z-coordinate is read from global memory only once per round, not once
    # per element.
    acc0 = 0.0
    num_z_rounds = (z_count + WARP_SIZE - 1) // WARP_SIZE
    for r in range(num_z_rounds):
        lz = r * WARP_SIZE + lane  # global z-index for this lane this round
        lane_active = lz < z_count
        if lane_active:
            zr = z0_re_flat[z_off + lz]
            zi = z0_im_flat[z_off + lz]
        else:
            zr = 0.0
            zi = 0.0

        for k in range(ne0):
            el = sh_el0[el0_base + k]
            etype = el_type[el]
            ncoef = el_ncoef[el]
            c_off = el_coef_offsets[el]
            q_el = el_q[el]
            r_el = el_radius[el]
            c_re = el_center_re[el]
            c_im = el_center_im[el]
            f0_el = el_frac0[el]
            a0_re = el_ep0a_re[el]
            a0_im = el_ep0a_im[el]
            b0_re = el_ep0b_re[el]
            b0_im = el_ep0b_im[el]
            a1_re = el_ep1a_re[el]
            a1_im = el_ep1a_im[el]
            b1_re = el_ep1b_re[el]
            b1_im = el_ep1b_im[el]

            if lane_active:
                if etype == 0:
                    dr, _ = _omega_intersection_pt(
                        zr,
                        zi,
                        fid0_id,
                        f0_el,
                        a0_re,
                        a0_im,
                        b0_re,
                        b0_im,
                        a1_re,
                        a1_im,
                        b1_re,
                        b1_im,
                        el_coef_re_flat[c_off : c_off + ncoef],
                        el_coef_im_flat[c_off : c_off + ncoef],
                        ncoef,
                        q_el,
                    )
                elif etype == 1:
                    dr, _ = _omega_bounding_circle_pt(
                        zr,
                        zi,
                        r_el,
                        el_coef_re_flat[c_off : c_off + ncoef],
                        el_coef_im_flat[c_off : c_off + ncoef],
                        ncoef,
                    )
                elif etype == 2:
                    dr, _ = _omega_well_pt(zr, zi, r_el, c_re, c_im, q_el)
                elif etype == 3:
                    dr, _ = _omega_const_head_pt(
                        zr,
                        zi,
                        a0_re,
                        a0_im,
                        b0_re,
                        b0_im,
                        el_coef_re_flat[c_off : c_off + ncoef],
                        el_coef_im_flat[c_off : c_off + ncoef],
                        ncoef,
                        q_el,
                    )
                elif etype == 4:
                    dr, _ = _omega_imp_circle_pt(
                        zr,
                        zi,
                        r_el,
                        c_re,
                        c_im,
                        el_coef_re_flat[c_off : c_off + ncoef],
                        el_coef_im_flat[c_off : c_off + ncoef],
                        ncoef,
                    )
                elif etype == 5:
                    dr, _ = _omega_imp_line_pt(
                        zr,
                        zi,
                        a0_re,
                        a0_im,
                        b0_re,
                        b0_im,
                        el_coef_re_flat[c_off : c_off + ncoef],
                        el_coef_im_flat[c_off : c_off + ncoef],
                        ncoef,
                    )
                else:
                    dr = 0.0
                acc0 += dr

    # ── warp-shuffle reduction (no syncthreads) ───────────────────────────
    sum0 = _warp_reduce_sum(acc0, lane)

    # ── frac1 pass (intersections only) ───────────────────────────────────
    sum1 = 0.0
    if de_type_j == 0:
        el1_base = warpId * MAX_ELEMENTS_PER_FRAC
        for k in range(lane, ne1, WARP_SIZE):
            sh_el1[el1_base + k] = frac_el_flat[fe1_off + k]
        cuda.syncwarp(cuda.activemask())

        acc1 = 0.0
        for r in range(num_z_rounds):
            lz = r * WARP_SIZE + lane
            lane_active1 = lz < z_count
            if lane_active1:
                zr = z1_re_flat[z_off + lz]
                zi = z1_im_flat[z_off + lz]
            else:
                zr = 0.0
                zi = 0.0

            for k in range(ne1):
                el = sh_el1[el1_base + k]
                etype = el_type[el]
                ncoef = el_ncoef[el]
                c_off = el_coef_offsets[el]
                q_el = el_q[el]
                r_el = el_radius[el]
                c_re = el_center_re[el]
                c_im = el_center_im[el]
                f0_el = el_frac0[el]
                a0_re = el_ep0a_re[el]
                a0_im = el_ep0a_im[el]
                b0_re = el_ep0b_re[el]
                b0_im = el_ep0b_im[el]
                a1_re = el_ep1a_re[el]
                a1_im = el_ep1a_im[el]
                b1_re = el_ep1b_re[el]
                b1_im = el_ep1b_im[el]

                if lane_active1:
                    if etype == 0:
                        dr, _ = _omega_intersection_pt(
                            zr,
                            zi,
                            fid1_id,
                            f0_el,
                            a0_re,
                            a0_im,
                            b0_re,
                            b0_im,
                            a1_re,
                            a1_im,
                            b1_re,
                            b1_im,
                            el_coef_re_flat[c_off : c_off + ncoef],
                            el_coef_im_flat[c_off : c_off + ncoef],
                            ncoef,
                            q_el,
                        )
                    elif etype == 1:
                        dr, _ = _omega_bounding_circle_pt(
                            zr,
                            zi,
                            r_el,
                            el_coef_re_flat[c_off : c_off + ncoef],
                            el_coef_im_flat[c_off : c_off + ncoef],
                            ncoef,
                        )
                    elif etype == 2:
                        dr, _ = _omega_well_pt(zr, zi, r_el, c_re, c_im, q_el)
                    elif etype == 3:
                        dr, _ = _omega_const_head_pt(
                            zr,
                            zi,
                            a0_re,
                            a0_im,
                            b0_re,
                            b0_im,
                            el_coef_re_flat[c_off : c_off + ncoef],
                            el_coef_im_flat[c_off : c_off + ncoef],
                            ncoef,
                            q_el,
                        )
                    elif etype == 4:
                        dr, _ = _omega_imp_circle_pt(
                            zr,
                            zi,
                            r_el,
                            c_re,
                            c_im,
                            el_coef_re_flat[c_off : c_off + ncoef],
                            el_coef_im_flat[c_off : c_off + ncoef],
                            ncoef,
                        )
                    elif etype == 5:
                        dr, _ = _omega_imp_line_pt(
                            zr,
                            zi,
                            a0_re,
                            a0_im,
                            b0_re,
                            b0_im,
                            el_coef_re_flat[c_off : c_off + ncoef],
                            el_coef_im_flat[c_off : c_off + ncoef],
                            ncoef,
                        )
                    else:
                        dr = 0.0
                    acc1 += dr

        sum1 = _warp_reduce_sum(acc1, lane)

    # ── lane 0 of each warp writes the result ─────────────────────────────
    if lane == 0:
        omega0_mean = (const0 + sum0) / z_count
        out_row = de_out_idx[j]
        if de_type_j == 0:
            omega1_mean = (const1 + sum1) / z_count
            head_matrix[out_row] = omega1_mean / t1 - omega0_mean / t0
        elif de_type_j == 2 or de_type_j == 3:
            head_matrix[out_row] = de_phi[j] - omega0_mean


# ═══════════════════════════════════════════════════════════════════════════════
# Device array cache
# ═══════════════════════════════════════════════════════════════════════════════


class _DeviceCache:
    """Holds static device arrays and pre-allocated dynamic buffers."""

    def __init__(self):
        self.d = {}  # static device arrays
        self.d_dynamic = {}  # pre-allocated dynamic device buffers
        # Pinned host buffers for variable-length CSR coef packing.
        # Sized at worst-case (nel * MAX_COEF) so they never need reallocation.
        self.h_coef_re = None  # (nel * MAX_COEF,) float64 pinned flat
        self.h_coef_im = None
        self.h_offsets = None  # (nel,) int64 pinned — coef CSR offsets
        self.d_head_matrix = None
        self.initialised = False
        self.n_de = 0
        self.nel = 0
        self.coef_stride = MAX_COEF  # actual field width in the element dtype

    def invalidate(self):
        """Force re-upload of static arrays on the next call."""
        self.initialised = False
        self.d.clear()
        self.d_dynamic.clear()
        self.h_coef_re = None
        self.h_coef_im = None
        self.h_offsets = None
        self.d_head_matrix = None
        self.n_de = 0
        self.nel = 0


_device_cache = _DeviceCache()


def _cuda_atexit():
    """Explicitly release all CUDA resources before the interpreter exits.

    Python's garbage collector runs weakref finalizers (which call cuMemFree /
    cuMemFreeHost) AFTER the CUDA context has already been torn down, producing
    CudaAPIError [700] UNKNOWN_CUDA_ERROR on every cached allocation.

    By nulling all references here and calling cuda.close(), we ensure the
    driver frees the memory while the context is still valid, so the subsequent
    finalizer calls are no-ops on already-freed handles.
    """
    try:
        # Drop all references so the device arrays / pinned buffers are freed
        # while the CUDA context is still alive.
        _device_cache.d.clear()
        _device_cache.d_dynamic.clear()
        _device_cache.d_head_matrix = None
        _device_cache.h_coef_re = None
        _device_cache.h_coef_im = None
        _device_cache.h_offsets = None
        _device_cache.initialised = False
    except Exception:
        pass
    try:
        cuda.close()
    except Exception:
        pass


atexit.register(_cuda_atexit)


def _upload_static(
    cache,
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    z_int,
    discharge_int,
):
    """Build CSR arrays and upload all static geometry to the GPU.

    CSR arrays built here
    ---------------------
    z-points (uniform discharge_int per DE):
        de_z_offsets[j] = j * discharge_int   (stride-1 scan)
        de_z_counts[j]  = discharge_int
        z0_re_flat       = z0.ravel()  (n_de × discharge_int, C-order)

    frac_elements (variable per fracture):
        frac_el_offsets  = exclusive prefix sum of frac_nelements
        frac_el_flat     = rows of frac_elements concatenated without padding

    el_coef (uniform stride per element for growing-ncoef safety):
        el_coef_offsets[el] = el * coef_stride
        flat buffers allocated but filled in _update_dynamic each iteration
    """
    # ── sort discharge elements by frac0 for L2-cache friendliness ────────
    frac0_arr = np.ascontiguousarray(discharge_elements["frac0"]).astype(np.int64)
    sort_order = np.argsort(frac0_arr, kind="stable")
    de = discharge_elements[sort_order]
    z_int_s = z_int[sort_order]

    n_de = de.shape[0]
    # NOTE: discharge_int is passed explicitly — do NOT derive it from
    # z_int["z0"].shape[1], which equals MAX_NCOEF*2 (the full allocated
    # slot width), not the actual number of valid integration points.

    # ── static DE arrays ──────────────────────────────────────────────────
    cache.d["de_frac0"] = cuda.to_device(
        np.ascontiguousarray(de["frac0"]).astype(np.int64)
    )
    cache.d["de_frac1"] = cuda.to_device(
        np.ascontiguousarray(de["frac1"]).astype(np.int64)
    )
    cache.d["de_type"] = cuda.to_device(
        np.ascontiguousarray(de["_type"]).astype(np.int64)
    )
    cache.d["de_phi"] = cuda.to_device(
        np.ascontiguousarray(de["phi"]).astype(np.float64)
    )
    cache.d["de_out_idx"] = cuda.to_device(sort_order.astype(np.int64))

    # ── CSR z-points ──────────────────────────────────────────────────────
    de_z_offsets = np.arange(n_de, dtype=np.int64) * discharge_int
    de_z_counts = np.full(n_de, discharge_int, dtype=np.int32)
    cache.d["de_z_offsets"] = cuda.to_device(de_z_offsets)
    cache.d["de_z_counts"] = cuda.to_device(de_z_counts)

    z0_all = np.ascontiguousarray(z_int_s["z0"])[
        :, :discharge_int
    ]  # (n_de, discharge_int) complex128
    z1_all = np.ascontiguousarray(z_int_s["z1"])[:, :discharge_int]
    cache.d["z0_re_flat"] = cuda.to_device(np.ascontiguousarray(z0_all.real.ravel()))
    cache.d["z0_im_flat"] = cuda.to_device(np.ascontiguousarray(z0_all.imag.ravel()))
    cache.d["z1_re_flat"] = cuda.to_device(np.ascontiguousarray(z1_all.real.ravel()))
    cache.d["z1_im_flat"] = cuda.to_device(np.ascontiguousarray(z1_all.imag.ravel()))

    # ── CSR frac_elements ─────────────────────────────────────────────────
    frac_nelements = np.ascontiguousarray(fractures_struc_array["nelements"]).astype(
        np.int64
    )
    frac_el_offsets = np.zeros(len(frac_nelements), dtype=np.int64)
    if len(frac_nelements) > 1:
        frac_el_offsets[1:] = np.cumsum(frac_nelements[:-1])

    frac_elements_2d = np.ascontiguousarray(fractures_struc_array["elements"])
    rows = []
    for fi in range(len(frac_nelements)):
        ne = int(frac_nelements[fi])
        if ne > 0:
            rows.append(frac_elements_2d[fi, :ne].astype(np.int32))
    frac_el_flat = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)

    cache.d["frac_t"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["t"]).astype(np.float64)
    )
    cache.d["frac_nelements"] = cuda.to_device(frac_nelements)
    cache.d["frac_el_offsets"] = cuda.to_device(frac_el_offsets)
    cache.d["frac_el_flat"] = cuda.to_device(frac_el_flat)
    cache.d["frac_id_arr"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["_id"]).astype(np.int64)
    )

    # ── element static geometry ───────────────────────────────────────────
    nel = element_struc_array.shape[0]
    cache.nel = nel

    cache.d["el_type"] = cuda.to_device(
        np.ascontiguousarray(element_struc_array["_type"]).astype(np.int64)
    )
    cache.d["el_frac0"] = cuda.to_device(
        np.ascontiguousarray(element_struc_array["frac0"]).astype(np.int64)
    )
    cache.d["el_frac1"] = cuda.to_device(
        np.ascontiguousarray(element_struc_array["frac1"]).astype(np.int64)
    )
    cache.d["el_radius"] = cuda.to_device(
        np.ascontiguousarray(element_struc_array["radius"]).astype(np.float64)
    )

    ep0 = np.ascontiguousarray(element_struc_array["endpoints0"])
    ep1 = np.ascontiguousarray(element_struc_array["endpoints1"])
    cache.d["el_ep0a_re"] = cuda.to_device(np.ascontiguousarray(ep0[:, 0].real))
    cache.d["el_ep0a_im"] = cuda.to_device(np.ascontiguousarray(ep0[:, 0].imag))
    cache.d["el_ep0b_re"] = cuda.to_device(np.ascontiguousarray(ep0[:, 1].real))
    cache.d["el_ep0b_im"] = cuda.to_device(np.ascontiguousarray(ep0[:, 1].imag))
    cache.d["el_ep1a_re"] = cuda.to_device(np.ascontiguousarray(ep1[:, 0].real))
    cache.d["el_ep1a_im"] = cuda.to_device(np.ascontiguousarray(ep1[:, 0].imag))
    cache.d["el_ep1b_re"] = cuda.to_device(np.ascontiguousarray(ep1[:, 1].real))
    cache.d["el_ep1b_im"] = cuda.to_device(np.ascontiguousarray(ep1[:, 1].imag))

    center_all = np.ascontiguousarray(element_struc_array["center"]).astype(
        np.complex128
    )
    cache.d["el_center_re"] = cuda.to_device(np.ascontiguousarray(center_all.real))
    cache.d["el_center_im"] = cuda.to_device(np.ascontiguousarray(center_all.imag))

    # ── CSR coef layout — uniform stride (safe with growing ncoef) ────────
    # el_coef_offsets becomes a *dynamic* array because it is recomputed each
    # iteration from the current ncoef values (variable-length CSR packing).
    _coef_shape = element_struc_array.dtype["coef"].shape
    coef_stride = int(_coef_shape[0]) if _coef_shape else MAX_COEF
    cache.coef_stride = coef_stride

    # ── pre-allocate dynamic device buffers ───────────────────────────────
    nf = fractures_struc_array.shape[0]
    worst_case_coef = nel * coef_stride  # maximum possible total active coefs

    cache.d_dynamic["frac_constant"] = cuda.device_array(nf, dtype=np.float64)
    cache.d_dynamic["el_q"] = cuda.device_array(nel, dtype=np.float64)
    cache.d_dynamic["el_ncoef"] = cuda.device_array(nel, dtype=np.int64)
    # el_coef_offsets is dynamic: recomputed each iteration from ncoef
    cache.d_dynamic["el_coef_offsets"] = cuda.device_array(nel, dtype=np.int64)
    # Coef flat arrays pre-allocated at worst case; only the first
    # total_active entries are written/read per iteration.
    cache.d_dynamic["el_coef_re_flat"] = cuda.device_array(
        worst_case_coef, dtype=np.float64
    )
    cache.d_dynamic["el_coef_im_flat"] = cuda.device_array(
        worst_case_coef, dtype=np.float64
    )

    # Pinned host buffers — no malloc per iteration.
    # h_coef_re/im: worst-case flat coef buffer (nel × coef_stride floats)
    # h_offsets:    per-element coef start offsets (nel int64)
    cache.h_coef_re = cuda.pinned_array(worst_case_coef, dtype=np.float64)
    cache.h_coef_im = cuda.pinned_array(worst_case_coef, dtype=np.float64)
    cache.h_offsets = cuda.pinned_array(nel, dtype=np.int64)

    # ── head matrix output buffer ─────────────────────────────────────────
    cache.d_head_matrix = cuda.device_array(n_de, dtype=np.float64)
    cache.n_de = n_de
    cache.initialised = True


def _update_dynamic(cache, fractures_struc_array, element_struc_array):
    """Copy only the iteration-varying arrays to the GPU.

    Key optimisation: variable-length CSR coef packing
    ---------------------------------------------------
    Instead of copying the full (nel × coef_stride) coef buffers, we use
    a parallel JIT packer (_fill_coef_flat) that:
      1. Reads directly from the structured-array "coef" field — no intermediate
         ascontiguousarray copy of the large complex128 array.
      2. Packs only el_ncoef[el] coefs per element into a flat CSR buffer.
      3. Copies only total_active = sum(ncoef) entries over PCIe.

    For 105k elements with avg ncoef ≈ 15 (typical early iterations):
      packed size ≈ 105k × 15 × 8 × 2 = 25 MB
      vs old uniform transfer  = 105k × 200 × 8 × 2 = 336 MB  → ~13× less

    The dynamic coef-offset array (el_coef_offsets) is also recomputed and
    uploaded each iteration, replacing the static uniform-stride version.
    """
    cache.d_dynamic["frac_constant"].copy_to_device(
        np.ascontiguousarray(fractures_struc_array["constant"]).astype(np.float64)
    )
    cache.d_dynamic["el_q"].copy_to_device(
        np.ascontiguousarray(element_struc_array["q"]).astype(np.float64)
    )

    ncoef_arr = np.ascontiguousarray(element_struc_array["ncoef"]).astype(np.int64)
    cache.d_dynamic["el_ncoef"].copy_to_device(ncoef_arr)

    # ── variable-length CSR coef packing ──────────────────────────────────
    h_offsets = cache.h_offsets  # pinned (nel,) int64
    h_re = cache.h_coef_re  # pinned (nel * coef_stride,) float64
    h_im = cache.h_coef_im

    # Step 1: compute per-element offsets (exclusive prefix sum of ncoef)
    total_active = int(_compute_coef_offsets(ncoef_arr, h_offsets))

    # Step 2: parallel pack — reads structured-array coef field directly,
    # writes to pinned flat CSR buffers
    coef_field = element_struc_array["coef"]  # (nel, coef_stride) complex128
    _fill_coef_flat(coef_field, ncoef_arr, h_offsets, h_re, h_im)

    # Step 3: transfer only the active prefix to the pre-allocated device arrays.
    # Numba supports sliced device-array destinations; the slice creates a view
    # with the matching shape so copy_to_device succeeds.
    cache.d_dynamic["el_coef_offsets"].copy_to_device(h_offsets)
    cache.d_dynamic["el_coef_re_flat"][:total_active].copy_to_device(
        h_re[:total_active]
    )
    cache.d_dynamic["el_coef_im_flat"][:total_active].copy_to_device(
        h_im[:total_active]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public host function
# ═══════════════════════════════════════════════════════════════════════════════


def build_head_matrix_cuda(
    fractures_struc_array,
    element_struc_array,
    discharge_elements,
    discharge_int,
    head_matrix,
    z_int,
    omega_int,
    _skip_dynamic=False,
    _threads_per_block=None,
):
    """CUDA build_head_matrix with CSR / jagged-array memory layout.

    Static arrays are uploaded once and cached across solver iterations.
    Only the dynamic arrays (``frac_constant``, ``el_q``, ``el_coef``) are
    re-uploaded each iteration via ``_update_dynamic``.

    VRAM savings vs the padded 2-D layout (300 k elements, discharge_int=25)
    -------------------------------------------------------------------------
    z_int:        n_de × (400-25) × 32 B ≈ -3.6 GB
    frac_elements: depends on fill factor, typically -100–200 MB
    Total: comfortably fits within 4 GB VRAM on RTX A1000.

    Call ``_device_cache.invalidate()`` when the mesh/fracture layout changes.

    Parameters
    ----------
    _skip_dynamic : bool
        Skip ``_update_dynamic`` (benchmark use only).
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    assert discharge_int <= MAX_DISCHARGE_INT, (
        f"discharge_int ({discharge_int}) exceeds MAX_DISCHARGE_INT "
        f"({MAX_DISCHARGE_INT}).  Increase the constant and recompile."
    )

    n_de = discharge_elements.shape[0]
    if n_de == 0:
        return

    nel = element_struc_array.shape[0]

    # ── first call or mesh change: upload static CSR arrays ───────────────
    if (
        not _device_cache.initialised
        or _device_cache.n_de != n_de
        or _device_cache.nel != nel
    ):
        _upload_static(
            _device_cache,
            fractures_struc_array,
            element_struc_array,
            discharge_elements,
            z_int,
            discharge_int,
        )

    # ── every call: upload dynamic arrays ─────────────────────────────────
    if not _skip_dynamic:
        _update_dynamic(_device_cache, fractures_struc_array, element_struc_array)

    d = _device_cache.d
    dd = _device_cache.d_dynamic
    d_hm = _device_cache.d_head_matrix

    # ── launch configuration ──────────────────────────────────────────────
    # Warp-per-DE kernel: tpb is always DES_PER_BLOCK * 32.
    # _threads_per_block override is kept for benchmarking different
    # DES_PER_BLOCK values without recompiling.
    if _threads_per_block is not None:
        threads_per_block = int(_threads_per_block)
        assert threads_per_block % WARP_SIZE == 0
        des_per_blk = threads_per_block // WARP_SIZE
    else:
        threads_per_block = _TPB_WARP_KERNEL  # DES_PER_BLOCK * 32
        des_per_blk = DES_PER_BLOCK
    blocks_per_grid = (n_de + des_per_blk - 1) // des_per_blk

    _build_head_matrix_kernel[blocks_per_grid, threads_per_block](
        d["de_frac0"],
        d["de_frac1"],
        d["de_type"],
        d["de_phi"],
        d["de_z_offsets"],
        d["de_z_counts"],
        dd["frac_constant"],
        d["frac_t"],
        d["frac_nelements"],
        d["frac_el_offsets"],
        d["frac_el_flat"],
        d["frac_id_arr"],
        d["el_type"],
        d["el_frac0"],
        d["el_ep0a_re"],
        d["el_ep0a_im"],
        d["el_ep0b_re"],
        d["el_ep0b_im"],
        d["el_ep1a_re"],
        d["el_ep1a_im"],
        d["el_ep1b_re"],
        d["el_ep1b_im"],
        dd["el_coef_re_flat"],
        dd["el_coef_im_flat"],
        dd["el_coef_offsets"],
        dd["el_ncoef"],
        dd["el_q"],
        d["el_radius"],
        d["el_center_re"],
        d["el_center_im"],
        d["z0_re_flat"],
        d["z0_im_flat"],
        d["z1_re_flat"],
        d["z1_im_flat"],
        d["de_out_idx"],
        d_hm,
    )

    # ── copy result back into head_matrix[:n_de] ──────────────────────────
    d_hm.copy_to_host(head_matrix[:n_de])
