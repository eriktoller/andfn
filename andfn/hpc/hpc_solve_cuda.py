"""
CUDA implementation of build_head_matrix and all dependencies.

Parallelisation strategy
------------------------
* blockIdx.x  = discharge element index  (j)
* threadIdx.x = integration-point index  (i)

Every thread computes the full omega contribution for ONE z-point across ALL
elements of its fracture.  The omega values are stored in shared memory arrays,
and a parallel tree-reduction computes the mean.

This gives up to MAX_DISCHARGE_INT (400) × more GPU parallelism than a
purely 1-D "one thread per discharge element" approach.

Requires a CUDA-enabled GPU and Numba CUDA support.

Note on ``frac0`` vs ``frac1``
------------------------------
The CPU ``build_head_matrix`` passes ``element_struc_array["frac0"]`` as the
comparison array for the first ``calc_omega_mean`` call (frac0 omega), and
``element_struc_array["frac1"]`` for the second call (frac1 omega).

For an intersection element, `frac0` stores the primary-fracture ID and
`frac1` the secondary-fracture ID.  The CPU ``build_head_matrix`` passes
``element_struc_array["frac0"]`` as the comparison array for the frac0 call,
and ``element_struc_array["frac1"]`` for the frac1 call.  Inside
``calc_omega_sum`` the condition is ``frac_is_id == frac0_param``:

* frac0 call: ``primary_id == element["frac0"]``   → **True**  → ep0a/ep0b, +coef
* frac1 call: ``secondary_id == element["frac1"]`` → **True**  → ep0a/ep0b, +coef

Both calls therefore use ep0a/ep0b with positive coefficients.  The difference
between omega0 and omega1 comes entirely from z0 vs z1 being in different
fracture coordinate systems.

This CUDA implementation passes ``el_frac0`` for the first call and
``el_frac1`` for the second call, exactly matching the CPU behaviour.
"""

import math
import numpy as np
import numba as nb
from numba import cuda

# ── compile-time constants (must match element.py / hpc_solve.py) ──────────
MAX_DISCHARGE_INT = 400  # MAX_NCOEF * 2
MAX_COEF = 200  # MAX_NCOEF
MAX_ELEMENTS_PER_FRAC = 500  # MAX_ELEMENTS
WARP_SIZE = 32

# Shared memory size — rounded up to a warp multiple so that every thread
# (including the padding threads added to fill the last warp) has a valid
# shared-memory slot.  Without this, threads beyond MAX_DISCHARGE_INT would
# write out of bounds.
SHARED_SIZE = ((MAX_DISCHARGE_INT + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE  # 416


# ═══════════════════════════════════════════════════════════════════════════════
# Complex math helpers
# (cmath / NumPy are not available in Numba CUDA device code)
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True, inline=True)
def _csqrt(re, im):
    """Return (re, im) of sqrt(re + im·j).

    Guards against slightly-negative arguments caused by floating-point
    round-off so that ``math.sqrt`` never receives a negative value.
    """
    r = math.sqrt(re * re + im * im)
    # max(0.0, ...) prevents NaN from floating-point round-off
    real_part = math.sqrt(max(0.0, (r + re) * 0.5))
    imag_part = math.copysign(math.sqrt(max(0.0, (r - re) * 0.5)), im)
    return real_part, imag_part


@cuda.jit(device=True, inline=True)
def _clog(re, im):
    """Return (re, im) of log(re + im·j).

    Uses ``math.log(r)`` with a floor to avoid ``log(0)``.
    """
    r = math.sqrt(re * re + im * im)
    # Floor at tiny positive value to avoid -inf / domain error
    if r < 1e-300:
        r = 1e-300
    return math.log(r), math.atan2(im, re)


@cuda.jit(device=True, inline=True)
def _cabs(re, im):
    """Return |re + im·j|."""
    return math.sqrt(re * re + im * im)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry device functions  (single-point versions)
# ═════════════════════════════════════���═════════════════════════════════════════


@cuda.jit(device=True, inline=True)
def _map_z_line_to_chi(z_re, z_im, ep_a_re, ep_a_im, ep_b_re, ep_b_im):
    """Joukowski map  z → χ  for a line element.  Returns (χ_re, χ_im)."""
    den_re = ep_b_re - ep_a_re
    den_im = ep_b_im - ep_a_im
    den_abs2 = den_re * den_re + den_im * den_im
    num_re = 2.0 * z_re - ep_a_re - ep_b_re
    num_im = 2.0 * z_im - ep_a_im - ep_b_im
    bZ_re = (num_re * den_re + num_im * den_im) / den_abs2
    bZ_im = (num_im * den_re - num_re * den_im) / den_abs2
    s1_re, s1_im = _csqrt(bZ_re - 1.0, bZ_im)
    s2_re, s2_im = _csqrt(bZ_re + 1.0, bZ_im)
    return bZ_re + s1_re * s2_re - s1_im * s2_im, bZ_im + s1_re * s2_im + s1_im * s2_re


@cuda.jit(device=True, inline=True)
def _map_z_circle_to_chi(z_re, z_im, r, c_re, c_im):
    """Map  z → χ  for a circle element.  Returns (χ_re, χ_im)."""
    return (z_re - c_re) / r, (z_im - c_im) / r


# ═══════════════════════════════════════════════════════════════════════════════
# Math device functions  (single chi-point)
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True)
def _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef):
    """Asymptotic expansion  Σ aₙ / χⁿ  for a single χ.  Returns (re, im)."""
    tmp_re = 0.0
    tmp_im = 0.0
    # |χ|² is constant across the Horner iterations — hoist out of the loop
    chi_abs2 = chi_re * chi_re + chi_im * chi_im
    for n in range(ncoef - 1):
        tmp_re += coef_re[ncoef - n - 1]
        tmp_im += coef_im[ncoef - n - 1]
        # tmp /= chi  ⟹  tmp * conj(chi) / |chi|²
        new_re = (tmp_re * chi_re + tmp_im * chi_im) / chi_abs2
        new_im = (tmp_im * chi_re - tmp_re * chi_im) / chi_abs2
        tmp_re = new_re
        tmp_im = new_im
    return tmp_re + coef_re[0], tmp_im + coef_im[0]


@cuda.jit(device=True)
def _taylor_series_pt(chi_re, chi_im, coef_re, coef_im, ncoef):
    """Taylor series  Σ aₙ χⁿ  for a single χ.  Returns (re, im)."""
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
    """Well potential  (q/2π) log(χ)  for a single χ.  Returns (re, im)."""
    log_re, log_im = _clog(chi_re, chi_im)
    f = q / (2.0 * math.pi)
    return f * log_re, f * log_im


# ═══════════════════════════════════════════════════════════════════════════════
# Per-element omega contribution  (single z-point)
#
# All functions are ``inline=True`` so that the compiler can fold them into
# the calling kernel without extra call overhead or register spills.
# ═══════════════════════════════════════════════════════════════════════════════


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
    """Omega for one intersection element at one z-point.

    Negation of coefficients / discharge on the secondary side is done by
    negating the *result* rather than copying and negating the coefficient
    array, which avoids a 1 600-byte ``cuda.local.array`` allocation that
    would otherwise spill to slow L2/DRAM.
    """
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
        # -Σ(aₙ/χⁿ) = Σ(-aₙ/χⁿ)  and  -(q/2π)log(χ) = (-q/2π)log(χ)
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
    z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im, coef_re, coef_im, ncoef, q
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
    z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im, coef_re, coef_im, ncoef
):
    chi_re, chi_im = _map_z_line_to_chi(z_re, z_im, ep0a_re, ep0a_im, ep0b_re, ep0b_im)
    return _asym_expansion_pt(chi_re, chi_im, coef_re, coef_im, ncoef)


# ═══════════════════════════════════════════════════════════════════════════════
# Device helper: compute omega for ONE z-point across all elements of a fracture
#
# IMPORTANT: does NOT include the fracture constant.  The constant is a scalar
# that the CPU ``calc_omega_mean`` adds once before dividing by n.  If we added
# it per z-point we would multiply it by n, producing the wrong mean.
# The caller must add the constant after the reduction.
# ══════════════════════════════════════════════════════════════════��════════════


@cuda.jit(device=True)
def _omega_one_point(
    z_re,
    z_im,
    frac_id,
    fid,  # row index into frac_elements
    frac_elements,  # 2-D int32 [nfrac, MAX_ELEMENTS_PER_FRAC]  (global)
    nelements,
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
    el_coef_re,
    el_coef_im,  # 2-D [nel, MAX_COEF]
    el_ncoef,
    el_q,
    el_radius,
    el_center_re,
    el_center_im,
):
    """Return (omega_re, omega_im) using global frac_elements array (fallback)."""
    o_re = 0.0
    o_im = 0.0
    for k in range(nelements):
        el = frac_elements[fid, k]
        etype = el_type[el]
        ncoef = el_ncoef[el]
        cr = el_coef_re[el]
        ci = el_coef_im[el]
        if etype == 0:
            dr, di = _omega_intersection_pt(
                z_re,
                z_im,
                frac_id,
                el_frac0[el],
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                el_ep1a_re[el],
                el_ep1a_im[el],
                el_ep1b_re[el],
                el_ep1b_im[el],
                cr,
                ci,
                ncoef,
                el_q[el],
            )
        elif etype == 1:
            dr, di = _omega_bounding_circle_pt(z_re, z_im, el_radius[el], cr, ci, ncoef)
        elif etype == 2:
            dr, di = _omega_well_pt(
                z_re, z_im, el_radius[el], el_center_re[el], el_center_im[el], el_q[el]
            )
        elif etype == 3:
            dr, di = _omega_const_head_pt(
                z_re,
                z_im,
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                cr,
                ci,
                ncoef,
                el_q[el],
            )
        elif etype == 4:
            dr, di = _omega_imp_circle_pt(
                z_re,
                z_im,
                el_radius[el],
                el_center_re[el],
                el_center_im[el],
                cr,
                ci,
                ncoef,
            )
        elif etype == 5:
            dr, di = _omega_imp_line_pt(
                z_re,
                z_im,
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                cr,
                ci,
                ncoef,
            )
        else:
            dr, di = 0.0, 0.0
        o_re += dr
        o_im += di
    return o_re, o_im


@cuda.jit(device=True)
def _omega_one_point_sh(
    z_re,
    z_im,
    frac_id,
    fid,
    sh_el,  # 1-D int32 shared-memory array of element indices (length nelements)
    nelements,
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
    el_coef_re,
    el_coef_im,
    el_ncoef,
    el_q,
    el_radius,
    el_center_re,
    el_center_im,
):
    """Return (omega_re, omega_im) using pre-loaded shared-memory element indices.

    ``sh_el[k]`` must contain ``frac_elements[fid, k]`` for k in 0..nelements-1,
    cooperatively loaded by all threads in the block before this call.
    Reads element indices from shared memory (L1, ~4 cycles) instead of the
    global ``frac_elements`` 2-D array (DRAM, ~100–300 cycles per miss).
    """
    o_re = 0.0
    o_im = 0.0
    for k in range(nelements):
        el = sh_el[k]  # L1 shared memory read
        etype = el_type[el]
        ncoef = el_ncoef[el]
        cr = el_coef_re[el]
        ci = el_coef_im[el]
        if etype == 0:
            dr, di = _omega_intersection_pt(
                z_re,
                z_im,
                frac_id,
                el_frac0[el],
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                el_ep1a_re[el],
                el_ep1a_im[el],
                el_ep1b_re[el],
                el_ep1b_im[el],
                cr,
                ci,
                ncoef,
                el_q[el],
            )
        elif etype == 1:
            dr, di = _omega_bounding_circle_pt(z_re, z_im, el_radius[el], cr, ci, ncoef)
        elif etype == 2:
            dr, di = _omega_well_pt(
                z_re, z_im, el_radius[el], el_center_re[el], el_center_im[el], el_q[el]
            )
        elif etype == 3:
            dr, di = _omega_const_head_pt(
                z_re,
                z_im,
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                cr,
                ci,
                ncoef,
                el_q[el],
            )
        elif etype == 4:
            dr, di = _omega_imp_circle_pt(
                z_re,
                z_im,
                el_radius[el],
                el_center_re[el],
                el_center_im[el],
                cr,
                ci,
                ncoef,
            )
        elif etype == 5:
            dr, di = _omega_imp_line_pt(
                z_re,
                z_im,
                el_ep0a_re[el],
                el_ep0a_im[el],
                el_ep0b_re[el],
                el_ep0b_im[el],
                cr,
                ci,
                ncoef,
            )
        else:
            dr, di = 0.0, 0.0

        o_re += dr
        o_im += di

    return o_re, o_im


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel tree reduction helper
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(device=True, inline=True)
def _block_reduce_sum(sh_arr, tid, n):
    """In-place parallel tree reduction of ``sh_arr[0:n]``.

    After the call, ``sh_arr[0]`` contains the sum.  All threads must
    participate (even those with ``tid >= n``).

    Uses power-of-two strides; handles non-power-of-two ``n`` correctly.
    """
    stride = 1
    while stride < n:
        stride <<= 1
    stride >>= 1  # largest power-of-2 < n

    while stride > 0:
        if tid < stride and tid + stride < n:
            sh_arr[tid] += sh_arr[tid + stride]
        cuda.syncthreads()
        stride >>= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Main CUDA kernel
#
#   Grid layout
#   -----------
#   blockIdx.x  = discharge element j      (one block per discharge element)
#   threadIdx.x = integration point i      (one thread per z-point)
#
#   Each thread independently computes omega for its z-point.
#   Shared memory holds all omega values for the block.
#   A parallel tree reduction computes the sum; thread 0 adds the constant
#   and divides by n to get the mean — matching CPU ``calc_omega_mean``.
#
#   CPU semantics of calc_omega_mean
#   --------------------------------
#   omega = constant                        # added ONCE (scalar)
#   for el: omega += calc_omega_sum(z, ...) # sum over ALL z-points
#   omega_mean = omega / n
#
#   Result: (constant + Σ_el Σ_i c(el,zi)) / n
#
#   Each thread computes Σ_el c(el, z_i) (no constant).  The reduction sums
#   these across z-points.  Thread 0 adds the constant once and divides by n.
# ═══════════════════════════════════════════════════════════════════════════════


@cuda.jit(cache=True)
def _build_head_matrix_kernel(
    # ── discharge elements ─────────────────────────────────────────────────
    de_frac0,
    de_frac1,
    de_type,
    de_phi,
    # ── fractures (indexed by fracture id) ────────────────────────────────
    frac_constant,
    frac_t,
    frac_nelements,
    frac_elements,  # int32 [nfrac, MAX_ELEMENTS_PER_FRAC]
    frac_id_arr,  # int64 [nfrac]  — _id of each fracture
    # ── elements (indexed by element position) ────────────────────────────
    el_type,
    el_frac0,
    el_frac1,
    el_ep0a_re,
    el_ep0a_im,
    el_ep0b_re,
    el_ep0b_im,
    el_ep1a_re,
    el_ep1a_im,
    el_ep1b_re,
    el_ep1b_im,
    el_coef_re,
    el_coef_im,  # [nel, MAX_COEF]
    el_ncoef,
    el_q,
    el_radius,
    el_center_re,
    el_center_im,
    # ── integration points [n_de, MAX_DISCHARGE_INT] ──────────────────────
    z0_re,
    z0_im,
    z1_re,
    z1_im,
    discharge_int,
    # ── output index (inverse sort permutation) ───────────────────────────
    de_out_idx,  # int64 [n_de]: de_out_idx[sorted_j] = original head_matrix row
    # ── output ────────────────────────────────────────────────────────────
    head_matrix,
):
    j = cuda.blockIdx.x  # discharge element index
    i = cuda.threadIdx.x  # integration point index (z-point)

    if j >= de_frac0.shape[0]:
        return

    n = discharge_int

    # ── shared memory — one slot per thread ──────────────────────────────
    # Sized to SHARED_SIZE (a warp multiple ≥ MAX_DISCHARGE_INT) so that
    # padding threads added to fill the last warp have valid slots.
    sh_omega0 = cuda.shared.array(SHARED_SIZE, dtype=nb.float64)
    sh_omega1 = cuda.shared.array(SHARED_SIZE, dtype=nb.float64)

    # Zero all slots — padding threads (i >= n) must contribute 0 to the
    # reduction so the sum equals Σ_{i<n} omega(z_i).
    sh_omega0[i] = 0.0
    sh_omega1[i] = 0.0

    # Pre-initialise to satisfy static analysis (always assigned before use)
    t1 = 1.0
    const1 = 0.0

    # ── fracture 0 ────────────────────────────────────────────────────────
    fid0 = de_frac0[j]
    const0 = frac_constant[fid0]
    t0 = frac_t[fid0]
    ne0 = frac_nelements[fid0]
    fid0_id = frac_id_arr[fid0]

    # Each thread computes omega for its own z-point (parallel over i).
    # The fracture constant is NOT included here — see reduction below.
    if i < n:
        o_re, _ = _omega_one_point(
            z0_re[j, i],
            z0_im[j, i],
            fid0_id,
            fid0,
            frac_elements,
            ne0,
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
            el_coef_re,
            el_coef_im,
            el_ncoef,
            el_q,
            el_radius,
            el_center_re,
            el_center_im,
        )
        sh_omega0[i] = o_re

    cuda.syncthreads()

    de_type_j = de_type[j]

    # ── fracture 1 (intersections only) ──────────────────────────────────
    # de_type_j is the same for every thread in the block (depends only on
    # blockIdx.x), so all threads agree on whether to enter this branch —
    # no warp divergence and the syncthreads inside is safe.
    if de_type_j == 0:
        fid1 = de_frac1[j]
        const1 = frac_constant[fid1]
        t1 = frac_t[fid1]
        ne1 = frac_nelements[fid1]
        fid1_id = frac_id_arr[fid1]

        if i < n:
            o_re, _ = _omega_one_point(
                z1_re[j, i],
                z1_im[j, i],
                fid1_id,
                fid1,
                frac_elements,
                ne1,
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
                el_coef_re,
                el_coef_im,
                el_ncoef,
                el_q,
                el_radius,
                el_center_re,
                el_center_im,
            )
            sh_omega1[i] = o_re

        cuda.syncthreads()

    # ── parallel reduction to compute sum ─────────────────────────────────
    blockdim = cuda.blockDim.x
    _block_reduce_sum(sh_omega0, i, blockdim)

    if de_type_j == 0:
        _block_reduce_sum(sh_omega1, i, blockdim)

    # ── thread 0: add constant, divide by n, write result ─────────────────
    # Matches CPU: omega_mean = (constant + Σ_el Σ_i c(el,zi)) / n
    if i == 0:
        omega0_mean = (const0 + sh_omega0[0]) / n
        out_row = de_out_idx[
            j
        ]  # map sorted block index back to original head_matrix row

        if de_type_j == 0:
            omega1_mean = (const1 + sh_omega1[0]) / n
            head_matrix[out_row] = omega1_mean / t1 - omega0_mean / t0
        elif de_type_j == 2 or de_type_j == 3:
            head_matrix[out_row] = de_phi[j] - omega0_mean


# ═══════════════════════════════════════════════════════════════════════════════
# Device array cache
#
# Static arrays (geometry, integration points) are uploaded once on the first
# call and reused every iteration.  Only the arrays that change
# between solver iterations are re-uploaded each call:
#   • frac_constant  — zeroed then updated by post_matrix_solve
#   • el_q           — zeroed then updated by post_matrix_solve
#   • el_ncoef       — increased by element_solver2 each iteration
#   • el_coef_re/im  — updated by the element solver each iteration
# ═══════════════════════════════════════════════════════════════════════════════


class _DeviceCache:
    """Holds device arrays that are static across solver iterations."""

    def __init__(self):
        self.d = {}  # static device arrays
        self.d_dynamic = {}  # pre-allocated device buffers for dynamic arrays
        self.h_coef_re = None  # pinned host buffer (nel, MAX_COEF) – reused every call
        self.h_coef_im = None  # pinned host buffer (nel, MAX_COEF) – reused every call
        self.d_head_matrix = None
        self.initialised = False
        self.n_de = 0
        self.max_el_shared = 0  # the value used when current device arrays were built

    def invalidate(self):
        """Force re-upload of static arrays on the next call (e.g. after mesh change)."""
        self.initialised = False
        self.d.clear()
        self.d_dynamic.clear()
        self.h_coef_re = None
        self.h_coef_im = None
        self.d_head_matrix = None
        self.max_el_shared = 0


_device_cache = _DeviceCache()


def _upload_static(
    cache, fractures_struc_array, element_struc_array, discharge_elements, z_int
):
    """Extract and upload all static (geometry) arrays to the device.

    Discharge elements are **sorted by frac0** before upload so that
    consecutive CUDA blocks process elements belonging to the same fracture.
    This keeps the fracture's element coefficient data (el_coef_re/im) hot in
    the GPU L2 cache, reducing DRAM bandwidth by up to N× where N is the
    average number of discharge elements per fracture.

    ``de_out_idx`` is the inverse permutation: ``de_out_idx[sorted_j]`` gives
    the original row in ``head_matrix`` that block ``sorted_j`` must write to.
    """
    # ── sort discharge elements by frac0 for L2 cache friendliness ────────
    frac0_arr = np.ascontiguousarray(discharge_elements["frac0"]).astype(np.int64)
    # sort_order[sorted_j] = original_j
    # GPU block j processes the element originally at position sort_order[j],
    # so it must write to head_matrix[sort_order[j]].
    sort_order = np.argsort(frac0_arr, kind="stable")  # stable preserves frac1 order

    de = discharge_elements[sort_order]  # sorted view

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
    # de_out_idx[sorted_j] = original_j  — maps block index back to head_matrix row
    cache.d["de_out_idx"] = cuda.to_device(sort_order.astype(np.int64))

    # also store the sorted z_int rows (z_int is indexed by discharge element order)
    z_int_sorted = z_int[sort_order]

    # ── fractures — static fields ─────────────────────────────────────────
    cache.d["frac_t"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["t"]).astype(np.float64)
    )
    cache.d["frac_nelements"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["nelements"]).astype(np.int64)
    )
    cache.d["frac_id_arr"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["_id"]).astype(np.int64)
    )
    cache.d["frac_elements"] = cuda.to_device(
        np.ascontiguousarray(fractures_struc_array["elements"]).astype(np.int32)
    )

    # ── elements — static fields ──────────────────────────────────────────
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

    # ── integration points (fixed once z_int is computed) ─────────────────
    z0_all = np.ascontiguousarray(z_int_sorted["z0"])
    z1_all = np.ascontiguousarray(z_int_sorted["z1"])
    cache.d["z0_re"] = cuda.to_device(np.ascontiguousarray(z0_all.real))
    cache.d["z0_im"] = cuda.to_device(np.ascontiguousarray(z0_all.imag))
    cache.d["z1_re"] = cuda.to_device(np.ascontiguousarray(z1_all.real))
    cache.d["z1_im"] = cuda.to_device(np.ascontiguousarray(z1_all.imag))

    # ── pre-allocate device buffers for dynamic arrays ─────────────────────
    nf = fractures_struc_array.shape[0]
    nel = element_struc_array.shape[0]
    cache.d_dynamic["frac_constant"] = cuda.device_array(nf, dtype=np.float64)
    cache.d_dynamic["el_q"] = cuda.device_array(nel, dtype=np.float64)
    cache.d_dynamic["el_ncoef"] = cuda.device_array(nel, dtype=np.int64)

    # Derive the coef width from the actual element dtype so device buffers
    # always match the host array — element_struc_array may have been built
    # with a tight dtype smaller than the global MAX_COEF constant.
    _coef_shape = element_struc_array.dtype["coef"].shape
    actual_max_coef = int(_coef_shape[0]) if _coef_shape else MAX_COEF
    cache.actual_max_coef = actual_max_coef  # remember for _update_dynamic

    cache.d_dynamic["el_coef_re"] = cuda.device_array(
        (nel, actual_max_coef), dtype=np.float64
    )
    cache.d_dynamic["el_coef_im"] = cuda.device_array(
        (nel, actual_max_coef), dtype=np.float64
    )

    # Pre-allocate pinned (page-locked) host buffers for coef arrays.
    # Pinned memory enables faster PCIe transfers and avoids malloc every call.
    cache.h_coef_re = cuda.pinned_array((nel, actual_max_coef), dtype=np.float64)
    cache.h_coef_im = cuda.pinned_array((nel, actual_max_coef), dtype=np.float64)

    # ── pre-allocate head matrix output buffer ─────────────────────────────
    cache.d_head_matrix = cuda.device_array(
        discharge_elements.shape[0], dtype=np.float64
    )

    cache.n_de = discharge_elements.shape[0]
    cache.initialised = True


def _update_dynamic(cache, fractures_struc_array, element_struc_array):
    """Copy only the arrays that change between solver iterations to the device.

    Optimisations
    -------------
    * ``frac_constant``, ``el_q``, ``el_ncoef`` are small — transferred in full.
    * ``el_coef`` reuses pre-allocated pinned host buffers (no malloc per call).
      Only the first ``active_cols`` columns are filled; the rest stay zero.
      The kernel ignores columns beyond ``el_ncoef[el]`` so zeros are safe.
    """
    cache.d_dynamic["frac_constant"].copy_to_device(
        np.ascontiguousarray(fractures_struc_array["constant"]).astype(np.float64)
    )
    cache.d_dynamic["el_q"].copy_to_device(
        np.ascontiguousarray(element_struc_array["q"]).astype(np.float64)
    )
    ncoef_arr = np.ascontiguousarray(element_struc_array["ncoef"]).astype(np.int64)
    cache.d_dynamic["el_ncoef"].copy_to_device(ncoef_arr)

    max_ncoef = int(ncoef_arr.max()) if ncoef_arr.size > 0 else 1
    actual_max_coef = getattr(cache, "actual_max_coef", MAX_COEF)
    active_cols = min(((max_ncoef + 31) // 32) * 32, actual_max_coef)

    coef_all = np.ascontiguousarray(element_struc_array["coef"])
    # coef_all may have fewer columns than active_cols if the structured-array
    # dtype was built with a smaller MAX_COEF — clamp to what is available.
    src_cols = min(active_cols, coef_all.shape[1])

    # Reuse pinned host buffers — zero only the active region then fill it.
    h_re = cache.h_coef_re
    h_im = cache.h_coef_im
    h_re[:, :src_cols] = coef_all[:, :src_cols].real
    h_im[:, :src_cols] = coef_all[:, :src_cols].imag
    # Zero any columns between src_cols and active_cols (stale or padding)
    if src_cols < active_cols:
        h_re[:, src_cols:active_cols] = 0.0
        h_im[:, src_cols:active_cols] = 0.0
    # Zero columns beyond active_cols that may hold stale data from a prior
    # iteration with a larger active_cols value.
    if active_cols < actual_max_coef:
        h_re[:, active_cols:] = 0.0
        h_im[:, active_cols:] = 0.0

    cache.d_dynamic["el_coef_re"].copy_to_device(h_re)
    cache.d_dynamic["el_coef_im"].copy_to_device(h_im)


# ═══════════════════════════════════════════════════════════════════════════════
# Host function
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
):
    """
    CUDA version of ``build_head_matrix``.

    Static geometric arrays are uploaded to the GPU once on the first call
    and cached for all subsequent iterations.  Only the three arrays that
    change between solver iterations (``frac_constant``, ``el_q``,
    ``el_coef``) are re-uploaded each call via ``_update_dynamic``, which
    uses pre-allocated pinned host buffers to minimise PCIe transfer time.

    Parameters
    ----------
    _skip_dynamic : bool, optional
        If ``True``, skip the ``_update_dynamic`` step (data already on
        device).  Use **only** in benchmarks to measure pure kernel time.
        Do **not** use in the actual solver.

    Call ``_device_cache.invalidate()`` if the mesh/fracture layout changes.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    assert discharge_int <= MAX_DISCHARGE_INT, (
        f"discharge_int ({discharge_int}) exceeds MAX_DISCHARGE_INT "
        f"({MAX_DISCHARGE_INT}). Increase the constant and recompile."
    )

    n_de = discharge_elements.shape[0]
    if n_de == 0:
        return

    # ── compute the shared-memory size needed for this DFN ───────────────
    # Round up to the next multiple of 32 for alignment; cap at MAX_ELEMENTS_PER_FRAC.
    raw_max = int(fractures_struc_array["nelements"].max())
    max_el_shared = min(
        ((raw_max + 31) // 32) * 32,
        MAX_ELEMENTS_PER_FRAC,
    )

    # ── first call (or mesh change): upload static arrays ────────────────
    if (
        not _device_cache.initialised
        or _device_cache.n_de != n_de
        or _device_cache.max_el_shared != max_el_shared
    ):
        _upload_static(
            _device_cache,
            fractures_struc_array,
            element_struc_array,
            discharge_elements,
            z_int,
        )
        _device_cache.max_el_shared = max_el_shared

    # ── every call: upload only the dynamic arrays ────────────────────────
    if not _skip_dynamic:
        _update_dynamic(_device_cache, fractures_struc_array, element_struc_array)

    d = _device_cache.d
    dd = _device_cache.d_dynamic
    d_head_matrix = _device_cache.d_head_matrix

    # ── launch configuration ──────────────────────────────────────────────
    threads_per_block = ((discharge_int + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    threads_per_block = min(threads_per_block, SHARED_SIZE, 1024)
    blocks_per_grid = n_de

    _build_head_matrix_kernel[blocks_per_grid, threads_per_block](
        d["de_frac0"],
        d["de_frac1"],
        d["de_type"],
        d["de_phi"],
        dd["frac_constant"],
        d["frac_t"],
        d["frac_nelements"],
        d["frac_elements"],
        d["frac_id_arr"],
        d["el_type"],
        d["el_frac0"],
        d["el_frac1"],
        d["el_ep0a_re"],
        d["el_ep0a_im"],
        d["el_ep0b_re"],
        d["el_ep0b_im"],
        d["el_ep1a_re"],
        d["el_ep1a_im"],
        d["el_ep1b_re"],
        d["el_ep1b_im"],
        dd["el_coef_re"],
        dd["el_coef_im"],
        dd["el_ncoef"],
        dd["el_q"],
        d["el_radius"],
        d["el_center_re"],
        d["el_center_im"],
        d["z0_re"],
        d["z0_im"],
        d["z1_re"],
        d["z1_im"],
        discharge_int,
        d["de_out_idx"],
        d_head_matrix,
    )

    # ── copy result back into the first n_de entries only ────────────────
    # head_matrix has size n_de + n_fractures; the fracture-constant rows
    # are filled separately by post_matrix_solve — the kernel only writes
    # the first n_de entries.
    d_head_matrix.copy_to_host(head_matrix[:n_de])
