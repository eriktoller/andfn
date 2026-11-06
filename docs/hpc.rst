HPC
===

This module contains the High Performance Computing (HPC) functions. The module uses Numba to create a compiled version
of the solve and plotter. The code is separated into different files, each containing a different set of functions.

The compiled function utilizes parallel processing to speed up the computation.

These functions are not to be called from the user directly, but are used by the other functions in the package.

Opt-in flat storage (Phase 1)
----------------------------

``andfn.storage`` provides a conversion boundary for the memory-layout
migration. ``DFNStorage`` is not passed to numerical kernels. Phase 2 uses
separately extracted CSR connectivity and ordered indices in the default
discharge path; coefficient arrays and persistence formats remain unchanged.

.. code-block:: python

    from andfn.storage import from_hpc, to_hpc

    storage = from_hpc(
        dfn.elements_struc_array_hpc,
        dfn.fractures_struc_array_hpc,
    )
    storage.validate()
    coefficients = storage.coefficients(0)  # writable logical-length view
    legacy_elements, legacy_fractures = to_hpc(storage)

The storage owns compact fixed-size metadata, ordered CSR fracture connectivity,
and contiguous coefficient/integration buffers. Metadata IDs must equal array
positions, as required by the existing solver. CSR IDs default to int64;
``index_dtype=np.int32`` requests checked narrowing. Global offsets remain
int64.

Coefficients, candidate coefficients, and the compatibility ``old_coef`` buffer
share starts and capacities. Theta and correction buffers share a separate
layout because ``nint`` need not equal twice ``ncoef``. Conversions copy all
logical entries, including ``nint`` correction entries, but intentionally
discard unused legacy padding. Candidate coefficients start at zero and are
not exported to the legacy element dtype.

``storage.expand(ncoef, nint)`` accepts arrays of requested lengths for the
whole batch and does not permit shrinking. Default capacities round up to
multiples of four; exceeded capacities grow by at least 1.25 times their old
size. This is an allocation policy, not an adaptive-order algorithm or a
numerical limit. Growth within capacity reuses the data buffers. Exceeding
capacity repacks the affected family of buffers once for the batch, preserving
active prefixes and zeroing newly exposed entries. Reacquire views after
expansion: views into a repacked buffer refer to the old allocation.

Expansion does not rebuild quadrature, exponentials, branch cuts, or stream
function corrections. It must run outside numerical kernels, and those caches
must be recomputed by future callers before solving at a new order. Scratch
and cache migration belongs to later phases.

``to_hpc`` validates that logical lengths and connectivity still fit the
import-time legacy dtypes, then returns independent zero-padded arrays.
Storage itself can grow beyond those limits; export then raises ``ValueError``
rather than truncating. ``validate`` checks dtypes, contiguity, lengths,
partitioning, references, CSR offsets, and non-aliasing of data buffers.

Memory accounting
~~~~~~~~~~~~~~~~~

``storage.nbytes`` counts numeric payload, not Python objects or allocator
overhead. This is the sum of owned NumPy ``array.nbytes`` values, not process
resident memory (RSS). RSS also includes compiled code, libraries, allocator
overhead, retained views, and other arrays. Temporary peak memory during
conversion or repacking is a separate measurement: both old and new allocations
can coexist, and stale views can keep an old allocation alive after repacking.
No process RSS or peak-memory measurement has been performed for these phases.

With 8-byte integer metadata, element payload is::

    192 * number_of_elements
    + 32 * number_of_elements
    + 48 * sum(coef_capacity)
    + 16 * sum(integration_capacity)

The 48-byte coefficient term includes current, candidate, and compatibility
old coefficients. Fracture metadata, connectivity IDs, and CSR offsets are
additional. Exponential caches, solver scratch, matrices, and labels are not
included in this Phase 1 representation.

For 100,000 elements with ``ncoef=10`` and ``nint=20``, default capacities
12 and 20 give 112 MB of element payload, versus 3,219.2 MB for the legacy
element dtype with storage ``MAX_NCOEF=500``. At one million elements these
become 1,120 MB and 32,192 MB. These are payload estimates, not process-memory
or runtime benchmarks. Conversion temporarily holds both representations,
and repacking temporarily holds old and new buffers. The default solver
retains padded element/fracture storage and work arrays until later phases;
Phase 2 removes selected complete-record copies, not those underlying arrays.

Python diagnostic views
~~~~~~~~~~~~~~~~~~~~~~~

``DFNStorage`` exposes ``coefficients(i)``, ``old_coefficients(i)``,
``candidate_coefficients(i)``, ``theta_values(i)``, ``dpsi_corrections(i)``,
and ``fracture_element_ids(i)``. All are writable logical-length views,
excluding capacity slack. Corrections retain ``nint`` entries for lossless
conversion, although some numerical consumers use only ``nint - 1``.
``Connectivity.element_ids(i)`` also provides a connectivity view.
These are Python convenience APIs: future Numba kernels receive numeric buffers
and offsets directly. A view acquired before repacking remains attached to its
old allocation; acquire a fresh view to observe or modify the new storage.

Phase 2 connectivity interfaces
------------------------------

``connectivity_from_hpc(fractures, nelements)`` extracts only ordered CSR IDs
and int64 offsets, without converting coefficient arrays.
``fracture_metadata_from_hpc`` copies only fixed numeric fracture fields.
All CSR kernel entry points accept plain NumPy arrays, never storage dataclasses.
Low-level kernels assume validated connectivity; Python boundaries validate IDs,
offsets, and selection ranges before invoking them.

``hpc_fracture.calc_omega_csr``, ``calc_w_csr``, and ``get_heads_csr`` accept
compact fracture metadata with no padded ``elements`` field. Element coefficients
remain in their legacy numeric dtype. The underlying scalar/array evaluators
also accept a supplied numeric ID view. Numba overload dispatch selects legacy,
supplied-view, or CSR connectivity at compile time, so compact records never
need to type-check accesses to missing legacy fields. Existing legacy signatures
and their Python test entry points remain available.

``get_heads_from_hpc`` is the Python compatibility boundary used by DFN head
plotting. It builds CSR and compact metadata once per evaluation; it does not
reconstruct padded records. An optional ordered fracture-index array supports
subsets and repeated selections without gathering padded fracture records.
The former plotting-only ``fractures_struc_array_hpc_fracs`` copy is no longer
created.

The default solve extracts CSR once and uses ``get_discharge_indices`` instead
of gathering complete discharge records. Indexed quadrature preparation,
head assembly, nonzero counting, matrix filling, discharge retrieval, and
post-solve writes dereference the original element array. ``build_discharge_matrix_csr``
validates ordered discharge positions and assembles the same CSC matrix.
``build_discharge_matrix`` retains its legacy record-based signature and
extracts connectivity once at that boundary. Existing low-level matrix
signatures remain usable; their optional CSR arguments select the new path.

Branch-cut traversal no longer gathers full neighbor records. Its optional
CSR arguments address fracture neighbors only: branch-cut event arrays,
their multiplicity, integration grids, and cache lifetimes are unchanged.
Fracture adjacency counting and selected-potential diagnostics likewise use
ordered element indices instead of full-record gathers.

Compatibility memory and remaining gathers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default coefficient/adaptive kernels still need the legacy padded fracture
and element arrays. Thus the default solve retains ``F * MAX_ELEMENTS`` integer
slots and additionally holds CSR; Phase 2 does not claim that padding has
disappeared from solver steady-state memory. Standalone CSR evaluation requires
only compact metadata plus CSR, but a legacy compatibility caller may still
own the original arrays. No hot-loop conversion back to legacy storage occurs.

CSR owned payload is::

    ids.size * ids.dtype.itemsize + (number_of_fractures + 1) * 8

For example, 1,000 fractures with degree 20 use 88,008 bytes with int32 IDs
or 168,008 bytes with int64 IDs, versus 6,400,000 bytes for the local legacy
connectivity capacity of 800 int64 slots per fracture. This excludes fixed
metadata, which uses 136 bytes per fracture in the compact layout.

Complete padded-record copies remaining intentionally are:

* ``hpc_solve.get_discharge_elements`` retains its record-returning legacy API
  for callers that require it; the default main solve no longer calls it.
* ``hpc_error_solve.get_discharge_elements`` and the copied error element state
  in ``DFN.solve_error`` remain unchanged because error-solver migration is
  explicitly out of scope.
* Selected fracture records in error-head plotting remain on the error-evaluation
  interface, also outside this phase. Ordinary head plotting uses indices.
* Explicit ``to_hpc`` conversion still allocates complete padded arrays as its
  documented compatibility contract; it is not called by CSR kernels.

Legacy coefficient kernels, adaptive growth, integration arrays, work dtypes,
and error-solver interfaces are not migrated in Phase 2. Legacy flow-net and
scalar DFN evaluation interfaces also remain available; they traverse padded
connectivity directly but do not gather whole element arrays. Cache/scratch
migration and eliminating the retained legacy arrays require later phases.

Fixed-order flat coefficients (Phase 3A)
---------------------------------------

The opt-in main fixed-order path uses coefficient-only compact records:

.. code-block:: python

    from andfn.storage import fixed_order_from_hpc, fixed_order_to_hpc
    from andfn.hpc.hpc_solve import solve_fixed_order

    state = fixed_order_from_hpc(dfn.elements_struc_array_hpc)
    _, work = solve_fixed_order(
        state, dfn.fractures_struc_array_hpc,
        dfn.discharge_int, dfn.constants,
    )
    # Optional explicit export for legacy consumers, never part of an iteration:
    legacy_elements = fixed_order_to_hpc(state)

``FixedOrderStorage.elements`` omits ``coef`` and ``old_coef`` completely.
It retains the original padded ``thetas`` and ``dpsi_corr`` fields unchanged;
this is not integration-storage migration. Current, candidate, and compatibility
old coefficients have distinct flat buffers sharing ``coef_start`` and
``coef_capacity``. Logical lengths remain in ``elements["ncoef"]``.
Neither lengths, starts, capacities nor ``nint`` change during a solve.
The existing padded work dtype and fracture records are retained.
The compatibility old buffer preserves import/export values and is not the
per-calculation convergence snapshot.

The default public DFN solve remains legacy/adaptive. At the HPC Python boundary,
``solve(..., fixed_order=True)`` provides a legacy fixed-order comparison path;
``solve_fixed_order`` validates compact state and invokes the same main driver.
Supplying flat buffers to adaptive execution raises an error. Optional
``iteration_callback(nit, elements, fractures, work, discharges, error_q, cnt_error)``
observes each completed iteration, including any final extra element solve.
Callbacks must not mutate solver state. No conversion or repacking occurs in
the iteration loop, and storage owners are never passed to Numba.

Consumer map and temporary lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Current coefficient reads: scalar/vector potential and derivative routines
  for intersection, bounding circle, constant-head line and impermeable objects.
  Fracture potential, head, Cauchy integration, discharge RHS and boundary-error
  evaluation propagate the same current buffers. Wells use discharge, not
  polynomial coefficients.
* Candidate writes: the main element ``solve`` routines write directly into
  ``candidate_coef_data`` in the compact specialization. Legacy specialization
  still writes into ``work["coef"]``.
* Temporary coefficient work: intersection ``work["coef0"]`` and ``work["coef1"]``
  hold two side integrals and are consumed within the same candidate calculation.
  They do not cross a calculation/commit barrier. Padded allocation is retained.
* Convergence comparison: ``work["old_coef"]`` snapshots the current logical
  segment within one element calculation. It is consumed before commit and does
  not need to survive that barrier, but remains allocated to avoid general work
  migration. The existing error formulas and history updates are unchanged.
* Coefficient commit: a separate compiled parallel loop applies the existing
  expression ``damping * candidate + (1 - damping) * current``. No candidate
  calculation writes current coefficients. The legacy adaptive wrapper has
  a second damping commit from iteration two even when no growth occurs;
  fixed-order execution deliberately retains it without recomputing candidates.
  The driver's extra element solve at the iteration limit is also retained.
* Reserved work: ``work["coef_error"]`` has no active main-solver consumers;
  its padded allocation remains. Compact execution does not use the retained
  ``work["coef"]`` allocation.
* Compatibility-only: ``element["old_coef"]``, conversion, unconsolidation and
  persistence remain on their existing contracts.
* Error-solver-only: ``solve_error``, error potential routines, and
  ``hpc_error_solve`` retain padded coefficients and their existing candidate
  and convergence work fields.

Numba dispatch and compiled views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hpc_coefficients.coefficient_view`` and ``candidate_view`` use Numba overloads
to select legacy fields or flat logical segments at compile time. Thus compact
records never type-check missing coefficient fields; numeric formulas are
shared, rather than duplicated legacy/compact implementations.
Optional array arguments create separate compiled specializations, not
runtime switches between incompatible record layouts.

Views exclude capacity slack. Tests verify pointer offsets and write-through
aliasing inside compiled code and inspect the view kernel LLVM for absence of
``NRT_MemInfo_alloc_aligned``. Warm benchmarks exercise the compiled consumers,
not Python slicing alone. The reverse-side intersection vector evaluator no
longer constructs a negated coefficient array; its expansion kernel negates the
scalar result. Empty expansions return zero; a zero-/one-term Taylor polynomial
has zero derivative. Candidate/integral zero-index writes are guarded for empty
segments.

This does not eliminate existing expression temporaries: damping and
bounding-circle candidate negation can create logical-length expression results;
convergence reductions and mapped-point arrays also retain their existing
temporary behavior. No global coefficient copy is made by the view helpers.
Temporary peak memory and process RSS have not been measured.

Payload and warm benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``uv run python tests\benchmark_fixed_order.py`` for a reproducible warm
comparison. It reports first-call compilation/setup separately, alternates
legacy/compact measurements, and reports medians of 21 samples. Timed iterations
include discharge RHS, LU solve, candidate calculation and commits; conversion,
reset, initial allocation and factorization are outside the timed interval.
The fixture has all six element types with non-overlapping boundaries and
heterogeneous orders. Nonfinite comparisons fail rather than counting matching
NaNs as numerical equivalence.

For 128 elements with orders 0, 8, 12, 16, 20 and 24, measured owned payload is:

.. list-table::
   :header-rows: 1

   * - Allocation
     - Bytes
   * - Legacy current coefficients
     - 409,600
   * - Legacy compatibility old
     - 409,600
   * - Compact current
     - 23,552
   * - Compact candidate
     - 23,552
   * - Compact compatibility old
     - 23,552
   * - Compact starts and capacities
     - 2,048
   * - Retained five work coef fields
     - 2,048,000
   * - Entire retained work array
     - 9,015,424
   * - Compact records (including grids)
     - 843,776
   * - Entire compact coefficient state
     - 916,480

Each compact coefficient buffer is ``16 * sum(coef_capacity)`` bytes, not
``16 * element_count * MAX_NCOEF``. Each of the five retained padded work
coefficient fields is 409,600 bytes in this fixture. Logical-length complex
expression temporaries scale with the active element order (up to 24 here),
not with all elements' capacities; their concurrent peak is not measured.

If the caller retains legacy elements, their 1,662,976 bytes coexist with the
916,480-byte compact state. The resulting 2,579,456-byte element-state payload
is larger than either representation alone; work, fractures, quadrature,
matrices, Python objects and compiled code are additional. Dropping the source
array avoids that coexistence, but neither conversion nor an explicit export
can avoid temporary overlap. No full-solver memory-reduction claim is made.

Observed warm legacy/compact milliseconds on this machine:

.. list-table::
   :header-rows: 1

   * - Threads
     - Scalar batch
     - Vector element batch
     - One iteration
     - Six iterations
   * - 1
     - 0.227 / 0.211
     - 1.296 / 1.257
     - 4.757 / 4.777
     - 21.873 / 22.545
   * - 2
     - 0.253 / 0.236
     - 1.208 / 1.214
     - 2.421 / 2.405
     - 11.455 / 11.558
   * - 28
     - 0.238 / 0.226
     - 1.236 / 1.210
     - 0.855 / 0.895
     - 4.399 / 4.297

These are local measurements, not portable performance guarantees. Main
iteration ratios in this run were between 0.977 and 1.046, below the 1.10 review
threshold. Tests compare every iteration, repeated runs, and 1, 2 and the
normal available Numba thread count. Full convergence is also compared against
the legacy adaptive driver with growth disabled: both converge in 30 iterations.
Observed maximum absolute and relative differences were zero for all 30
iteration snapshots (current/candidate coefficients, errors, discharge and
fracture constants), scalar/vector potential, derivatives, final heads,
DFN discharge and boundary errors. Regression tolerances retain
``rtol=1e-13`` and ``atol=1e-12``; IDs, orders and offsets use exact equality.

Remaining padded consumers and Phase 3B proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``element_solver2`` adaptive coefficient checks and its adaptive commit remain
legacy-only. Default DFN evaluation, flow-net/plotting APIs, Python element
objects, unconsolidation, save/load, and error-solver evaluation still expect
legacy records unless explicitly given flat-buffer arguments at a migrated
HPC entry point. ``get_max_min_phi`` retains its legacy diagnostic signature.
Integration fields and every work field remain padded, even on the compact
fixed-order path. Explicit exports remain available; no legacy fields or APIs
are removed.

Proposed Phase 3B synchronization (not implemented): preserve the first commit,
then record growth requests without changing active lengths inside parallel
work; leave the parallel region; validate and batch-plan capacity growth;
allocate/copy all affected families transactionally; publish new
buffers/offsets/lengths together; invalidate and rebuild order-dependent grids,
exponentials, quadrature and branch-event caches; recompute candidates only for
elements requiring the existing adaptive recalculation, before the existing
second commit. Unaffected candidates must retain their staged values. In
particular, growing coefficients cannot safely ignore the corresponding
integration/work capacity and cache requirements. Repacking or publishing new
offsets from inside ``prange`` remains prohibited.

Solver functions
----------------

.. automodule:: andfn.hpc.hpc_solve
   :members:
   :undoc-members:
   :show-inheritance:

Fracture functions
------------------

.. automodule:: andfn.hpc.hpc_fracture
   :members:
   :undoc-members:
   :show-inheritance:

Bounding circle functions
-------------------------

.. automodule:: andfn.hpc.hpc_bounding_circle
   :members:
   :undoc-members:
   :show-inheritance:

Constant head line functions
----------------------------

.. automodule:: andfn.hpc.hpc_const_head_line
   :members:
   :undoc-members:
   :show-inheritance:

Intersection functions
----------------------

.. automodule:: andfn.hpc.hpc_intersection
   :members:
   :undoc-members:
   :show-inheritance:

Well functions
--------------

.. automodule:: andfn.hpc.hpc_well
    :members:
    :undoc-members:
    :show-inheritance:

Mathematical functions
----------------------

.. automodule:: andfn.hpc.hpc_math_functions
    :members:
    :undoc-members:
    :show-inheritance:

Geometry functions
------------------

.. automodule:: andfn.hpc.hpc_geometry_functions
    :members:
    :undoc-members:
    :show-inheritance:
