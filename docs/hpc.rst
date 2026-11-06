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

Phase 3A left ``element_solver2`` adaptive coefficient checks and its adaptive
commit legacy-only; the opt-in Phase 3B path below now handles both. Default
DFN evaluation, flow-net/plotting APIs, Python element
objects, unconsolidation, save/load, and error-solver evaluation still expect
legacy records unless explicitly given flat-buffer arguments at a migrated
HPC entry point. ``get_max_min_phi`` retains its legacy diagnostic signature.
Integration fields and every work field remain padded, even on the compact
fixed-order path. Explicit exports remain available; no legacy fields or APIs
are removed.

The Phase 3A synchronization proposal was to preserve the first commit,
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

Phase 3B: opt-in adaptive flat coefficients
------------------------------------------

``andfn.hpc.hpc_adaptive.adaptive_from_hpc(elements)`` constructs an
``AdaptiveSolver`` owner. Call ``solve_adaptive(owner, fractures,
discharge_int, constants)`` and obtain the published storage/work through
``owner.state`` afterwards. This is opt-in: the legacy fixed/adaptive paths
and compact fixed-order path are unchanged. No persistence, public DFN default,
error-solver or general work-storage migration is included.

Legacy sequence preserved
~~~~~~~~~~~~~~~~~~~~~~~~~

The original ``element_solver2`` sequence is:

1. If ``max_error >= 1``, skip the element step entirely.
2. Calculate first candidates from current coefficients for every element,
   including the original error/history updates and type-specific integration.
3. Commit every candidate using ``damping * candidate + (1-damping) * current``.
4. Only from iteration two, assess coefficient ratios/errors using the
   committed coefficients. Keep the existing ratio arithmetic, including
   multiplication by zero, error-history checks and thresholds.
5. For eligible non-wells, check the *old* order against ``MAX_NCOEF``, then add
   ``COEF_INCREASE`` once. The existing ``max_cnt=1`` prevents another increase.
   The numerical maximum can therefore be overshot; this behavior is retained.
6. For each growing element set ``nint = 2 * new_ncoef``. This rule is not an
   invariant for unchanged elements, which may have independent ``nint``.
7. Rebuild theta values, reset the branch-event count, rebuild both exponential
   arrays, and call ``fill_z_integral`` in that order.
8. Recalculate that element's candidate at the expanded order. The existing
   type-specific solve rebuilds dpsi corrections and branch events where used,
   and updates coefficient/element errors again.
9. Retain the post-recalculation checks; ``max_cnt=1`` prevents another growth
   and the legacy ``set_zero`` flag ends false.
10. Commit all candidates a second time, including unaffected elements'
    previously calculated candidates, with the same damping expression.
11. The driver retains its separate discharge/RHS solve before each element
    step, convergence checks afterwards, and final extra candidate/commit at
    the iteration limit.

The legacy implementation interleaves steps 4--9 inside ``prange``. The compact
implementation deliberately separates assessment, publication and recalculation
by parallel barriers, as required for safe reallocation. Zero newly exposed
coefficients make this equivalent in the regression fixtures; both the actual
legacy loop and an instrumented staged padded reference are checked. Intermediate
transition snapshots use the staged reference, not a claim that the original
interleaved loop has a global publication barrier.

The approved exception for independent integration counts is important:
growth may *shrink logical nint* when the exact legacy rule requires it.
Physical integration capacity is retained. Tests characterize ``ncoef=4,
nint=30 -> ncoef=5, nint=10`` and old-order numerical-maximum overshoot.
Requests beyond retained physical work/integration limits raise rather than
clamping, silently changing the maximum rule, or accessing out of bounds.

Assessment, construction and publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``assess_growth`` is a nopython parallel kernel receiving plain arrays.
It reads lengths, offsets, capacities and current coefficients without writing
active storage; requested ncoef/nint and a growth mask are separate arrays.
Candidate coefficients also remain untouched during assessment. Integer request
arithmetic is checked before addition/multiplication.

On a growth batch, Python validates IDs, requests, nonshrinking coefficient
lengths, exact legacy integration rules, maximum eligibility, physical limits,
and prefix/capacity overflow. The quantum-4, factor-1.25 policy is unchanged.
``build_replacement`` privately allocates three zero-filled coefficient buffers,
copies logical prefixes, stages coefficient-free integration records and
retained work, rebuilds affected caches, then validates the replacement.
New current/candidate/compatibility-old entries are zero; capacity slack is
never copied into newly exposed entries. Local convergence history in
``work.old_coef`` remains on the existing type-specific calculation path.

There is at most one coefficient layout repack per batch. Within-capacity
growth still allocates private staging buffers to guarantee rollback, but
does not change starts/capacities and is counted separately from a layout
repack. Copy accounting includes this staging; a zero repack count does not
mean zero allocation or copying.

Only after all failure-prone construction/cache validation succeeds does
``owner.state = replacement`` publish the complete state. The orchestration
then reacquires all numerical arrays, recalculates selected candidates against
the *full* element array, and performs the second commit. It never uses a
singleton slice as the neighbor array. No owner/dataclass enters a compiled
kernel. Normal execution reuses the compiled ``element_solver`` for the first
candidate/commit pair; diagnostic callbacks split those same shared kernels
to observe the intervening state.

No-growth steps reuse the previously validated state rather than scanning its
entire padded work/integration allocation again. The standalone transaction
builder still validates no-growth requests and returns the original state
identity without allocations. Transaction failures preserve published array
identities and contents; the rollback boundary is the committed state at
transaction entry, not the beginning of the whole iteration.

Integration/cache bridge and invalidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full flat integration migration is unnecessary in this phase. Theta/dpsi
arrays and work caches retain their existing physical padded capacity; thus
there are **zero physical integration repacks**. A coefficient-only layout
change, integration/cache-only growth within spare coefficient capacity, and
simultaneous coefficient/grid changes are tested through this bridge. Tests do
not pretend to exercise integration-family relocation that cannot occur here.

For affected elements, a rebuilt theta grid invalidates branch-event
*topology*, even though fracture CSR connectivity is unchanged. Clearing
``len_discharge_element`` forces the existing circle/line routines to rebuild
crossing positions, IDs and signs. Their existing dpsi reconstruction also
updates discharge-dependent values. A discharge change alone affects correction
values, not geometric connectivity. Unaffected grid/cache records are copied
unchanged. Intersection ``z_integral`` follows its existing geometry/grid
builder; driver discharge quadrature depends on geometry and ``discharge_int``,
not adaptive ``nint``, and remains valid.

Consumers and limitations
~~~~~~~~~~~~~~~~~~~~~~~~

Compact growth assessment, candidate recalculation, damping commits and the
main adaptive driver use flat current/candidate arrays. Existing compile-time
coefficient overloads avoid missing-field typing on compact records.
``work.coef0``, ``work.coef1`` and local convergence history remain padded
temporary compatibility storage; the padded candidate field remains allocated
but compact persistent candidates use only ``candidate_coef_data``.

The legacy ``element_solver2`` still reads padded current coefficients,
recalculates padded candidates and commits padded fields. Default Python/DFN
evaluation, legacy wrappers/exports, error solving, plotting/unconsolidation
and persistence still require their existing padded representations. No export
occurs inside adaptive iterations, and no legacy field or path is removed.
Import rejects nonzero unused current/compatibility-old padding: otherwise
later growth would silently lose previously initialized legacy tail values.
Supporting those noncanonical tails requires a separate design decision.

Validation and measured behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive comparisons use all six element types, heterogeneous starting orders,
independent integration counts, no/single/multiple/consecutive growth, spare
capacity, repacking, overshoot, and repeated execution at 1, 2 and the normal
28 Numba threads. Requests, lengths, IDs and trajectories use exact comparison;
finite numerical values use the existing baseline-derived ``rtol=1e-13``,
``atol=1e-12``. Every iteration compares current/candidate/compatibility-old
coefficients, coefficient/element errors, fracture constants, discharge,
DFN discharge, heads, boundary errors and convergence state. Transition tests
add integration/cache/event and local-history comparisons.

Observed maximum absolute and relative differences were both zero in all 12
reported iteration snapshots, and in final benchmark coefficients.
Failure tests cover invalid requests/IDs, simulated and real prefix overflow,
allocation failure, cache exceptions/nonfinite values, and unchanged published
identities/contents after failure.

The final ``uv run pytest tests`` run passed all 286 tests. Both project Ruff
commands were attempted but Ruff was unavailable in the environment. Pinned
``uvx ruff@0.16.2`` lint and formatting checks passed for the changed Python
files; ``git diff --check`` also passed. Dependencies/configuration were not
changed.

``uv run python tests\benchmark_adaptive.py --peak-rss`` reports warm medians of
seven alternating paired runs, excluding storage import, matrix construction,
LU factorization and compilation/cache loading. Each run includes six
discharge/RHS plus adaptive-element iterations. Stage timing totals describe
the final compact sample, not the median of independent stage timings.
Compilation/cache loading and setup took 1.07 seconds in the final recorded run.

.. list-table::
   :header-rows: 1

   * - Case
     - 1 thread legacy/compact ms
     - 2 threads legacy/compact ms
     - 28 threads legacy/compact ms
   * - 128, no growth
     - 18.883 / 19.096
     - 24.711 / 23.261
     - 4.148 / 4.303
   * - 128, forced repacks
     - 119.539 / 142.672
     - 64.622 / 97.718
     - 11.468 / 39.106
   * - 128, growth within spare capacity
     - 14.218 / 21.919
     - 7.314 / 12.750
     - 2.474 / 7.442
   * - 1024, simultaneous/consecutive growth
     - 990.606 / 1187.831
     - 355.006 / 496.961
     - 76.189 / 301.042

The initial redundant no-growth validation was removed after measurement.
Heterogeneous no-growth ratios in this run were 0.941--1.037. Growth-heavy timings remain
materially worse, principally because transaction staging/validation copies
retained padded work/integration allocations even when coefficients fit spare
capacity. At 28 threads, summed detection/staging/cache/recalculation
milliseconds were 0.243/13.109/1.006/6.344 for 128 forced-growth elements and
0.326/94.516/6.554/51.400 for 1024 elements. Complete transaction times, including
validation and cache rebuilding, were 29.508 and 196.923 ms respectively.
Staging alone excludes request and replacement validation. Local timings vary
with shared-machine load; they are not portable performance guarantees.

A separate 24-iteration spare-capacity investigation at 28 threads measured
9.533/15.484 ms overall (1.624x), and 1.118x across iterations after growth.
The host-only orchestration with stubbed numerical kernels measured 5.85 us
per step; profiling also shows three dispatcher crossings per assessment step
versus the legacy compiled orchestration. Host bookkeeping explains only part
of the residual dispatch/parallel-stage overhead; this low-order case remains
over the approximate 10-percent review threshold. It is not presented as
performance-approved. No unrelated numeric optimization or work migration is
used to conceal the regression.

Payload, capacity and copying
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy payload measurements (not process RSS):

.. list-table::
   :header-rows: 1

   * - Case
     - Initial/final capacity sum
     - Max ncoef
     - Batches / coefficient repacks / integration repacks
     - Expanded element events
     - Final utilization
   * - 128, no growth
     - 1472 / 1472
     - 24
     - 0 / 0 / 0
     - 0
     - 100%
   * - 128, forced repacks
     - 1472 / 4544
     - 44
     - 5 / 5 / 0
     - 448
     - 81.69%
   * - 128, spare capacity
     - 1152 / 1152
     - 10
     - 1 / 0 / 0
     - 96
     - 83.33%
   * - 1024, forced repacks
     - 11776 / 36352
     - 44
     - 5 / 5 / 0
     - 3584
     - 81.69%

For 128 elements, legacy current and old payloads are 409,600 bytes *each*.
Each compact family is ``16 * sum(coef_capacity)``: 23,552 bytes initially,
72,704 after forced growth, or 18,432 in the spare-capacity fixture.
Current, candidate and compatibility-old are three distinct allocations.
Retained integration is 819,200 bytes; full retained work is 9,015,424 bytes,
including 2,048,000 bytes for its five padded coefficient fields. The final
forced-growth compact state plus work is 10,079,360 bytes. Retaining source
legacy records adds another 1,662,976 bytes, excluding fractures, discharge
matrices, quadrature and Python/compiled-code overhead.

Across five forced-growth batches, coefficient prefixes copy 583,680 bytes,
integration fields 4,096,000, other metadata 122,880, and work 45,077,120:
49,879,680 bytes total. The spare-capacity transaction copies 9,900,672 bytes
despite zero layout repacks. The 1024-element totals are eight times the
128 forced-growth copy counts. Logical-length numeric expression temporaries
remain unchanged; their concurrent peak is not measured.

Maximum old/new published-state coexistence during forced transactions is
20,124,928 bytes at 128 elements and 160,999,424 at 1024 elements, excluding
source legacy records, request/planning arrays and expression temporaries.
The isolated 4096-element Windows OS measurement observed process-lifetime
peak working set rising from 855,756,800 to 1,114,804,224 bytes; simultaneous
old/new NumPy state payload was 636,821,504 bytes. These are lifetime high-water
measurements, not a precise incremental-repack RSS attribution. The optional
probe uses Windows ``GetProcessMemoryInfo`` or Unix ``getrusage``, with no new
dependency.

Stop after Phase 3B. Padded work staging cost, the low-order residual slowdown,
physical integration/work limits and initialized-unused-tail compatibility
remain review items before any general work-array migration. This phase makes
no complete-solver memory-reduction claim.

Phase 4A: separate compact work ownership
-----------------------------------------

The legacy ``dtype_work`` and error solver are unchanged. The compact adaptive
owner instead publishes a shallow ``WorkArrays`` tuple of separately owned
plain NumPy arrays. ``work_record`` constructs element-local *views* for the
same numerical kernels; compile-time overloads select legacy records or compact
views. No Python storage owner enters nopython code. This is per-element work,
not per-thread scratch.

The Phase 3B guard at the legacy coefficient/integration export limits remains
in force. Compact work is capacity-sized, but this phase does not separately
approve previously unsupported numerical orders.

Work lifetime audit
~~~~~~~~~~~~~~~~~~~

The table refers to the original ``hpc_solve.dtype_work``. Producers/consumers
were checked across repository Python code, including error solving and
diagnostics. ``E`` is element-local scratch, ``B`` persistent cache, ``C``
iteration state, ``D`` candidate state, ``F`` branch events, ``G`` compatibility,
and ``H`` apparently unused. Persistent committed numerical state (category A)
is in element/coefficient storage, not this work dtype.

``local`` below means no numerical dependency beyond that element calculation:
values may physically remain after a commit, growth or solve, but are not
required by later calculations. Diagnostic exports can inspect them. ``grid``
means valid across iterations/commits until integration geometry changes;
``events`` means valid until branch topology/grid invalidation.

.. list-table::
   :header-rows: 1
   :widths: 18 9 28 20 25

   * - Field
     - Class
     - Producer -> consumers
     - Semantic lifetime
     - Size/dependency
   * - phi
     - E
     - real Cauchy integration -> same reduction
     - local; not needed after return
     - nint; no topology dependency
   * - psi
     - E
     - imaginary/domega integration -> differences and reduction
     - local; not needed after return
     - nint; correction values depend on events
   * - coef
     - D/G
     - legacy type solves -> legacy commits/error driver
     - crosses candidate/commit barrier; overwritten next calculation
     - ncoef; removed only from compact work
   * - coef0
     - E
     - intersection first-fracture integral -> intersection combination
     - local; not required across commits/growth
     - ncoef
   * - coef1
     - E
     - intersection second-fracture integral -> intersection combination
     - local; not required across commits/growth
     - ncoef
   * - coef_error
     - H
     - no live repository producer/consumer found
     - no established numerical lifetime
     - retained at coefficient capacity, not MAX_NCOEF
   * - a0_error
     - H
     - no live repository producer/consumer found
     - no established numerical lifetime
     - one complex scalar per element retained
   * - old_coef
     - E
     - each type solve snapshots current -> calc_error/calc_coef_error
     - local convergence snapshot; diagnostics retain last value
     - ncoef; preserved prefix on coefficient relocation
   * - dpsi
     - E
     - domega integrals -> cumulative psi reconstruction
     - local; not required across commits/growth
     - nint; corrected using current branch values
   * - error
     - H
     - no work-field consumer (element.error is distinct)
     - no established numerical lifetime
     - one real scalar per element retained
   * - integral
     - E
     - Cauchy routines -> coefficient calculation
     - local; reset before use
     - ncoef, despite legacy allocation of 2*MAX_NCOEF
   * - sign_array
     - F
     - find_branch_cuts/get_sign -> dpsi correction/diagnostics
     - events; survives commits and unchanged-grid iterations
     - event capacity; topology dependent
   * - discharge_element
     - F
     - find_branch_cuts -> correction/diagnostics
     - events; survives commits and unchanged-grid iterations
     - event capacity; topology dependent
   * - element_pos
     - F
     - find_branch_cuts -> correction/diagnostics
     - events; survives commits and unchanged-grid iterations
     - event capacity; topology and nint dependent
   * - len_discharge_element
     - F
     - find_branch_cuts/invalidation -> correction/diagnostics
     - events; reset on changed grid
     - one integer per element
   * - exp_array_m
     - B
     - fill_exp_array -> Cauchy recurrence
     - grid; survives solve iterations and commits
     - nint
   * - exp_array_p
     - B
     - fill_exp_array -> evaluation, branch and z-grid construction
     - grid; survives solve iterations and commits
     - nint
   * - z_integral
     - B/H
     - fill_z_integral -> diagnostic export; numeric reads commented out
     - grid; retained rather than deleting an unused field
     - nint for intersection geometry
   * - set_zero
     - C/G
     - legacy assessment -> legacy commit/count; compact always false
     - iteration only; no persistent numerical dependency
     - one bool per element retained

No work field is needed by subsequent public evaluations of a completed
compact solution. The legacy DFN stores returned work as ``self.work_array``;
tests/diagnostics inspect coefficient history, caches and branch fields.
Compact diagnostic exports remain explicit and are never execution bridges.
Category H fields are retained pending a separate removal decision.

Theta is a persistent grid cache. Dpsi correction is iteration state, rebuilt
from current discharge and persistent branch events whenever its type solve
runs. Both move from padded element fields into sized integration ownership.
Exponentials are persistent per-element caches in this phase; converting them
or Cauchy temporaries into thread scratch is explicitly deferred.

Compact representation and transaction boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coefficient work fields and ``integral`` use independent flat NumPy arrays
sized by coefficient capacity. Theta, dpsi correction, exponentials,
``z_integral``, phi, psi and dpsi use independent arrays sized by integration
capacity. Compact element metadata contains no padded integration arrays.
Coefficient and integration capacities use the established quantum-4,
factor-1.25 policy and can change independently.

The only remaining padded arrays are the three branch-event arrays, retaining
``MAX_ELEMENTS`` event slots per element. Event counts are not fracture degree;
changing their growth policy would combine a separate topology migration with
this phase. The scalar branch count is independently owned. No compact work
field retains MAX_NCOEF padding.

Construction produces a ``GrowthTransaction`` with affected metadata and
theta/correction/exponential/z cache segments, plus replacement families only
when their own capacity is exceeded. It makes no writes to published state.
All allocations, cache rebuilding and validation finish before publication.
The commit barrier then performs prevalidated scalar/slice writes, zeros newly
exposed coefficient/history entries, invalidates affected event counts,
updates logical lengths, and publishes the owner state. There are no compiled
workers executing during publication. Atomicity here is the solver barrier,
not a promise of lock-free concurrent Python readers.

Within coefficient capacity, current/candidate/compatibility-old and local
history arrays retain identity: no coefficient copying or relocation occurs.
Within integration capacity, integration buffers also retain identity; only
affected logical cache segments are patched. A coefficient repack copies
current, candidate, compatibility-old, local ``old_coef`` and retained
``coef_error`` prefixes; coefficient-local scratch is freshly allocated without
copying dead values. It neither copies nor replaces integration/branch arrays.
An integration repack copies only theta, correction, exponentials and
``z_integral`` prefixes; dead phi/psi/dpsi scratch is freshly allocated.
Branch arrays and scalar work arrays retain identity for every growth case.

Before publication, failures preserve every active buffer identity and value.
After successful publication, references to old layouts must not be used;
even a previous state object can share in-place buffers and metadata. Numerical
arrays are reacquired at the driver boundary, as in Phase 3B. Exported snapshots
copy their values instead of retaining mutable views.

The overloads for work-field indexing intentionally use ordinary compiled
lowering, not forced Numba inlining. Forced inline field access lost nested
branch-sign writes during initial verification; exact branch/cache transition
tests caught this. Existing arithmetic and reduction order are unchanged.

Phase 4A validation and measured cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``uv run pytest tests`` passed all 296 tests. Legacy/compact comparisons now
require exact equality after every observed transition and iteration, including
current/candidate/history coefficients, errors, caches and branch events.
Repeated runs at 1, 2 and 28 threads preserve exact growth requests and order
trajectories. The reported 12-iteration legacy/compact comparison has maximum
absolute and relative differences of zero. Cross-thread boundary reductions
can differ in the legacy baseline (one observed difference was
``3.46944695e-18``); tests require compact cross-thread deltas to equal the
legacy deltas exactly, without changing tolerances or arithmetic.

Failure injection covers the original construction/cache failures and
within-capacity cache failure with every owned buffer identity/value checked.
The six transaction scenarios below use a small heterogeneous eight-element
fixture, with ``10 -> 12`` coefficient growth, independent coefficient capacity
16 (or 10 to force relocation), and integration capacity 32 (or 20 to force
relocation). The several-element case grows six elements in one batch.

.. list-table::
   :header-rows: 1

   * - Scenario
     - Coefficient copy bytes
     - Integration copy bytes
     - Cache rebuild bytes
     - Unrelated work / branch copy bytes
   * - A: no growth
     - 0
     - 0
     - 0
     - 0 / 0
   * - B: within both capacities
     - 0
     - 2,016
     - 1,536
     - 0 / 0
   * - C: coefficient repack only
     - 4,800
     - 2,016
     - 1,536
     - 0 / 0
   * - D: integration repack only
     - 0
     - 9,952
     - 1,536
     - 0 / 0
   * - E: both repack
     - 4,800
     - 9,952
     - 1,536
     - 0 / 0
   * - F: several elements, both repack
     - 4,800
     - 20,032
     - 9,216
     - 0 / 0

Coefficient repack/relocation counters are also exactly zero in B. Integration
copy counts include affected correction/z prefixes staged for preservation and
the final cache-segment publication; these are not whole-work copies. Cache
rebuild counts are writes constructing the replacement grids/caches, reported
separately rather than counted as coefficient relocation.

The original benchmark cases, orders, six iteration counts, seven alternating
paired samples, discharge solve and warm-execution boundaries are unchanged.
New accounting and an optional ``--profile`` add separate diagnostic runs.
The final warm legacy/compact milliseconds were:

.. list-table::
   :header-rows: 1

   * - Fixture
     - 1 thread
     - 2 threads
     - 28 threads
   * - 128, no growth
     - 19.184 / 18.805
     - 9.825 / 10.005
     - 4.165 / 4.177
   * - 128, forced repacks
     - 83.204 / 96.255
     - 42.408 / 55.514
     - 12.355 / 34.855
   * - 128, spare-capacity growth
     - 10.869 / 12.572
     - 5.838 / 7.471
     - 2.603 / 5.600
   * - 1024, multiple expansions
     - 675.767 / 771.762
     - 344.626 / 442.390
     - 91.772 / 263.548

Compared directly with the Phase 3B review numbers at 28 threads, spare growth
improved from 7.442 to 5.600 ms (24.7%), and 1024-element multiple expansion
improved from 301.042 to 263.548 ms (12.5%). The final no-growth paired ratios
were 0.980, 1.018 and 1.003. Shared-machine timing varied between runs; these
are measurements, not portable latency guarantees.

The separate warm instrumented six-step element-only spare-capacity profile
at 28 threads measured detection 0.608 ms, transaction construction excluding
cache rebuild 1.799 ms, publication 0.786 ms, cache rebuild 0.134 ms, candidate
recalculation 0.557 ms, commits 0.315 ms and remaining element work 2.433 ms.
This diagnostic uses split first-candidate/commit calls to expose commit time;
it is not substituted for the unchanged comparable benchmark. Ordinary
benchmark JSON also reports transaction/publication/recalculation/second-commit
totals for the final sample separately from median elapsed times.

Spare growth remains materially slower than legacy, so Phase 4A stops here
without claiming final performance approval. CProfile of 100 within-capacity
constructions took 0.214 seconds: per-element finite-prefix checks alone used
38,400 generator calls and about 0.081 seconds cumulatively; request/layout
validation and many small NumPy reductions are additional overhead. Publication
still executes per-element cache patches in Python. These costs now scale
with changed state, not the padded work payload, but are expensive at low order.
The 24-iteration residual probe measured 11.918/17.893 ms and 1.243x after
the growth iteration; another run measured 1.051x after growth. Steady-step
profiling attributes most time to compiled numerical/parallel calls plus their
dispatcher boundary, not Python request detection. The residual is
load-sensitive and is not hidden with looser numerical tolerances or new
thread-local scratch.

Phase 4A payload comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

Payload accounting deduplicates shared NumPy owners by identity. Coefficient
payload below is the sum of current, candidate and compatibility-old arrays.
Integration/cache includes theta, correction, both exponentials and
``z_integral``. Work includes sized temporaries/history, scalar state, event
storage and offset arrays. Total state accounting counts shared coefficient
offsets only once.

.. list-table::
   :header-rows: 1

   * - Fixture
     - Coefficients
     - Integration/cache
     - Compact work
     - Total state
     - Transaction temporary
     - Peak simultaneous state
   * - 128, no growth
     - 70,656
     - 196,608
     - 2,657,408
     - 2,949,248
     - 0
     - 2,949,248
   * - 128, forced growth
     - 218,112
     - 524,288
     - 3,026,048
     - 3,793,024
     - 1,628,288
     - 5,280,512
   * - 128, spare capacity
     - 55,296
     - 131,072
     - 2,607,232
     - 2,818,176
     - 148,608
     - 2,966,784
   * - 1024, multiple expansions
     - 1,744,896
     - 4,194,304
     - 24,208,384
     - 30,344,192
     - 13,026,304
     - 42,244,096

These are bytes, not RSS. Peak accounts for shared old/new families once,
replacement caches, requests and transaction control arrays; transient NumPy
expression allocations, BLAS/matrix/quadrature storage and compiled/Python
overhead are not included. No-growth has no transaction peak beyond its
persistent state. Explicitly retained legacy source records add 1,662,976
bytes at 128 elements or 13,303,808 at 1024; they are not required by the
compact iteration. Each 128-element legacy coefficient field remains
409,600 bytes. Legacy work, if separately retained by the caller, adds its
unchanged payload as well.

The Phase 3B 128-element forced-growth comparison was 819,200 bytes of padded
integration, 9,015,424 bytes of work, 10,079,360 bytes of compact state plus work,
and 20,124,928 bytes of old/new coexistence. Phase 4A reduces work by 66.4%,
state by 62.4%, and the measured transaction peak by 73.8%. Its cache category
now includes exponentials and ``z_integral``, which were counted inside work
in Phase 3B, so those two category columns are not identical inventories.
Branch arrays alone still consume 2,457,600 bytes at 128 elements.

Total copying across 128 forced-growth batches fell from 49,879,680 to
4,826,112 bytes; spare-capacity transaction copying fell from 9,900,672 to
182,784 bytes, all affected integration/cache/metadata rather than coefficient
or unrelated work. At 1024 elements total copying fell from 399,037,440 to
38,608,896 bytes. Both families repack five times in the forced cases; neither
repacks in the spare-capacity case.

The optional isolated 4096-element Windows probe reported a lifetime peak
working set of 582,414,336 bytes both before and after the forced transaction:
the repack did not exceed the earlier process high-water mark. Measured
simultaneous transaction state was 124,395,520 bytes, versus Phase 3B's
636,821,504 bytes. The unchanged RSS high-water mark is not a zero-memory-cost
claim.

Project Ruff commands were attempted but unavailable; pinned Ruff 0.16.2
lint/format checks on changed Python files and ``git diff --check`` passed.
No dependencies, persistence, numerical algorithms, legacy solver dtype or
error solver were changed. No Phase 4B implementation is included.

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
