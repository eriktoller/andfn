HPC
===

This module contains the High Performance Computing (HPC) functions. The module uses Numba to create a compiled version
of the solve and plotter. The code is separated into different files, each containing a different set of functions.

The compiled function utilizes parallel processing to speed up the computation.

These functions are not to be called from the user directly, but are used by the other functions in the package.

Current runtime architecture
----------------------------

The normal ``DFN.solve()``, ``DFN.solve_error()``, and post-processing paths use
compact state directly. Element metadata is fixed-size; coefficients and
integration caches use contiguous capacity-based buffers; fracture connectivity
uses CSR; branch events and worker scratch scale with their actual capacities;
and quadrature is allocated from the selected rows and requested count. No
normal runtime path constructs padded HPC records.

Adaptive growth prepares replacement buffers outside numerical kernels, then
publishes the new compact state. ``MAX_COEFFICIENT_ORDER`` is an algorithmic
limit and does not determine allocation size. Historical ``MAX_NCOEF`` config
values are accepted as a deprecated alias; ``MAX_ELEMENTS`` and
``MAX_DISCHARGE_INT`` no longer size runtime storage. Persistence continues to
store logical Python object state, so loading does not reconstruct the former
padded runtime.

The phase sections below preserve the migration and benchmark history. Their
descriptions of opt-in compact modes, padded runtime state, and outstanding
work describe the historical phase at the time, not the current runtime.

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
error solver were changed in the Phase 4A snapshot.

Phase 4B: affected-only adaptive control plane
--------------------------------------------

``build_replacement`` now explicitly distinguishes no growth, within-capacity
logical growth, coefficient-only repacking, integration-only repacking and
both-family repacking. Only physical repacks enter ``build_repack``. The
production no-growth step skips transaction construction and owner publication
entirely, while retaining the legacy second commit.

Logical growth prepares eight payload arrays: affected IDs, packed cache
offsets, affected metadata, and the five cache arrays. It keeps the exact same
``AdaptiveState``, ``AdaptiveStorage`` and ``WorkArrays`` identities. No complete
ownership tuple is rebuilt. The three assessment request arrays remain alive
until the second commit; their bytes are included in transaction peak
accounting. Offsets and preserved-prefix byte accounting are computed in
compiled serial loops rather than Python element loops or NumPy expression
temporaries. A small ``CacheStage`` adapter preserves custom cache-failure
callbacks without constructing a full work ownership tuple, even on that
opt-in diagnostic path.

Preparation validates affected IDs, old-order eligibility, requested lengths,
the approved legacy integration shrink and maximum overshoot, and affected
segment bounds. It preserves only affected correction/z prefixes, constructs
theta/exponential/z values privately, and checks all five prepared arrays in
one compiled finite-value loop. It neither reads nor copies unchanged current,
candidate, compatibility-old or convergence-history prefixes. Physical repacks
retain complete structural validation and the previous rollback-by-private-
replacement behavior.

The compiled serial publication kernel zeros only newly exposed coefficient/
history slots, copies prepared cache segments, invalidates affected event
counts, and finally publishes logical lengths. Candidate recalculation
reacquires owner buffers afterward. Branch topology is still rebuilt by the
original subsequent numerical operations, not transaction preparation.
Publication and event invalidation share one dispatch; there is no separate
parallel launch for either. No detection/publication fusion, numerical-stage
fusion, synchronization removal or commit-order change is introduced.

Validation levels and failure model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Structural:** ``validate_state(state)`` checks complete storage/work layout
  and writable ownership at import or explicit public calls. Physical repacks
  retain complete storage/work checks before publication.
* **Transaction:** ordinary logical growth checks only request shape/dtype,
  affected metadata/segment endpoints, destination writability and newly
  prepared cache values. No complete-state or unchanged-prefix finite scan
  occurs. The three affected metadata passes are validation, offset
  construction and cache preparation.
* **Debug:** ``validate_state(state, finite=True)`` additionally scans element
  numeric metadata, all three persistent coefficient families, coefficient
  work prefixes, all integration work prefixes and scalar work.
  ``owner.debug_validation = True`` enables this before every publication.
  It defaults to false.

All allocations, cache building, finite checks, debug validation and the
optional ``owner.before_publication`` injection hook run before publication.
The hook observes the committed pre-growth state; raising there leaves it
unchanged. Valid publication consists solely of bounded writes in a compiled
serial kernel: no allocation, user callback or validation remains inside it.
This assumes an exclusively owned, structurally valid state and no mutation
of prepared requests/buffers between preparation and publication. Concurrent
mutation, asynchronous process termination and memory corruption are outside
that contract; publication is not advertised as recoverable from them.
No whole-state rollback clone is retained.

The new failure contracts compare every published buffer identity and value
after preparation failure, nonfinite prepared values, read-only destinations
and an exception immediately before publication. Successful and failed cases
run explicit complete structural/debug validation. Existing construction,
cache, branch-event and numerical trajectory tests remain unchanged.

Call profiles and publication measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before editing, 100 warm spare-capacity constructions took 0.230 seconds and
435,701 profiled calls. Finite-prefix generator checks accounted for 38,400
calls and 0.086 cumulative seconds; ``np.all`` had 29,300 calls. A final
instrumented control using the retained Phase 4A replacement builder measured
0.227 seconds / 440,301 calls, including the additional accounting. Its finite
generator cost was 0.087 seconds. The optimized path measured 0.011 seconds /
14,301 calls at 28 threads. Warm unprofiled construction medians were
80.0/78.0/81.0 microseconds at 1/2/28 threads.

The instrumented 100-construction API inventory is:

.. list-table::
   :header-rows: 1

   * - Operation
     - Retained replacement-builder control
     - Affected-only construction
   * - Python ``np.zeros`` / ``np.empty`` / ``np.ones``
     - 1500 / 100 / 100
     - 500 / 100 / 0
   * - Python ndarray ``copy`` / ``astype``
     - 400 / 700
     - 0 / 0
   * - ``flatnonzero``
     - 300
     - 100
   * - Python ``_layout`` / complete ``_validate_layout``
     - 900 / 600
     - 0 / 0
   * - Complete storage / work validation
     - 200 / 200
     - 0 / 0
   * - Complete work tuple reconstruction
     - 200
     - 0

These are API calls, not an assertion that NumPy ufunc intermediates are
visible to CProfile. Each optimized transaction additionally allocates one
offset array inside compiled ``patch_layout`` and the ID payload through
``flatnonzero``: eight builder payloads in total. Source-level affected-element
Python loops are eliminated from preparation/publication. There are four
measured construction dispatcher crossings per transaction: affected
validation, offset construction, cache preparation and finite validation.
All four kernels are serial; construction launches no parallel region.
Publication adds one serial crossing and no ownership-state publication.
Only the small patch/result objects are reconstructed.

The compiled finite check alone measured 3.2/3.2/3.3 microseconds per
transaction, or approximately 0.00033 seconds per 100 at 28 threads. This
checks newly prepared caches only, unlike the old finite-prefix scan.
Five cache arrays are checked together without concatenating them.

For the 96-affected-element spare fixture, serial publication measured
9.0/9.5/14.6 microseconds at 1/2/28 threads. A benchmark-only parallel
compilation of the same kernel measured 5.9/7.1/32.4 microseconds.
At 28 threads parallel launch overhead outweighed its work reduction.
The production implementation therefore remains serial; no unmeasured
element-count threshold was introduced. A trivial serial dispatcher measured
0.2 microseconds, while a trivial parallel launch measured 0.5/1.1/12.6
microseconds. These probes are not subtracted from real numerical timings:
actual kernels have different argument-marshalling and scheduling costs.
Standalone event-count invalidation, including its dispatcher, measured
0.3/0.3/0.5 microseconds. In production it is part of publication and adds no
separate dispatcher.

The 28-thread six-step instrumented element-only profile measured detection
0.236 ms, construction excluding cache building 0.094 ms, cache preparation
0.088 ms, publication including event invalidation 0.022 ms, adaptive candidate
recalculation 0.474 ms, commits 0.331 ms and remaining element work 1.523 ms.
Construction submeasurements were request extraction/patch allocation
0.033 ms, affected validation 0.017 ms and finite validation 0.007 ms.
Submeasurements overlap the aggregate construction category and must not be
added to it again. Phase 4A's corresponding construction/publication totals
were 1.799/0.786 ms. The callback-instrumented split-stage diagnostic is not
substituted for the unchanged comparable benchmark.

Runtime comparison and remaining review gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fixtures, seven alternating paired samples and six-iteration timed
boundaries remain unchanged. The new logical-growth kernels are explicitly
warmed before timing. A 24-iteration spare-capacity case now also runs at each
thread setting. Final legacy/compact median milliseconds were:

.. list-table::
   :header-rows: 1

   * - Fixture
     - 1 thread
     - 2 threads
     - 28 threads
   * - 128, no growth
     - 18.754 / 18.979
     - 9.657 / 9.855
     - 3.811 / 4.264
   * - 128, forced repacks
     - 83.335 / 96.016
     - 42.469 / 55.011
     - 12.682 / 34.429
   * - 128, spare-capacity growth
     - 11.177 / 10.979
     - 5.573 / 5.856
     - 2.844 / 3.204
   * - 1024, multiple expansions
     - 678.006 / 765.796
     - 353.683 / 447.152
     - 94.399 / 269.469
   * - 128 spare capacity, 24 iterations
     - 34.242 / 35.077
     - 18.828 / 19.932
     - 9.796 / 11.297

The corresponding 10th--90th percentile intervals, legacy / compact ms:

.. list-table::
   :header-rows: 1

   * - Fixture
     - 1 thread
     - 2 threads
     - 28 threads
   * - 128, no growth
     - [18.391, 19.573] / [18.791, 19.489]
     - [9.495, 10.316] / [9.700, 10.309]
     - [3.687, 4.385] / [3.981, 4.508]
   * - 128, forced repacks
     - [82.732, 84.029] / [95.245, 97.323]
     - [41.860, 42.929] / [54.369, 55.883]
     - [11.838, 15.011] / [33.564, 37.498]
   * - 128, spare capacity
     - [10.638, 11.788] / [10.906, 11.535]
     - [5.401, 7.134] / [5.773, 6.235]
     - [2.602, 3.245] / [3.029, 3.587]
   * - 1024, multiple expansions
     - [669.392, 680.729] / [761.554, 771.914]
     - [344.159, 368.962] / [441.341, 462.534]
     - [89.212, 101.193] / [251.905, 281.014]
   * - 128 spare capacity, 24 iterations
     - [33.539, 35.233] / [34.818, 40.269]
     - [17.891, 20.077] / [18.984, 22.141]
     - [9.361, 10.506] / [11.044, 11.945]

Retained compact medians side by side (each cell lists 1/2/28 threads, ms):

.. list-table::
   :header-rows: 1

   * - Fixture
     - Phase 3B
     - Phase 4A
     - Phase 4B
   * - 128, no growth
     - 19.096 / 23.261 / 4.303
     - 18.805 / 10.005 / 4.177
     - 18.979 / 9.855 / 4.264
   * - 128, forced repacks
     - 142.672 / 97.718 / 39.106
     - 96.255 / 55.514 / 34.855
     - 96.016 / 55.011 / 34.429
   * - 128, spare capacity
     - 21.919 / 12.750 / 7.442
     - 12.572 / 7.471 / 5.600
     - 10.979 / 5.856 / 3.204
   * - 1024, multiple expansions
     - 1187.831 / 496.961 / 301.042
     - 771.762 / 442.390 / 263.548
     - 765.796 / 447.152 / 269.469
   * - 128 spare capacity, 24 iterations
     - not retained / not retained / 15.484
     - not retained / not retained / 17.893
     - 35.077 / 19.932 / 11.297

The exact retained Phase 4A critical baseline is legacy 2.603 ms, compact
5.600 ms at 128 elements / spare growth / 28 threads. The Phase 4B paired
measurement is legacy 2.8444 ms, compact 3.2035 ms: **1.1262x**, not a pass
against the requested 1.10 threshold. At 1/2 threads the ratios are
0.9823/1.0508. The 24-iteration ratios are 1.0244/1.0586/1.1532.
An earlier run measured 1.1192x for the six-step 28-thread case, so this is not
reported as a sustained target achievement. Shared-machine dispersion and
legacy baseline movement remain visible rather than selecting a favorable run.

A separate profile of 200 post-growth element steps took 0.060 seconds;
0.058 seconds were inside the adaptive entry point's compiled stage calls
and their dispatch/marshalling, with roughly 0.002 seconds in visible Python
bookkeeping. Native compute versus scheduling time is not separately
attributable from CProfile. Adaptive candidate calculation, commit launches,
retained work views and physical repack validation/copying remain significant.
No stage fusion was adopted without evidence of a benefit that justifies
changing stage visibility. Detection, publication and both commits retain
their synchronization boundaries. Stop at Phase 4B for review; no thread-local
scratch or Phase 4C migration is included.

Payload and failure accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within-capacity construction has exactly eight new numeric payloads; including
the three assessment requests gives eleven. At 128 elements / 96 affected,
the builder allocates 142,848 bytes, requests add 2,176, and simultaneous
persistent-plus-transaction payload is 2,963,200 bytes. Of the builder bytes,
122,880 are new caches, 18,432 affected metadata and 1,536 IDs/offsets.
No current, candidate, compatibility/history coefficient, unrelated work or
branch payload is copied. Copies comprise 41,472 affected correction/z prefix
bytes, 122,880 cache publication bytes and 18,432 metadata bytes.
No full-state scan or ownership replacement occurs.

The heterogeneous eight-element transaction scenarios use the same fixtures
and growth requests as the Phase 4A ownership tests:

.. list-table::
   :header-rows: 1

   * - Scenario
     - Live temporary payload arrays
     - Temporary bytes
     - Simultaneous bytes
     - Coefficient copy
     - Integration/cache copy
     - Complete structural passes
   * - No growth
     - 0
     - 0
     - persistent only
     - 0
     - 0
     - 0
   * - Within capacity
     - 11
     - 1,880
     - 187,424
     - 0
     - 2,016
     - 0
   * - Coefficient only
     - 25
     - 10,600
     - 191,536
     - 4,800
     - 2,016
     - 2
   * - Integration only
     - 25
     - 14,120
     - 193,328
     - 0
     - 9,952
     - 2
   * - Both
     - 35
     - 22,696
     - 197,296
     - 4,800
     - 9,952
     - 2
   * - Several, both
     - 35
     - 38,856
     - 213,456
     - 4,800
     - 20,032
     - 2

Unrelated-work and branch copy bytes are zero in every scenario. Logical
growth has three affected metadata passes plus one combined prepared-cache
finite pass, followed by two serial affected publication loops. Physical
repacks retain the prior complete request/layout validation and affected
finite-prefix checks. No-growth production creates no transaction; its
compiled growth assessment still creates the original three request arrays.

Across the comparable benchmark, 128 forced-growth transactions retain at
most 35 new payload arrays / 1,628,288 bytes per transaction, and simultaneous
state peaks at 5,280,512 bytes. Across five batches, coefficient and
integration/cache copies total 972,800 and 3,767,296 bytes, respectively.
For 1024 elements the corresponding temporary/peak payloads are
13,026,304 / 42,244,096 bytes and copy totals 7,782,400 / 30,138,368 bytes.
Physical repack accounting is unchanged from Phase 4A.

Array counts describe identity-deduplicated transaction payloads, including
replacement arrays and requests, not zero-copy view objects. As in the
retained baseline, physical-repack figures exclude transient validation
ufunc expressions, Python objects and Numba/NRT bookkeeping; they are not
allocator-complete cumulative allocation or OS RSS measurements. The logical
fast path has no such NumPy expression temporaries: its eight builder
payloads and the three assessment payloads are explicitly accounted for.
The separate API inventory exposes constructor/copy calls rather than hiding
this measurement limitation.

Exact transition/iteration comparisons retain current/candidate/history,
requests and lengths, coefficient/element errors, all retained caches,
corrections and branch state, fracture constants, discharge/DFN discharge,
heads, boundary errors and convergence. Recorded maximum absolute and
relative differences remain zero. Same-thread comparisons and the legacy
cross-thread reduction deltas remain exact; no tolerance changed.
All original failure tests remain in place.

Validation uses a fresh, session-local Numba cache after the call-graph
changes. An initial run reusing the Phase 4A cache hit a native access
violation; fresh-cache targeted and full runs succeeded. Old compiled caches
must not be treated as compatible proof after solver/layout changes.
The project Ruff commands were attempted but the executable was unavailable.
Pinned Ruff 0.16.2 checks are used on the five changed Python files; no
dependency or project configuration was changed.

The final ``uv run pytest tests`` passed all **311 tests**, including all
original failures and fifteen new control-plane/failure cases.
``uv run ruff check .`` and ``uv run ruff format --check .`` both failed to
start because Ruff is not installed in the project environment.
``uvx ruff@0.16.2 check`` and ``uvx ruff@0.16.2 format --check`` passed on all
five changed Python files. ``git diff --check`` and the corresponding
``git diff --check HEAD`` check of staged plus unstaged changes passed.

Changed files are ``hpc_adaptive.py``, ``hpc_math_functions.py``,
``benchmark_adaptive.py``, the new ``benchmark_adaptive_control.py`` and
``test_adaptive_control.py``, and this documentation. No thread-local scratch,
error-solver or persistence migration, cache removal, legacy removal,
capacity-policy change or numerical-limit correction is included.

Phase 4C: worker scratch and ragged branch events
------------------------------------------------

The Phase 4B reference remains commit ``c865fa7`` and the existing adaptive
benchmark retains its fixtures, seven alternating paired samples and timing
boundaries. The retained 28-thread medians are no-growth compact 4.264 ms and
spare growth legacy/compact 2.8444/3.2035 ms (1.1262x). Phase 4C does not further
optimize transactions in isolation.

Scratch lifetime and ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Logical size
     - Producer / consumers
     - Minimum lifetime
   * - coef0 / coef1
     - ncoef complex values each
     - intersection's two real Cauchy calculations / weighted candidate
     - both must coexist until that intersection's candidate is assembled
   * - integral
     - ncoef complex values
     - real/imaginary Cauchy reductions / coefficient output
     - one Cauchy calculation; reset before reuse
   * - phi
     - nint real values
     - real potentials / real Cauchy reduction
     - one Cauchy calculation
   * - psi
     - nint real values
     - imaginary potentials, then corrected cumulative stream function /
       imaginary Cauchy reduction
     - one Cauchy calculation
   * - dpsi
     - nint real values
     - adjacent psi differences and branch corrections / cumulative psi
     - same imaginary Cauchy calculation

None of these six fields crosses the candidate parallel barrier, survives a
commit numerically, or is a neighbor-element input. All consumers finish in
the same element invocation. The candidate/current arrays and local
``old_coef`` convergence snapshots remain separately element-owned. The latter
are deliberately **not** migrated to scratch, preserving diagnostic histories.
No helper below the candidate loop opens another parallel region.

Per-element temporary histories were observable through ``work_record`` and
``export_work``. Worker reuse cannot preserve those histories. Explicit
``adaptive_from_hpc(elements, scratch_groups=0)`` therefore retains the old
capacity-sized element-owned scratch for diagnostics. Default production mode
uses group mask 7; masks 1 and 3 allow separately reproducing the coefficient
and integral migration measurements. ``export_work(..., include_scratch=False)``
explicitly exports persistent state only. Asking for temporary histories in
worker mode raises a descriptive error instead of returning another element's
last worker values. Importing nonzero element scratch into worker mode is also
rejected. The element-owned diagnostic mode has an exact six-field regression
against the legacy solver.

The isolated ``benchmark_working_set.py`` prototype compares capacity-sized
element buffers, public-worker buffers, and explicit chunk buffers. It writes
heterogeneous orders 8/12/16/24/32/48/64 with independent integration lengths,
checks exact per-element results and untouched guard columns after repeated
parallel execution, and requires nopython signatures. It never uses
``element_id % thread_count``. Chunk mode explicitly partitions element ranges.
Worker mode uses the documented public ``numba.get_thread_id()``:
https://numba.readthedocs.io/en/stable/user/threading-layer.html#numba.get_thread_id

Each worker's buffers are exclusive during the single candidate region.
Rows are sized from observed coefficient/integration **capacities**, not
``MAX_NCOEF``. Slot count is the configured maximum
``numba.config.NUMBA_NUM_THREADS``, so later changes to the active thread mask
do not outgrow ownership. One solver owner must not be used by concurrent
invocations or reentrant candidate regions. Allocation and resizing occur
outside parallel kernels. A physical order/grid repack may enlarge worker
strides, but scratch values are never copied. Logical growth within existing
capacity does not replace scratch.

At 10,000 elements, isolated element/worker/chunk medians were:

.. list-table::
   :header-rows: 1

   * - Threads
     - Element ms
     - Worker ms
     - Chunk ms
   * - 1
     - 3.355
     - 1.246
     - 1.222
   * - 2
     - 1.957
     - 0.688
     - 0.767
   * - 4
     - 1.329
     - 0.363
     - 0.406
   * - 8
     - 1.009
     - 0.328
     - 0.332
   * - 16
     - 0.580
     - 0.212
     - 0.198
   * - 28
     - 0.470
     - 0.146
     - 0.135

These are isolation/locality probes, not solver speedup claims. Public worker
ownership was chosen because it preserves the existing prange scheduling and
barriers without imposing fixed chunks. The six probe arrays use complex
values to check exact writes; production phi/psi/dpsi remain real arrays.

Migration was gated separately: coef0/coef1, then integral, then phi/psi/dpsi.
Exact numerical suites ran after each group, followed by actual candidate
kernels at 128/1,024/10,000 elements. A noisy initial 1,024-element 28-thread
comparison prompted a new alternating element-owned/worker-owned control
inside the same process, rather than accepting a comparison across runs.
At 10,000 elements, worker/element ratios across 1/2/4/8/16/28 threads were
1.000/0.991/1.106/0.995/0.998/0.991 for coef0/coef1,
0.986/0.998/0.980/0.993/1.013/0.990 after integral, and
1.008/0.996/0.990/1.090/0.993/0.896 after phi/psi/dpsi.
The isolated 1.106 result did not persist at the next gate; no sustained
large-kernel slowdown above 10 percent was observed. Group payloads at 10,000
elements before branch migration were 226,751,536, 224,922,288 and
219,194,544 bytes, respectively, versus 230,410,000 bytes before migration.

Branch semantics and storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``find_branch_cuts`` visits integration intervals first, then fracture
connectivity order. For an intersection or constant-head line it emits direct
events before optional mirrored events. A well can be crossed repeatedly on
a non-monotone path; mirrored intersection crossings can carry opposite signs.
Events therefore form an **ordered multiset**, not a neighbor set or fracture
CSR. Each event stores integration interval, discharge-element ID and sign.
``get_dpsi_corr`` accumulates ``q * sign`` in precisely that event order; this
order affects floating-point summation and is preserved exactly.

Topology depends on geometry, fracture connectivity and the integration
grid, not discharge. Discharge changes update correction values by traversing
the existing events. Adaptive nint changes invalidate only affected event
counts. Zero count retains the old meaning: either invalid or a genuinely
empty topology, so empty cases still rebuild when the legacy code would.
Geometry/connectivity are assumed fixed during a solve.

The compact representation uses flat int64 ``sign_array``,
``discharge_element`` and ``element_pos`` with ``branch_start`` and
``branch_capacity``. The existing per-element ``len_discharge_element`` is
the logical length. Event-producing types initially reserve four slots;
other types reserve none. An exact count-only pass uses the **same crossing
routine** and interval/neighbor order, with bounded writes into a private
single-slot record. It does not construct published topology early.
Counts exceeding capacity are collected and repacked together at a host
barrier using the existing quantum-4 / 1.25 growth helper for this independent
event family. Coefficient/integration growth policies are unchanged.
Only live branch segments are copied when a branch repack is necessary.
Candidate calculations still construct topology at the original numerical
point. Shared helper arithmetic, direct/mirror signs and correction order
are unchanged; a missing preparation now raises before consuming overflowed
main-solver event arrays.

Initial preparation runs once per owner, before the first candidate region.
For adaptive growth, branch capacity preparation runs after cache preparation
but **before** the failure hook or publication. Replacement branch buffers
remain private until publication. Other work families are shared, not cloned.
A branch allocation failure leaves every pre-growth buffer identity, length,
cache, event and convergence value unchanged. Tests inject this failure after
the first commit/assessment and compare the complete state at that boundary.

The heterogeneous eight-element seed has event counts
``[3, 0, 0, 0, 1, 0, 2, 0]``: maximum 3, mean 0.75 over all elements
(1.5 over its four event-producing elements). Histogram is
``0:5, 1:1, 2:1, 3:1``. Tiled large fixtures retain this distribution.
No duplicate IDs or position/ID pairs occur in this seed, which is not
evidence that duplicates are impossible. A separate adversarial circle/well
test crosses the same well branch twice and repeats the neighbor 800 times,
requiring 1,600 ordered events. It checks every event against the repeated
small padded reference, beyond the legacy 800-slot capacity. Legacy storage
is unchanged; diagnostic export explicitly rejects compact live event counts
that cannot fit its legacy dtype.

z_integral audit
~~~~~~~~~~~~~~~~

A repository-wide audit covered kernels, tests, diagnostics, plotting,
persistence, examples and local Python scripts. The only potential numerical
reader is commented out in real Cauchy integration. Active writers remain
in cache initialization, adaptive preparation and the error/legacy solvers.
However, ``export_work`` is an active diagnostic consumer and exact cache
transition tests read this field. Accordingly **z_integral is retained**,
not silently deleted. A regression changes all compact z values to different
finite values and requires identical candidates, coefficients and numerical
metadata while demonstrating that the diagnostic cache values differ.
Removal would be a separate diagnostic-API/cache-contract change; it is not
bundled with scratch or branch migration.

Owned payload and remaining padding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The no-growth heterogeneous fixture's category allocation, bytes:

.. list-table::
   :header-rows: 1

   * - Category
     - 128
     - 1,024
     - 10,000
     - 100,000 storage-only
   * - Fixed element metadata/layout
     - 28,832
     - 230,432
     - 2,250,032
     - 22,500,032
   * - Coefficient state including convergence history
     - 94,208
     - 753,664
     - 7,360,000
     - 73,600,000
   * - Integration/cache
     - 196,608
     - 1,572,864
     - 15,360,000
     - 153,600,000
   * - Worker scratch
     - 64,512
     - 64,512
     - 64,512
     - 64,512
   * - Branch events and metadata
     - 9,216
     - 73,728
     - 720,000
     - 7,200,000
   * - Retained compatibility work
     - 26,624
     - 212,992
     - 2,080,000
     - 20,800,000
   * - Total compact
     - 420,000
     - 2,908,192
     - 27,834,544
     - 277,764,544
   * - Phase 4A/4B same-fixture compact
     - 2,949,248
     - 23,593,984
     - 230,410,000
     - 2,304,100,000
   * - Legacy work alone
     - 9,015,424
     - 72,123,392
     - 704,330,000
     - 7,043,300,000

Logical versus capacity bytes and utilization are produced separately by
``uv run python tests\benchmark_working_set.py --memory``. For this fixture,
metadata, coefficient state, compatibility and concurrent worker scratch
have 100% utilization. Integration logical bytes are 97.9167% of capacity:
192,512 / 1,540,096 / 15,040,000 / 150,400,000 bytes at the four sizes.
Branch logical bytes including offsets/counts are 5,376 / 43,008 / 420,000 /
4,200,000, or 58.3333% of capacity. Compatibility includes retained
``coef_error``, ``a0_error`` and scalar work ``error``; none is silently removed.
Coefficient state counts current/candidate/compatibility-old and local
convergence history exactly once.

Worker scratch scales as
``T * (3 * max(coef_capacity) * 16 + 3 * max(integration_capacity) * 8)``,
not N times either legacy maximum. Branch payload scales with independently
measured event capacities plus O(N) offsets/counts, not N*MAX_ELEMENTS.
Even the diagnostic element-owned scratch mode uses real capacities rather
than MAX_NCOEF.

The kernel timing probe allocates no new Python NumPy ownership buffers
inside its timed region: owned compact peak equals the totals above. It
reports the larger persistent **benchmark harness** payload separately,
because legacy, element-owned control and worker-owned states coexist.
Native numerical kernels still allocate Numba-managed temporaries; these
are not claimed to be zero or measured by the owned-NumPy accounting.
Physical repack figures, as in Phase 4B, are payload bounds rather than
allocator-complete transient-expression or process-RSS measurements.
Initial worker scratch allocation directly allocates worker-sized arrays;
it does not first allocate and discard all six per-element families.

Quadrature is deliberately separate: the main driver still retains two
``dtype_z_arrays`` arrays with MAX_NCOEF padding, totaling 3,276,800 /
26,214,400 / 256,000,000 / 2,560,000,000 bytes at the four sizes. They are
not allocated or used in the candidate-only kernel probe. Caller-retained
legacy element+work records add 10,678,400 / 85,427,200 / 834,250,000 /
8,342,500,000 bytes and are not necessary for compact candidate execution.
Fracture connectivity/metadata, matrices and Python/compiled-code overhead
are additional. This phase makes no whole-process or full-large-solve memory
claim.

Candidate-kernel scaling and dispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actual-kernel benchmark tiles heterogeneous eight-element DFNs. It is
representative of the existing element calculations and orders, but is
**not** a globally connected 10,000-element solve. Branch capacity preparation
is outside these candidate-only timings. Legacy, element-owned compact
control and worker-owned compact runs alternate, reversing order on each
sample. Eight samples are taken; the first is discarded. All three owners
coexist, and their combined payload is reported separately from compact
ownership. Run with ``uv run python tests\benchmark_working_set.py --solver``.

The repeat measurement below follows an initial complete run, rather than
replacing or concealing its unfavorable results. Cells contain median ms
``[p10, p90]``; speedups are T1/Tn, legacy / compact.

.. list-table::
   :header-rows: 1

   * - Elements
     - Threads
     - Legacy ms [p10, p90]
     - Compact ms [p10, p90]
     - Speedup legacy / compact
   * - 128
     - 1
     - 2.770 [2.715, 3.056]
     - 2.726 [2.710, 2.797]
     - 1.00 / 1.00
   * - 128
     - 2
     - 1.386 [1.364, 1.440]
     - 1.401 [1.395, 1.449]
     - 2.00 / 1.95
   * - 128
     - 4
     - 0.707 [0.696, 0.774]
     - 0.735 [0.722, 0.979]
     - 3.92 / 3.71
   * - 128
     - 8
     - 0.641 [0.616, 0.734]
     - 0.650 [0.643, 0.928]
     - 4.32 / 4.19
   * - 128
     - 16
     - 0.461 [0.400, 0.509]
     - 0.433 [0.402, 0.532]
     - 6.01 / 6.30
   * - 128
     - 28
     - 0.412 [0.391, 0.464]
     - 0.489 [0.452, 0.604]
     - 6.72 / 5.57
   * - 1,024
     - 1
     - 22.156 [21.717, 22.696]
     - 22.320 [21.510, 22.648]
     - 1.00 / 1.00
   * - 1,024
     - 2
     - 11.216 [10.930, 11.632]
     - 11.087 [10.853, 11.280]
     - 1.98 / 2.01
   * - 1,024
     - 4
     - 5.760 [5.639, 6.338]
     - 5.772 [5.627, 7.667]
     - 3.85 / 3.87
   * - 1,024
     - 8
     - 5.253 [4.444, 5.362]
     - 5.413 [4.615, 5.433]
     - 4.22 / 4.12
   * - 1,024
     - 16
     - 3.076 [3.034, 3.297]
     - 3.064 [3.018, 3.090]
     - 7.20 / 7.29
   * - 1,024
     - 28
     - 2.284 [2.051, 2.757]
     - 2.297 [2.028, 2.303]
     - 9.70 / 9.71
   * - 10,000
     - 1
     - 217.087 [214.316, 228.315]
     - 217.651 [212.439, 221.893]
     - 1.00 / 1.00
   * - 10,000
     - 2
     - 114.439 [114.057, 119.055]
     - 113.936 [109.132, 118.344]
     - 1.90 / 1.91
   * - 10,000
     - 4
     - 58.646 [56.343, 64.713]
     - 57.320 [56.562, 60.067]
     - 3.70 / 3.80
   * - 10,000
     - 8
     - 37.271 [36.259, 44.852]
     - 35.510 [33.693, 41.625]
     - 5.82 / 6.13
   * - 10,000
     - 16
     - 31.365 [31.212, 32.278]
     - 31.530 [30.650, 31.736]
     - 6.92 / 6.90
   * - 10,000
     - 28
     - 19.553 [18.914, 20.674]
     - 18.948 [18.624, 19.441]
     - 11.10 / 11.49

The first run's 10,000-element legacy / compact medians at
1/2/4/8/16/28 threads were 224.060/226.589, 111.414/112.166,
60.417/60.478, 46.488/41.715, 31.718/31.964 and 19.738/21.658 ms.
At 28 threads its intervals were [19.101, 28.106] / [19.901, 29.481]
and speedups 11.35 / 10.46, not a scaling improvement.
Its worker/element-owned control ratios were
0.975/1.007/0.945/0.999/0.996/1.050. The repeat ratios were
0.987/0.997/0.985/0.873/1.005/0.940. There is no sustained
10,000-element worker-scratch regression above 10 percent, but these
measurements also do not establish a general parallel-scaling improvement.

The first run's 1,024-element two-thread legacy median was 23.229 ms,
versus 11.216 ms on repeat: machine variability was substantial.
Its 28-thread worker/control ratio of 1.160 fell to 1.001 on repeat.
The first 128-element eight-thread worker/control ratio of 1.702 fell
to 0.989; the repeat's 128-element 28-thread result is itself unfavorable.
These small timings are not used to select a favorable storage design or
claim a universal speedup.

The machine reports an Intel Core i7-13850HX, 20 physical / 28 logical
processors, 28,672 KiB L2 and 30,720 KiB L3, with approximately 63.8 GiB
OS-visible memory. No PMU events or memory-bandwidth counters were measured.
The working-set reduction is established by owned payload; a cache-miss
reduction, bandwidth saturation point, or causal cache-scaling improvement
is **not** established by these timings.

Unchanged adaptive benchmark and performance review gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unchanged ``benchmark_adaptive.py --profile`` boundaries include branch
capacity preparation. Final legacy / compact median ms:

.. list-table::
   :header-rows: 1

   * - Fixture
     - 1 thread
     - 2 threads
     - 28 threads
   * - 128, no growth
     - 21.553 / 25.121
     - 10.024 / 11.670
     - 3.382 / 4.183
   * - 128, forced repacks
     - 88.276 / 132.944
     - 43.069 / 71.302
     - 11.283 / 39.153
   * - 128, spare growth
     - 12.588 / 16.590
     - 6.177 / 8.518
     - 2.241 / 3.456
   * - 1,024, multiple expansions
     - 711.597 / 1039.683
     - 352.903 / 544.431
     - 66.977 / 265.557
   * - 128, spare growth, 24 iterations
     - 35.686 / 43.458
     - 18.764 / 21.900
     - 7.699 / 9.758

The corresponding legacy / compact p10--p90 intervals, ms:

.. list-table::
   :header-rows: 1

   * - Fixture
     - 1 thread
     - 2 threads
     - 28 threads
   * - 128, no growth
     - [19.723, 22.659] / [23.333, 25.775]
     - [9.906, 10.528] / [11.514, 11.983]
     - [3.292, 3.777] / [3.899, 4.230]
   * - 128, forced repacks
     - [87.087, 93.521] / [127.633, 135.029]
     - [42.627, 45.625] / [69.235, 74.472]
     - [11.021, 12.001] / [37.866, 42.044]
   * - 128, spare growth
     - [11.660, 12.896] / [15.993, 17.430]
     - [5.985, 6.481] / [8.269, 8.822]
     - [2.220, 2.738] / [3.354, 3.976]
   * - 1,024, multiple expansions
     - [701.622, 735.708] / [997.771, 1070.098]
     - [345.568, 355.930] / [541.288, 607.350]
     - [63.190, 70.030] / [258.672, 279.831]
   * - 128, spare growth, 24 iterations
     - [35.369, 39.388] / [40.756, 45.218]
     - [17.910, 19.412] / [21.263, 22.546]
     - [7.287, 8.245] / [8.881, 10.530]

Retained compact medians side by side, 1/2/28 threads, ms:

.. list-table::
   :header-rows: 1

   * - Fixture
     - Phase 4B
     - Phase 4C
   * - 128, no growth
     - 18.979 / 9.855 / 4.264
     - 25.121 / 11.670 / 4.183
   * - 128, forced repacks
     - 96.016 / 55.011 / 34.429
     - 132.944 / 71.302 / 39.153
   * - 128, spare growth
     - 10.979 / 5.856 / 3.204
     - 16.590 / 8.518 / 3.456
   * - 1,024, multiple expansions
     - 765.796 / 447.152 / 269.469
     - 1039.683 / 544.431 / 265.557
   * - 128, spare growth, 24 iterations
     - 35.077 / 19.932 / 11.297
     - 43.458 / 21.900 / 9.758

Phase 3B and 4A medians remain in the preceding Phase 4B comparison.
Baseline movement is visible; comparing only the latest legacy ratios would
overstate some regressions. Nevertheless the one/two-thread adaptive
regressions are material and **Phase 4C performance acceptance is not met**.
The 28-thread spare-growth ratio is 1.542, versus Phase 4B's retained
1.1262, although compact time increased by 0.252 ms rather than that ratio
alone implying its entire increase.

Investigation isolated the new branch-count preflight. It executes the same
expensive crossing tests as topology construction, which still occurs later
at the original numerical point. Twenty-one warm standalone samples measured
the following medians (ms); serial is an isolated ``njit`` compilation of the
same function body, not a production-path change:

.. list-table::
   :header-rows: 1

   * - Elements
     - 1 thread parallel
     - 2 threads parallel
     - 28 threads parallel
     - Serial at 28-thread setting
   * - 128
     - 2.524
     - 1.295
     - 0.402
     - 2.604
   * - 1,024
     - 20.334
     - 10.367
     - 1.993
     - 20.639
   * - 10,000
     - 199.041
     - 103.914
     - 23.218
     - 198.389

This is real duplicate geometry work, not merely dispatcher overhead.
Switching to serial counting is not a remedy; no speculative serial threshold
was introduced. On the six-step 28-thread element-only instrumented spare
fixture, detection/construction-excluding-cache/cache/publication/
recalculation/commits/remaining work were
0.225/0.457/0.091/0.030/0.346/0.270/1.869 ms.
Phase 4B construction was 0.094 ms. Branch preflight is included in
construction on growth, and initial preflight in remaining work.
Structural branch-ID/sign/interval validation executes at public validation
and import, not as a new complete scan on ordinary logical growth.

The implementation candidate is therefore **stopped for review**, not
presented as an accepted performance migration. The next measured runtime
priority is avoiding duplicate branch crossing work while preserving exact
event order, delayed topology visibility and transactional publication.
Moving topology construction earlier or retaining a new topology staging
cache has not been silently introduced to pass the benchmark. Error-solver
migration and legacy removal should not precede resolving this new cost.
For further memory work, the retained padded quadrature is larger than the
remaining compact state (256 MB versus 27.8 MB at 10,000 elements).
That is a separate measured candidate, not work started in this phase.

Transaction accounting and final validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard-fixture branch counts fit their capacities, so no branch payload is
copied. Growth adds one exact-count dispatcher and three short-lived numeric
arrays (count, selected capacities and comparison mask) bounded by 17 bytes
per affected element. This pass is additional **affected-element** work, not
a complete finite/state scan. Branch replacement, when necessary, adds five
ownership payloads (starts, capacities and three event buffers); it copies
live event prefixes only, never unrelated work or coefficients.

Across each original six-iteration fixture, the maximum transaction-owned
payload accounting is:

.. list-table::
   :header-rows: 1

   * - Fixture
     - Payload arrays
     - Temporary bytes
     - Persistent-plus-transaction peak bytes
     - Complete validation passes
   * - 128, no growth
     - 0
     - 0
     - 420,000
     - 0
   * - 128, spare capacity
     - 14
     - 146,656
     - 440,576
     - 0
   * - 128, forced repacks
     - 39
     - 1,394,688
     - 2,242,208
     - 10
   * - 1,024, multiple expansions
     - 39
     - 10,216,480
     - 16,537,376
     - 10

These are maxima across batches, not cumulative allocated bytes; the
temporary and peak maxima need not occur at the same batch. They retain the
Phase 4B accounting boundaries. Initial activation is outside those
transaction counts: without a branch repack it adds an ID array plus the
three count/comparison arrays, bounded by 25*N bytes (3,200 / 25,600 /
250,000 at 128/1,024/10,000). Add this separately to persistent payload
when bounding first-activation owned-NumPy peak. At 100,000 the corresponding
2,500,000-byte bound is a formula, not a measured solve. Neither this bound
nor transaction accounting includes native Numba/NRT temporary allocation
or allocator-complete transient expressions.

Across five forced batches coefficient copy totals remain 972,800 /
7,782,400 bytes at 128 / 1,024 elements. Within-capacity coefficient copies,
all unrelated-work copies, and standard-fixture branch copies remain zero.
No new whole-state transaction copy or complete logical-growth scan was
introduced. The extra cost is geometry computation rather than transaction
copy bandwidth.

The adaptive branch-repack regression additionally changes a two-point empty
grid into an eight-point 1,600-event grid. It verifies private branch
ownership before publication, a failure immediately before publication with
unchanged complete state, successful capacity replacement, and complete
structural/finite validation afterward. Existing allocation-failure tests,
exact ordered-event checks, scratch isolation and numerical trajectories
remain intact. Recorded numerical absolute/relative differences are zero,
including the retained legacy cross-thread reduction comparisons; tolerances
are unchanged.

Final ``uv run pytest tests`` passed **324 tests** in 51.74 seconds.
Both project Ruff commands were attempted and could not start because Ruff
is unavailable. Pinned ``uvx ruff@0.16.2 check`` and
``uvx ruff@0.16.2 format --check`` passed on all eight changed Python files.
``git diff --check`` and ``git diff --check HEAD`` passed.

Changed files: ``hpc_work.py``, ``hpc_adaptive.py``, the new ``hpc_branch.py``,
``hpc_math_functions.py``, ``test_hpc_adaptive.py``, the new
``test_working_set.py`` and ``test_compact_branch.py``, the new
``benchmark_working_set.py``, and this documentation. Legacy/error storage,
persistence, quadrature, retained compatibility fields and numerical order/
damping/commit policies are unchanged. No later migration has begun.

Phase 4D: single-pass transactional branch construction
------------------------------------------------------

Phase 4C's memory/numerical implementation is retained, including all six
worker-owned scratch fields. Phase 4D changes only branch construction,
publication and associated validity metadata. Private pre-candidate topology
construction and transactional publication were explicitly approved for this
phase. Numerical crossing tests and their arithmetic are unchanged.

Duplicate-work map and measured baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Phase 4C path is:

1. Initial activation, or adaptive cache preparation, calls
   ``hpc_branch.prepare_branches``.
2. Its parallel ``count_events`` computes theta, exponentials and circle/line
   sample positions, then calls ``find_branch_cuts`` with one storage slot.
   The complete count is retained even after that slot fills.
3. Only overflow triggers ``growth_capacities``, ``_layout`` prefix sums,
   allocation of three event buffers and copying existing live prefixes.
   Within capacity there is no prefix sum, allocation or repack here.
4. Candidate Cauchy integration calls ``get_dpsi_corr``. When length is zero
   it maps the grid again and calls ``find_branch_cuts`` again, now filling
   events. Subsequent correction updates traverse events in order and add
   discharge times sign; they do not need geometry again.
5. Adaptive grid publication resets affected lengths; affected candidate
   recalculation repeats step 4. Genuine empty topologies are indistinguishable
   from invalid ones in Phase 4C and consequently repeat geometry on later
   candidates too.

Both crossing passes repeat interval/neighbor traversal, endpoint mapping,
the direct branch phase-jump test (including its 32-point arc test when
required), mirror eligibility/mirror endpoints and mirror crossing tests,
and well logarithm phase tests. They use geometry, fracture radius, ordered
fracture connectivity, element type and the integration grid. Coefficients,
candidate values, discharge and fracture constants are not topology inputs.
Outputs are an ordered multiset of interval index, discharge-element ID and
integer sign; a well/neighbor can appear repeatedly.

The order remains interval-major, connectivity-major, direct-before-mirror.
No sorting, deduplication, atomic append or arithmetic reordering is used.
The numerical correction's sequential summation order is unchanged.

Before editing, instrumenting the eight-element numerical bodies measured
**four find calls and 225 line-crossing tests in counting, then the same four
find calls and 225 tests in filling**. Well phase tests are additional, not
included in that 225. This body-level instrumentation is separate from native
timing; it does not pretend CProfile can observe every compiled inner call.
One hundred warm 128-element exact preparations at 28 threads took 0.043 s
and 701 visible Python calls: 100 preparation calls, 100 NumPy ``any`` calls,
100 reductions and dispatcher bookkeeping. Each invokes one count dispatcher;
the within-capacity case invokes no prefix-sum/repack operation.

Unmodified Phase 4C 10,000-element count / fill medians before editing, ms:
1 thread 199.581 / 198.975; 2 threads 103.456 / 102.738;
4 threads 53.478 / 55.232; 8 threads 32.151 / 33.673;
16 threads 32.773 / 33.728; 28 threads 20.156 / 20.690.
Counting plus filling is two geometry passes, not one slow dispatcher.

Direct construction, overflow and publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default adaptive import now binds its existing preparation entry point to
``prepare_direct``. The original ``count_events`` and ``prepare_branches``
remain callable reference/diagnostic implementations and are not called by
default compact execution. A regression replaces ``count_events`` with a
function that fails if invoked and runs initial construction and adaptation.

``construct_events`` takes compact numeric element/fracture arrays, affected
IDs, selected row indices, requested nint, starts/capacities and output arrays.
It passes a four-array ``BranchRecord`` to the unchanged crossing routine.
Each parallel iteration owns an exclusive staging segment, starts its local
count at zero, checks capacity before every write, and continues counting
after overflow without writing outside that segment. Counts and overflow-row
IDs are separate int64 arrays. Overflow is normal control flow, never a
parallel-loop exception.

The safety choice is **affected-only private staging**, with the same capacity
as each affected published segment. It avoids overwriting old events before
overflow is known. Staging is per preparation, not a second persistent copy
of all branch storage; unrelated live events are not copied on the normal
path. It is reused for every event in that construction and released after
publication; there is no retained workspace owner or scratch-ownership change.

When counts fit, the single construction pass is followed by compiled serial
validation of only new live prefixes. Validation rejects invalid lengths,
signs other than +/-1, IDs outside the element range and interval positions
outside ``[0, nint-1)``. Positions are intrinsically integer interval indices,
not floating coordinates. A deliberately floating NaN injected into private
position storage is also rejected; no position tolerance is used.
Bounded serial publication copies live prefixes and publishes lengths and
validity. Existing starts, capacities and all three event-buffer identities
remain unchanged.

Overflow recovery gathers only overflowing rows, grows their capacities
using the retained quantum-4 / 1.25 helper, preserves fitting staged results,
and reconstructs **only overflowing rows**. A batch replacement branch family
is allocated privately; unaffected live event prefixes are copied in original
order. There is one full branch-family repack when required, not one per
overflowing element. All reconstructed live prefixes are validated before
publication. The replacement shares unrelated solver buffers; no complete
solver-state clone is made.

All allocations, reconstruction, validation and the existing growth
``before_publication`` failure hook precede publication. Grid/cache publication
retains the original ``after_growth_publication`` observation boundary with
invalidated event lengths. Prepared topology is then published immediately
before candidate recalculation. Thus all original transition comparisons
remain intact rather than masking changed branch observations in tests.
There is no coefficient-commit or numerical barrier fusion.
Publication contains prevalidated, bounded compiled writes, with no callbacks,
allocation or expected input errors. Exclusive owner access and no mutation
of prepared arrays between preparation and publication remain required.
Concurrent mutation, asynchronous termination and memory corruption are not
recoverable transactional exceptions.

An explicit bool ``branch_valid`` costs one byte per element and distinguishes
valid empty topology from invalid topology. Grid changes reset both validity
and logical length at the same adaptive boundary as before. Discharge changes
update correction values only and never trigger reconstruction. Initial
activation also respects already imported valid topology. Geometry/connectivity
are fixed during solving, as in Phase 4C; external changes require invalidation
or a new owner. Unchanged connectivity alone does not establish validity if
the grid or geometry changed. Capacity changes affect storage layout, not
event order or topology semantics. Legacy/error validity behavior and storage
are unchanged.

Capacity-policy measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The heterogeneous seed and five subsequent forced-growth grids all have
counts ``[3,0,0,0,1,0,2,0]``. Across those 48 observations the mean is 0.75,
median 0, p90/p95/p99 are 3/3/3 and maximum is 3. This is measured on tiled
small DFNs, not evidence that a globally connected model has that distribution.
The retained adversarial test requires 1,600 ordered events from repeated
direct crossings and exercises real overflow/reconstruction.

The policy probe compares these alternatives per eight-element seed:

.. list-table::
   :header-rows: 1

   * - Policy
     - Initial/final event bytes
     - Utilization
     - Growth overflows / repacks / copy bytes
   * - Exact initial capacity
     - 144 / 144
     - 100%
     - 0 / 0 / 0
   * - Previous count plus quantum 4, event-producing types
     - 528 / 528
     - 27.3%
     - 0 / 0 / 0
   * - Previous count with approximately 25% slack, quantum-rounded
     - 288 / 288
     - 50%
     - 0 / 0 / 0
   * - Type-specific minimum 4 for types 1/4/5, zero otherwise
     - 384 / 384
     - 37.5%
     - 0 / 0 / 0

Exact and previous-count policies assume known counts at initialization; an
exact preliminary pass to obtain them would reintroduce the removed cost.
The selected policy therefore retains the simple type-specific four-slot
minimum with no preliminary count, reuses capacity thereafter, and expands
only on measured overflow. Quantum rounding is significant at these tiny
counts; the slack policy is not represented as exactly 25% utilization.
The benchmark now computes ``ceil(previous * 1.25)`` before quantum rounding.

The tiled 128/1,024/10,000 cases have zero natural overflows. Tiling the same
distribution at 100,000 (storage-only) does not increase overflow frequency.
This is not a forecast for other network geometries. A sparse forced-overflow
probe removes capacity from every 128th row, producing 1/8/79 overflowed
elements at those three runtime sizes: 0.78125% / 0.78125% / 0.79%.
Only those rows undergo a second crossing pass; all other affected rows
cross once. It requires one batch repack per sample.

Branch construction benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``uv run python tests\benchmark_branch.py --adaptive``. The original
adaptive fixture helper and timing boundaries are imported unchanged.
Branch probes alternate count, original topology fill, direct transaction
and sparse-overflow transaction in forward/reverse order; eight samples are
taken, first discarded. Direct totals include preparation, validation and
publication. Fill includes numerical correction updates; count plus fill is
the reference geometry path, not a separately measured summed median.
The fixtures remain heterogeneous tiled eight-element DFNs, not large
globally connected solves.

First 10,000-element measurement, milliseconds:

.. list-table::
   :header-rows: 1

   * - Threads
     - Exact count
     - Original fill alone
     - Direct total
     - Overflow detection pass
     - Repack
     - Overflow reconstruction
     - Overflow total
   * - 1
     - 195.464
     - 199.009
     - 197.616
     - 197.632
     - 1.254
     - 2.628
     - 202.364
   * - 2
     - 120.051
     - 121.076
     - 120.169
     - 119.302
     - 1.323
     - 1.362
     - 122.830
   * - 4
     - 54.335
     - 61.592
     - 70.995
     - 61.365
     - 1.358
     - 1.202
     - 64.633
   * - 8
     - 54.806
     - 58.327
     - 58.574
     - 56.989
     - 1.834
     - 0.762
     - 60.652
   * - 16
     - 32.865
     - 35.046
     - 37.256
     - 35.125
     - 1.935
     - 0.434
     - 38.494
   * - 28
     - 20.798
     - 21.320
     - 22.936
     - 20.543
     - 2.085
     - 0.386
     - 24.125

Stage medians need not sum to total medians. Detection above is the first
construction pass that discovers overflow, not a separate count kernel.
No-overflow direct geometry itself measured 196.797 / 119.232 / 70.017 /
57.442 / 36.228 / 21.715 ms at the six thread settings.

Direct total medians at smaller sizes, 1/2/4/8/16/28 threads:
128: 2.487 / 1.572 / 0.709 / 0.732 / 0.422 / 0.453 ms;
1,024: 20.265 / 12.262 / 5.585 / 6.097 / 3.724 / 3.272 ms.
Corresponding reference fill medians are
2.546 / 1.516 / 0.719 / 0.722 / 0.377 / 0.362 and
20.515 / 12.229 / 5.398 / 6.131 / 3.503 / 3.566 ms.
The direct path pays small fixed host/validation/publication costs, but
does not pay a second crossing pass.

A complete repeat gave the following 10,000-element count / fill / direct
medians and direct p10--p90 intervals, ms:

.. list-table::
   :header-rows: 1

   * - Threads
     - Count
     - Fill
     - Direct [p10, p90]
   * - 1
     - 228.612
     - 236.327
     - 229.439 [227.507, 231.858]
   * - 2
     - 120.260
     - 121.424
     - 123.479 [119.095, 160.691]
   * - 4
     - 70.033
     - 74.688
     - 90.793 [69.248, 102.657]
   * - 8
     - 56.434
     - 55.042
     - 62.117 [56.968, 64.064]
   * - 16
     - 37.408
     - 37.883
     - 37.589 [36.710, 38.617]
   * - 28
     - 25.738
     - 25.311
     - 23.496 [22.942, 32.117]

The unfavorable four/eight-thread ratios were investigated rather than
discarded. A further 32-sample rotated-order probe compared preallocated
direct kernel, exact count, original fill and complete direct preparation.
At four threads their medians were 65.477 / 63.501 / 69.822 / 66.255 ms;
direct-kernel p10--p90 was [52.801, 97.173], fill [57.086, 102.217].
At eight threads medians were 55.048 / 59.152 / 57.572 / 59.343 ms.
This does not show a sustained native direct-kernel regression above 10%.
It does show substantial shared-machine dispersion; no hardware bandwidth
or cache-miss explanation is claimed.

Adaptive acceptance and retained baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unchanged ``uv run python tests\benchmark_adaptive.py --profile`` reports
these compact medians, milliseconds; columns are 1/2/28 threads:

.. list-table::
   :header-rows: 1

   * - Fixture
     - Retained Phase 4B
     - Retained Phase 4C
     - Phase 4D
   * - 128, no growth
     - 18.979 / 9.855 / 4.264
     - 25.121 / 11.670 / 4.183
     - 12.176 / 6.379 / 2.384
   * - 128, spare growth
     - 10.979 / 5.856 / 3.204
     - 16.590 / 8.518 / 3.456
     - 8.933 / 4.600 / 2.435
   * - 128, forced repacks
     - 96.016 / 55.011 / 34.429
     - 132.944 / 71.302 / 39.153
     - 89.812 / 52.123 / 33.888
   * - 1,024, expansions
     - 765.796 / 447.152 / 269.469
     - 1039.683 / 544.431 / 265.557
     - 709.000 / 411.105 / 230.574
   * - 128, spare growth, 24 iterations
     - 35.077 / 19.932 / 11.297
     - 43.458 / 21.900 / 9.758
     - 22.147 / 12.101 / 7.246

The Phase 4D compact p10--p90 intervals, 1/2/28 threads, ms:
no-growth [12.044,12.361] / [6.250,6.467] / [2.300,2.718];
spare [8.760,10.363] / [4.575,4.652] / [2.252,3.071];
forced [89.076,91.307] / [50.854,52.590] / [32.976,35.171];
1,024 expansions [702.117,719.695] / [402.258,440.765] / [219.855,248.603].
Legacy medians in that run were respectively
19.812/9.660/3.297, 11.624/5.693/2.058,
86.689/42.536/11.777 and 695.967/351.244/67.485 ms.
High-thread forced-repack compact execution is still slower than legacy;
that predates this phase and is not reported as fixed.

For all six thread counts, Phase 4B sources at ``c865fa7`` were additionally
extracted into session-only storage and executed with the same unmodified
benchmark helper and a separate compiled cache. The repository worktree was
not switched or rewritten. Phase 4C was measured before editing. The following
cells show Phase 4B / Phase 4C / repeated Phase 4D compact medians, ms:

.. list-table::
   :header-rows: 1

   * - Threads
     - 128 no growth
     - 128 spare growth
     - 128 forced repacks
     - 1,024 expansions
   * - 1
     - 19.464 / 23.512 / 12.636
     - 11.424 / 14.483 / 8.544
     - 99.638 / 121.051 / 89.916
     - 794.749 / 943.315 / 706.526
   * - 2
     - 10.064 / 11.323 / 8.089
     - 6.073 / 7.810 / 4.715
     - 56.958 / 68.135 / 52.863
     - 444.040 / 531.162 / 421.603
   * - 4
     - 5.472 / 6.440 / 3.825
     - 3.501 / 4.264 / 4.609
     - 36.843 / 43.226 / 35.644
     - 295.218 / 327.264 / 291.534
   * - 8
     - 5.732 / 5.509 / 3.642
     - 3.577 / 4.131 / 2.958
     - 39.108 / 40.282 / 40.027
     - 303.231 / 243.746 / 298.800
   * - 16
     - 3.426 / 4.147 / 2.498
     - 2.645 / 3.402 / 2.390
     - 41.628 / 41.637 / 36.032
     - 258.476 / 289.749 / 238.197
   * - 28
     - 3.970 / 4.564 / 2.563
     - 2.813 / 3.434 / 3.080
     - 34.796 / 40.001 / 35.105
     - 243.103 / 275.338 / 234.015

Each run still uses seven alternating legacy/compact samples. Different
phase executions are not simultaneous interleaved phase comparisons; baseline
movement remains a limitation. At four threads, for example, the repeated
spare legacy median moved from 3.195 ms in Phase 4B to 5.413 ms in Phase 4D.
The first Phase 4D spare result there was 2.999 ms, not 4.609 ms.
The repeat's one-thread no-growth interval [12.352,65.939] also records an
outlier rather than discarding it. Eight-thread expansions were slower than
the before-edit Phase 4C snapshot, but 0.985x the freshly measured Phase 4B
baseline. Forced-growth Phase 4D/4B ratios were
0.902/0.928/0.967/1.023/0.866/1.009; expansions were
0.889/0.949/0.988/0.985/0.922/0.963.
No sustained forced/expansion regression above 10% versus Phase 4B was found.

The six-step element-only instrumented 28-thread profile measured detection
0.112 ms, construction excluding cache 0.418 ms, cache 0.090 ms,
cache/grid publication 0.022 ms, adaptive candidate plus branch publication
0.100 ms, commits 0.131 ms and remaining work 0.890 ms. The latter includes
initial branch preparation. CProfile's 200 post-growth steps took 0.028 s,
including 0.027 s in the adaptive entry point/compiled stages. Physical
coefficient/cache repacks, complete structural validation on those repacks,
dispatch/parallel launch costs and the unchanged numerical kernels remain.
No further Phase 4B transaction optimization or barrier fusion was performed.

Memory, copying and failure accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Event payload excludes starts, capacities, lengths and bool validity in the
following table. Event-capacity utilization is 37.5%; including metadata it
is 58.9%. Staging is allocated only for affected capacities.

.. list-table::
   :header-rows: 1

   * - Elements
     - Logical events bytes
     - Event-capacity bytes
     - Persistent branch metadata
     - Full affected staging bytes
     - Count/overflow metadata, no overflow
     - Branch peak payload bound
   * - 128
     - 2,304
     - 6,144
     - 3,200
     - 6,144
     - 1,024
     - 19,584
   * - 1,024
     - 18,432
     - 49,152
     - 25,600
     - 49,152
     - 8,192
     - 156,672
   * - 10,000
     - 180,000
     - 480,000
     - 250,000
     - 480,000
     - 80,000
     - 1,530,000
   * - 100,000 storage-only
     - 1,800,000
     - 4,800,000
     - 2,500,000
     - 4,800,000 (formula)
     - 800,000 (formula)
     - 15,300,000 (formula)

The peak bound includes persistent branch state, staging and compact stage
capacity/start/count/selected arrays (32 bytes per affected element).
Caller-owned IDs, transient validation/selection expressions and native
Numba/NRT theta/exponential/sample-position allocations are separate; this
is not allocator-complete RSS. No 100,000-element full solve or legacy work
allocation was attempted.

The sparse-overflow probe's additional int64 overflow-ID arrays cost 8/64/632
bytes at 128/1,024/10,000 elements. Private replacement branch families cost
8,192 / 65,536 / 640,000 bytes. Conservative preparation-payload bounds
(including both staging generations and the replacement) are
27,680 / 221,440 / 2,162,416 bytes. With old branch storage still alive,
branch peak bounds are 36,928 / 295,424 / 2,884,832 bytes.
Fitting-staging/unaffected live-prefix copies total
2,232 / 17,856 / 174,312 bytes; publication adds
2,304 / 18,432 / 180,000 bytes. Standard fixtures repack zero times and
copy only their newly constructed live events at publication.
There is no current/candidate/history or unrelated-work copy.
The existing accounting keys prefixed ``branch_count_temporary`` now describe
direct-preparation payload bounds for benchmark compatibility, not a hidden
preliminary count call.

Whole compact persistent payload is now
420,128 / 2,909,216 / 27,844,544 / 277,864,544 bytes at
128/1,024/10,000/100,000, exactly one byte per element above Phase 4C.
The separate retained padded quadrature and optional legacy payloads are
unchanged. No event family scales with MAX_ELEMENTS.

Failure regressions cover direct construction, injected NaN interval storage,
invalid event IDs, safe capacity overflow, replacement allocation,
overflow-only reconstruction and prepublication validation. Complete published
array identities and contents, lengths, caches, histories, convergence data
and validity are checked unchanged after failure. Successful publication
retains buffer identities within capacity and preserves unaffected event
prefixes during replacement. Read-only publication targets are rejected
before construction. Existing adaptive prepublication failure tests and
exact 1,600-event order tests remain in place.

All prior event/numerical transition comparisons are retained without
tolerance changes, including current/candidate/history state, adaptive requests,
caches/corrections, fracture constants, discharge/DFN discharge, heads,
boundary errors, convergence/iterations and legacy cross-thread deltas.
The unchanged adaptive benchmark records zero numerical differences.
Three-way integer event comparisons cover legacy padded, Phase 4C reference,
and direct construction at 1/2/28 threads. Discharge-only updates are tested
not to invoke topology preparation.

Phase 4D eliminates the exact preliminary crossing pass in sufficient
capacity, preserves the major memory reduction and recovers adaptive runtime.
The next priority should be a separately approved measured bottleneck:
padded quadrature for memory, or physical-repack validation/dispatch for
high-thread adaptive runtime. Error-solver migration and legacy removal have
not begun.

Validation and changed files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full ``uv run pytest tests`` passed **342 tests**, including eighteen new
direct-construction cases. Project ``uv run ruff check .`` and
``uv run ruff format --check .`` were attempted but could not start because
Ruff is unavailable in the environment. Pinned Ruff 0.16.2 lint and format
checks passed on all seven changed Python files. ``git diff --check`` passed.

Changed files are ``src\andfn\hpc\hpc_adaptive.py``,
``src\andfn\hpc\hpc_branch.py``, ``src\andfn\hpc\hpc_work.py``,
``src\andfn\hpc\hpc_math_functions.py``, the new
``tests\test_direct_branch.py`` and ``tests\benchmark_branch.py``,
``tests\benchmark_working_set.py`` (one-byte validity accounting), and
``docs\hpc.rst``. No dependencies, persistence, z_integral, legacy/error
storage, coefficient/integration capacity policy, worker-scratch layout,
adaptive eligibility/maximum behavior or damping/commit order changed.
Stop at Phase 4D; no subsequent migration is included.

Phase 5: runtime-sized main-solver quadrature
--------------------------------------------

Compact fixed-order/adaptive main solving now allocates a plain C-contiguous
``complex128`` array with shape ``(2, selected_elements, discharge_int)``.
Each side contains contiguous selected-element rows. Row ``j`` still means
``get_discharge_indices(elements)[j]``; no extra element-to-quadrature lookup
or copied discharge records are introduced. Two planes preserve the
intersection's two fracture coordinate systems. Single-sided elements leave
their second row zero. Mirrored discharge terms use the same coordinates and
the existing arithmetic, not an additional mirrored-coordinate allocation.

Allocation/consumer audit
~~~~~~~~~~~~~~~~~~~~~~~~

Here ``N`` is total elements, ``S`` ordered discharge elements, ``q`` requested
quadrature count, ``M`` storage MAX_NCOEF, and ``c_i`` integration capacity.
Bytes below are NumPy payload, excluding array headers and allocator overhead.
Runtime quadrature is uniform in q; integration-cache lengths vary with nint.

.. list-table::
   :header-rows: 1
   :widths: 18 23 30 29

   * - Allocation / site
     - Type, shape, physical/logical bytes
     - Producer -> active consumers
     - Lifetime, parallel use, diagnostics
   * - Main solve discharge ``z_int``
     - complex128 (2,S,q), 32q per selected row; formerly structured
       z0/z1[N,2M], 64M per total element
     - ``get_z_int_array`` -> ``fill_discharge_matrix``,
       ``build_head_matrix`` through ``solve_discharge_matrix``
     - Once per solve; reused across iterations; serial population,
       parallel matrix/RHS readers; not returned or retained by DFN
   * - Main solve ``z_int_error``
     - Former second structured z0/z1[N,2M], 64M per total element
     - ``get_z_int_array`` -> no active consumer
     - Removed only from compact main solve; legacy still allocates/populates
       it. No repository diagnostic/test reader; allocation guard regresses
       accidental reintroduction
   * - Boundary wrapper ``z_int``
     - complex128 (2,N,q), 32q per element; formerly structured z0/z1[N,2M]
     - ``compute_bnd_error`` / ``get_z_int_array`` -> ``get_bnd_error``
     - Once per boundary operation; positional all-element rows, including
       bounding circles; currently serial evaluation; only error output
       retained. Wells/impermeable rows may be unused
   * - Intersection / head-line ``z_array`` results
     - complex128 linspace(q+2), returned interior q-point view;
       16(q+2) backing bytes per construction
     - ``hpc_intersection.z_array`` / ``hpc_const_head_line.z_array`` ->
       corresponding z0/z1 row copy
     - Serial operation-local vectors; intersection constructs two sides
       sequentially; no retained duplicate. Endpoint/interior order unchanged
   * - Well ``z_array`` result and angles
     - float64 angles[q], complex128 z[q], logical q points
     - ``hpc_well.z_array`` -> side zero row
     - Operation-local linspace/exponential expression temporaries; serial
       preparation; no retained angle or mirrored cache
   * - Bounding-circle ``z_array`` angles, chi, z
     - float64 theta[q], complex128 chi[q] and z[q]
     - ``calc_thetas`` / ``hpc_bounding_circle.z_array`` -> boundary side zero
     - Operation-local; existing radius mapping unchanged; not a persistent
       integration grid
   * - Boundary evaluation vectors
     - complex128 omega[q] (two sides at intersections), float64 head[q];
       circle correction/psi[q] and omega[q]
     - ``get_bnd_error`` -> existing means/maxima/branch correction reduction
     - Element-local, serial; q uniform, no MAX_NCOEF padding; retained
       boundary output is float64[N,7], not quadrature
   * - Cauchy theta / correction / exp caches / ``z_integral``
     - float64 theta/correction, complex128 exp_array_m/exp_array_p/z_integral;
       64c_i allocated bytes per element, 64*nint logical bytes
     - Existing cache rebuilders -> type solves / Cauchy recurrence;
       ``z_integral`` -> diagnostic export (numeric read commented out)
     - Persistent across iterations; element-owned integration capacity,
       parallel element access; untouched. No ``z_array_m`` or
       ``z_array_p`` allocation exists in current source
   * - Cauchy temporary coordinates and worker buffers
     - Scalar mapped z/chi; existing worker phi/psi/dpsi and integral/coef0/coef1
     - Existing element kernels / ``hpc_math_functions`` -> Cauchy reduction
     - Unchanged worker ownership and capacity, no discharge quadrature
       dependency; not converted to CSR or thread-local quadrature
   * - Legacy/error quadrature and diagnostics
     - Independent padded ``dtype_z_arrays`` definitions; complex128
       z0/z1[N,2M], 64M bytes per element per array
     - Legacy main solve, ``hpc_error_solve.solve_error`` and boundary wrapper;
       legacy ``get_max_min_phi`` retains positional structured interface
     - Legacy/error behavior retained, including unused error-solver
       ``z_int_error``. Plotting/head sunflower/runtime-point arrays are not
       main-solver quadrature and are unchanged

The boundary wrapper retains its original all-element padded ``dtype_work``.
It is a separate diagnostic allocation, not compact adaptive worker scratch,
and migrating its branch/scratch ownership is outside Phase 5. The public DFN
legacy boundary wrapper still selects legacy execution unless flat
coefficients are supplied. Fixed-order compatibility execution without an
adaptive owner also retains its original padded main ``dtype_work``; the
compact category accounting below describes adaptive ownership, not that
compatibility allocation. Error-solver arrays have not been migrated or
removed. Persistence and all numerical/capacity policies are unchanged.

Ordinary Numba overload lowering chooses dense versus legacy row views.
Forced inlining is deliberately avoided: the legacy no-index ``prange``
specialization fails Numba parfor rewriting with a loop-index overwrite when
that adapter is forcibly inlined. Row views do not copy point data.

``allocate_quadrature`` validates integral, nonnegative rows and positive
counts, rejects booleans, and checks addressable dimensions/payload using
Python integer arithmetic before allocation. Public dense matrix validation
also checks dimensions, dtype and C-contiguity. The existing producer does
not reject nonfinite geometry; matching propagation is tested rather than
introducing a new numerical contract. Allocation and coordinate tests cover
zero/one/many rows, q=1/8/1024, both sides, duplicate/reordered selected IDs,
reuse, changed shapes and fixed-q independence from MAX_NCOEF=200/500.

Phase 5 payload and lifetime accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repeated heterogeneous fixture selects exactly half its elements.
At q=8, the payload is explicitly ``S * 8 * 16 * 2``: **256 bytes per
selected element**, or 128 bytes per total fixture element. The two original
main-driver grids cost 25,600 bytes per total element at configured M=200.
Decimal MB is used, not MiB.

.. list-table::
   :header-rows: 1

   * - Total elements / selected
     - Original two padded grids
     - Compact discharge grid
     - Separate compact boundary grid
   * - 128 / 64
     - 3,276,800
     - 16,384
     - 32,768
   * - 1,024 / 512
     - 26,214,400
     - 131,072
     - 262,144
   * - 10,000 / 5,000
     - 256,000,000
     - 1,280,000
     - 2,560,000
   * - 100,000 / 50,000
     - 2,560,000,000
     - 12,800,000
     - 25,600,000

The 10,000-element comparison is **256 MB -> 1.28 MB**, a 200-fold payload
reduction (99.5%). Even the separate all-element boundary grid is 2.56 MB.
The 100,000 case is storage-only: the dense payload was allocated and its
actual nbytes checked, not a 100,000-element solve.

Owned discharge quadrature peak is one dense grid, reused for matrix
construction and every RHS. There is no second compact error grid and no
size-discovery construction pass. Coordinate producers retain their existing
short native vectors; a conservative source-expression payload allowance is
``80*(q+2)`` bytes (800 bytes at q=8), independent of N/S. Thus the
10,000-element discharge coordinate payload upper estimate is 1,280,800
bytes, and a separate boundary operation 2,560,800 bytes. These are NumPy/
native coordinate payload estimates, **not measured process RSS or a bound
on all solver temporaries**. Boundary omega/head/correction reduction vectors,
Numba runtime bookkeeping, sparse conversion and LU storage are separate.

Both grids retain **zero bytes after their operation returns**. A diagnostic
boundary call inside a solve callback can coexist with the solve grid:
3,840,000 owned coordinate bytes at 10,000 elements, plus the active
producer's short temporaries. It also incurs the unchanged boundary
``dtype_work`` payload of 704,330,000 bytes. Such diagnostic work is not
silently counted as compact worker scratch.

The following allocated-byte breakdown reuses the Phase 4D heterogeneous
memory fixture (including its seven scratch groups); those categories were
not changed in Phase 5. Scratch group count is an independent setting, not a
claim that every benchmark uses seven threads. ``Sparse inputs`` includes
triplets, counts/offsets, existing element-to-discharge positions, selected
indices and connectivity. It is not a CSC/LU or process-peak measurement.

.. list-table::
   :header-rows: 1

   * - Category
     - 128
     - 1,024
     - 10,000
     - 100,000
   * - Fixed element metadata/layout (225N+32)
     - 28,832
     - 230,432
     - 2,250,032
     - 22,500,032
   * - Coefficient state/history (736/element)
     - 94,208
     - 753,664
     - 7,360,000
     - 73,600,000
   * - Integration caches (1,536/element)
     - 196,608
     - 1,572,864
     - 15,360,000
     - 153,600,000
   * - Worker scratch (fixed for these orders/groups)
     - 64,512
     - 64,512
     - 64,512
     - 64,512
   * - Branch events/layout (73/element)
     - 9,344
     - 74,752
     - 730,000
     - 7,300,000
   * - Compatibility state (208/element)
     - 26,624
     - 212,992
     - 2,080,000
     - 20,800,000
   * - Active discharge quadrature (128/total element here)
     - 16,384
     - 131,072
     - 1,280,000
     - 12,800,000
   * - Sparse inputs (104N+8, temporary construction envelope)
     - 13,320
     - 106,504
     - 1,040,008
     - 10,400,008
   * - Compact core plus quadrature/inputs
     - 449,832
     - 3,146,792
     - 30,164,552
     - 301,064,552
   * - Shared padded fracture input (1,636/total element)
     - 209,408
     - 1,675,264
     - 16,360,000
     - 163,600,000
   * - Three discharge/RHS/history vectors (18/element)
     - 2,304
     - 18,432
     - 180,000
     - 1,800,000
   * - Tracked compact execution envelope
     - 661,544
     - 4,840,488
     - 46,704,552
     - 466,464,552
   * - Above plus retained legacy element records
     - 2,324,520
     - 18,144,296
     - 176,624,552
     - 1,765,664,552
   * - Above plus optional retained legacy work
     - 11,339,944
     - 90,267,688
     - 880,954,552
     - 8,808,964,552
   * - Original padded elements/work/grids plus same inputs
     - 14,180,232
     - 113,441,800
     - 1,107,830,008
     - 11,078,300,008

The compact core alone does not include original fracture records: they remain
a shared padded input to existing interfaces, explicitly added above instead
of being hidden in a "compact-only" total. At 10,000 elements, retained legacy
elements add 129.92 MB, and optional retained legacy work adds 704.33 MB.
Compact persistent categories total 27,844,544 bytes before quadrature/inputs.
Largest compact persistent categories remain integration caches (15.36 MB)
and coefficients/history (7.36 MB); quadrature is no longer dominant.
The original padded fracture input's elements field also remains unchanged.
Matrix outputs/LU, validation expression temporaries, Python objects, compiled
code and allocator overhead are excluded from both tracked architecture
comparisons. These are category envelopes, not falsely simultaneous
all-allocation or OS high-water measurements.

Phase 5 numerical/performance results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests\benchmark_quadrature.py --adaptive`` compares padded and dense
quadrature with identical compact coefficient metadata, arithmetic and orders.
Preparation/matrix/RHS samples alternate layout order; compilation and the
first sample are excluded. It measures 128/1,024/10,000 elements at
1/2/4/8/16/28 threads. Preparation itself is serial; its variation across
thread settings reflects runtime conditions, not parallel speedup.

The 10,000-element warm median milliseconds below are **padded / dense**.
Allocation/preparation's padded control contains only the actively used grid;
it conservatively excludes the additional obsolete error-grid work.

.. list-table::
   :header-rows: 1

   * - Threads
     - Preparation, reused grid
     - Allocate + prepare
     - Discharge matrix
     - Head RHS
   * - 1
     - 0.691 / 0.679
     - 6.152 / 0.863
     - 5.134 / 4.484
     - 16.413 / 16.237
   * - 2
     - 0.679 / 0.677
     - 6.261 / 0.854
     - 2.655 / 2.389
     - 7.865 / 7.555
   * - 4
     - 0.707 / 0.693
     - 6.490 / 0.814
     - 1.846 / 1.679
     - 3.893 / 4.043
   * - 8
     - 0.692 / 0.665
     - 6.849 / 0.883
     - 1.704 / 1.688
     - 3.900 / 3.987
   * - 16
     - 0.672 / 0.659
     - 7.886 / 1.375
     - 1.546 / 1.446
     - 2.130 / 2.117
   * - 28 (normal maximum)
     - 1.764 / 1.672
     - 13.360 / 1.957
     - 1.639 / 1.529
     - 2.525 / 2.563

At 1,024 elements, allocation+preparation was 0.645/0.073 ms at one thread;
matrix 0.477/0.470 ms and RHS 1.572/1.570 ms. At 128 it was
0.092/0.010 ms; no implementation was tuned for this small fixture.
The first 1,024-element eight-thread matrix sample reported a 15.0% slowdown;
three separate alternating reruns yielded dense/padded ratios
1.039, 1.004 and 0.921, with overlapping p10/p90 intervals. It was not a
sustained >10% large-model regression.

Unchanged six-iteration adaptive cases, with only compact quadrature layout
switched outside the existing timed loop, gave:

.. list-table::
   :header-rows: 1

   * - Threads
     - 1,024-element expansions, padded / dense ms
     - 128 no-growth dense ms
     - 128 spare-capacity dense ms
     - 128 forced-growth dense ms
   * - 1
     - 741.596 / 754.682
     - 13.396
     - 9.290
     - 103.005
   * - 2
     - 430.534 / 452.342
     - 6.786
     - 5.384
     - 55.265
   * - 4
     - 349.458 / 335.205
     - 4.252
     - 3.989
     - 45.882
   * - 8
     - 344.851 / 311.549
     - 4.147
     - 3.504
     - 46.583
   * - 16
     - 290.209 / 299.300
     - 2.891
     - 2.813
     - 39.113
   * - 28
     - 371.276 / 322.506
     - 4.386
     - 4.757
     - 63.259

All six 1,024-element dense/padded adaptive ratios are within 0.869--1.051.
Small spare-capacity cases fluctuated by more than 10% at 2/4 threads;
they were not optimized. Independent unchanged padded controls also slowed
versus the retained Phase 4D session baseline: the subsequent 1,024-element
control measured 896.985/559.644/401.835 ms at 1/2/28 threads. Therefore
cross-session absolute timings do not establish a quadrature-caused
regression or justify adaptive/branch micro-optimization.

The unchanged ``benchmark_working_set.py --solver`` Cauchy/candidate control
was rerun at all six thread settings. Compact 10,000-element medians were
293.518/218.039/126.835/86.355/45.923/44.226 ms; corresponding legacy controls
were 291.014/234.949/122.561/88.493/45.818/43.064 ms. These kernels do not
consume migrated quadrature. Shared-machine timing dispersion is substantial;
no cache/bandwidth hardware counters were measured and no causal bandwidth
claim is made. The measured layout comparison shows material allocation
savings and no sustained >10% representative large-kernel regression.

Every benchmark coordinate/order, matrix data/indices/indptr and head RHS
comparison is exactly equal. All 48 adaptive layout/thread/case final
coefficient comparisons have absolute/relative differences zero. The unchanged
adaptive control's 12 trajectories match, and all 132 recorded numerical
field comparisons have zero absolute/relative difference. Full adaptive
history tests compare dense and padded quadrature exactly at 1/2/28 threads,
including candidates, coefficients, discharges, fracture constants, heads,
boundary errors, caches and branch events. Existing legacy/compact and
convergence tests remain unchanged; no tolerance was increased.

Validation and next-phase boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``uv run pytest tests`` passed **381 tests** (39 new quadrature cases).
``uv run ruff check .`` and ``uv run ruff format --check .`` were attempted
and could not start because project Ruff is unavailable. Pinned Ruff 0.16.2
lint/format checks passed on all five changed Python files;
``git diff --check`` passed.

Changed files: ``src\andfn\hpc\hpc_solve.py``, new
``src\andfn\hpc\hpc_quadrature.py``, ``tests\benchmark_adaptive.py``, new
``tests\benchmark_quadrature.py`` and ``tests\test_quadrature.py``, and this
documentation. No error-solver, persistence, numerical algorithm, coefficient
or integration/branch capacity policy, worker ownership, damping, publication,
commit or event order change is included.

For a solve-only resident-memory goal, legacy-storage removal is the next
large opportunity: 129.92 MB of source element records plus 16.36 MB of
shared padded fracture inputs at 10,000 elements, versus 1.28 MB of compact
quadrature. For end-to-end peak memory including boundary diagnostics, the
largest measured remaining main-path allocation is instead the **704.33 MB
boundary ``dtype_work``**. Prioritize a separately approved migration of that
diagnostic work allocation (option 3), without changing boundary arithmetic
or migrating the error solver; then legacy-storage removal. Error-solver
migration becomes first priority only for a workload actually running it.
This is a recommendation, not an implementation: **stop after Phase 5**.

Normal compact workflow and boundary work (Phase 6)
--------------------------------------------------

The public opt-in is ``dfn.solve(compact=True)``. The default remains the
legacy solver. For a construction-time connectivity check use
``dfn.check_connectivity(compact=True)`` as well. Direct consolidation reads
logical object coefficients and adjacency into the existing adaptive owner,
fixed fracture metadata and ordered CSR, without first constructing legacy
element/fracture records. It preserves object/element/adjacency order.
The existing adaptive solver performs matrix construction, factorization,
discharge, Cauchy integration, growth and commits unchanged.

Head, derivative, velocity, total discharge, head samples, head/flow plots,
flow-fracture selection and post-solve boundary diagnostics use the active
compact state. Explicit ``unconsolidate_dfn(hpc=True)`` copies logical prefixes
back to objects; particle tracking already uses this boundary. Explicit
legacy consolidation remains available and first refreshes logical objects.
It is a compatibility export, not part of compact execution. Persistence,
the separate error solver and error plots are not migrated: prepare/use the
legacy workflow for those operations. Existing compatibility snapshots are
not automatically kept synchronized with later compact solves; explicitly
consolidate again before exporting an existing snapshot.

``FractureArrays``/``FractureRecord`` are numeric record views over those
same metadata/CSR arrays, not another storage owner. Nopython functions
receive array tuples, never the Python adaptive owner. Fracture record
``elements`` is an adjacency slice, and ``nelements`` is its actual length.
Compiled consumers use ``.size``: Numba's tuple ``len`` means tuple arity,
not the number of fractures. Regression cases cover one, two, four and six
fractures. This also permits reuse of the original scalar numerical kernels
without duplicating formulas or exporting padded coefficients.

Boundary representation and lifetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prepare_boundary_work`` uses the existing ``WorkArrays`` representation.
Fixed metadata, coefficient layouts, coefficient state and integration caches
are shared, not copied. Worker-owned scratch is reused at a post-solve/barrier
boundary. If the requested diagnostic grid exceeds the actual integration
scratch width, only the real ``phi/psi/dpsi`` worker buffers grow for this
operation, to ``workers * max(existing_width, requested_count)``. Coefficient
scratch is not resized. This does not make concurrent/reentrant solve and
diagnostic access safe.

The diagnostic topology cannot share the solver's events: the diagnostic
grid may have a different count and coordinates. It gets private
``branch_start/branch_capacity/len_discharge_element/branch_valid`` arrays
and three int64 event buffers. Bounding-circle rows initially use the
existing four-event quantum; other rows have zero event capacity.
Phase 4D's direct builder consumes the already-produced boundary coordinates,
writes bounded segments, detects overflow and batch-expands only when
necessary. There is no preliminary crossing-count pass, no coordinate
reconstruction and no changed capacity policy. In particular, diagnostics
must not substitute the solver's circle mapping for the original diagnostic
coordinate mapping. Event order and corrections remain exactly equal.

Quadrature stays C-contiguous complex128 ``(2, N, q)``; row views do not copy.
``get_bnd_error`` reuses the same numerical formula, with ``work_record`` and
a prepared-topology flag rather than a second independent implementation.
Its element evaluation remains serial, as before; coordinate/topology
preparation can run in parallel. Existing impermeable-object diagnostic
rows remain zero; Phase 6 does not invent new error formulas.
Operation-private work and quadrature are released on return. Only the
float64 ``(N, 7)`` output survives; persistent solver caches/events are
unchanged. Ordinary q8 calls reuse all worker scratch.

Normal-workflow allocation audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here N=10,000, F=2,500, legacy M=200, configured E=800, q=8,
and selected discharge rows S=5,000. MB means decimal bytes.
Logical ncoef/nint and event distributions use the established heterogeneous
eight-element repeated fixture. Fixed-size geometry/endpoints are not ragged.

.. list-table::
   :header-rows: 1
   :widths: 23 25 25 27

   * - Allocation / site
     - Physical legacy shape, dtype, bytes
     - Logical requirement / lifetime
     - Normal compact replacement
   * - ``element_dtype_hpc``; legacy ``DFN.consolidate_dfn``
     - N records; 129,920,000 total, including two complex128[N,M] and two float64[N,2M] fields
     - Actual per-element ncoef/nint; persistent
     - Fixed metadata plus existing capacity coefficient/cache buffers; direct object consolidation
   * - Consolidation gathering arrays in ``dfn.py``
     - Two complex128[N,M] plus two float64[N,2M]; 128,000,000 temporary
     - Logical prefixes; construction only
     - No padded gather/bridge in ``hpc_compact.consolidate``
   * - ``fracture_dtype_hpc`` and consolidation IDs
     - int64[F,E], 16,000,000 padding; 16,360,000 records, plus 16,000,000 temporary IDs
     - Actual adjacency; persistent, gather temporary
     - Fixed fracture metadata plus ordered CSR: 450,008 total
   * - ``hpc_solve.dtype_work``; solve and boundary wrapper
     - N padded records; 704,330,000 for each allocation
     - Actual coefficients, integration, events, or worker temporary requirements
     - Existing compact solver work; private boundary work 490,000 with shared state/scratch
   * - ``hpc_solve.dtype_z_arrays``; solve
     - Two N-record complex128 padded buffers; 256,000,000
     - S rows * q * two sides; solve-local
     - Phase 5 complex128[2,S,q], 1,280,000; unused compact ``z_int_error`` remains absent
   * - Boundary quadrature; ``prepare_boundary_work``
     - complex128[N,2M] twice; 128,000,000
     - N rows * q * two sides; one evaluation
     - complex128[2,N,q], 2,560,000; never converted back to records
   * - Cauchy ``thetas/dpsi_corr/exp_array_m/exp_array_p/z_integral``
     - Previously width 2M per element
     - Actual nint; persistent between iterations
     - Existing capacity caches, unchanged: 15,360,000; ``z_integral`` retained
   * - Scalar/vector potentials and head/flow sampling
     - Actual requested point/grid sizes
     - Operation-local numerical outputs and expression temporaries
     - Same runtime numerical sizes; flat coefficient views and CSR; no global maximum dimension
   * - CSR matrix input, RHS/discharges/history
     - Actual nnz, S and F
     - Matrix construction / solve
     - Existing design unchanged; no storage-max padding

All normal compact allocation artifacts above are bypassed or replaced;
the definitions and compatibility allocation sites themselves remain.
Construction still uses bounded-size geometry (two endpoints, 3D vectors,
etc.); these are intrinsically fixed, not unused capacity. No additional
global-maximum-shaped normal compact allocation was found.

Maximum-constant occurrence classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repository-wide searches covered ``MAX_NCOEF``, ``MAX_ELEMENTS``,
``MAX_DISCHARGE_INT``, other ``MAX_*`` names, legacy dtype consumers and
literal maximum-shaped allocation patterns. Grouped source classification:

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Location / occurrence
     - Category
     - Disposition
   * - ``element.py`` definitions/config reads and padded dtype fields; per-object converters
     - C, D
     - Retained legacy compatibility/persistence schema; direct compact consolidation never calls them
   * - ``dfn.py`` imported maxima and legacy consolidation gather/output dimensions
     - C, D
     - Retained explicit legacy/export paths; not reached by the opted-in normal solve or diagnostics
   * - ``hpc_solve.py`` imported maxima and ``dtype_work/dtype_z_arrays`` definitions; legacy allocations
     - C
     - Retained default solver and boundary reference path; compact branches use runtime work/quadrature
   * - ``hpc_error_solve.py`` imported maxima, its independent work/quadrature definitions and consumers
     - B
     - Entire separate error execution remains padded and unchanged
   * - ``constants.py`` scalar dtype/settings, logging and ``hpc_solve`` scalar maximum
     - F, C
     - Adaptive numerical stopping controls and retained compatibility validation, not allocation dimensions
   * - ``hpc_adaptive._MAX_REQUEST_INCREMENT`` and dtype-derived length guards in ``storage.py``
     - F, C
     - Existing overflow-safety/physical compatibility checks, unchanged; they reject invalid requests but do not allocate maximum-sized compact arrays
   * - ``MAX_DISCHARGE_INT`` definition/config setting
     - G
     - Dormant storage setting, no active dimension use found; retained, not renamed/removed
   * - Examples, tests, benchmarks, configuration/user/HPC documentation and local user scripts
     - E
     - Controls/examples/reference accounting; no execution migration or edits to local user scripts
   * - ``MAX_ITERATIONS``, ``MAX_ERROR`` and related solver settings
     - F
     - Numerical iteration/accuracy controls, not storage dimensions

There are no remaining category-A maximum-sized allocation artifacts in the
opted-in workflow. The new compact consolidation/connectivity modules and
the existing compact work/quadrature constructors do not import storage
maximum constants (checked by AST, including imports and attributes).
Shared modules still intentionally contain the legacy definitions above.
``storage.py``'s coefficient-only ``fixed_order_element_dtype`` retains
padded integration fields for that older compatibility entry point, not the
public adaptive compact workflow.

The scalar adaptive maximum is still a real *algorithmic* stopping control
even though it shares the legacy name ``MAX_NCOEF``. It retains the approved
old-order check/overshoot semantics. Legacy dtype-derived physical limits
are compatibility guards, not claimed to be mathematical requirements;
relaxing them is outside this phase. ``MAX_ELEMENTS`` scalar validation also
remains for legacy compatibility, but does not size compact adjacency/events.

Storage-independence regressions vary runtime legacy maxima through
200/500/1,000 coefficients and 500/5,000 elements while holding logical state,
numerical controls and q fixed. Persistent/diagnostic/CSR payload and results
remain exactly equal. Import-time legacy dtypes do not change under
monkeypatching, so this is paired with the no-maxima constructor AST check
and direct allocation guards. No global maximum is renamed or removed.

Measured boundary memory
~~~~~~~~~~~~~~~~~~~~~~~~

No padded boundary work is allocated in the compact branch. Incremental fixed
metadata, coefficient references and integration/cache payload are all zero:
the existing arrays are referenced directly. Incremental q8 worker scratch and
compatibility allocation are also zero for this fixture. Private work is
49N bytes: 24N event data plus 25N layout/count/validity metadata.
Boundary quadrature is exactly ``N * 8 * 16 * 2 = 256N`` bytes; output is
56N bytes. Thus owned evaluation payload is 361N bytes.

.. list-table::
   :header-rows: 1

   * - Elements
     - Legacy work bytes
     - Compact private work
     - Quadrature
     - Output
     - Owned total
     - Peak envelope
   * - 128
     - 9,015,424
     - 6,272
     - 32,768
     - 7,168
     - 46,208
     - 51,584
   * - 1,024
     - 72,123,392
     - 50,176
     - 262,144
     - 57,344
     - 369,664
     - 412,672
   * - 10,000
     - 704,330,000
     - 490,000
     - 2,560,000
     - 560,000
     - 3,610,000
     - 4,030,000
   * - 100,000 (storage-only)
     - 7,043,300,000
     - 4,900,000
     - 25,600,000
     - 5,600,000
     - 36,100,000
     - 40,300,000

The 10,000-element work reduction is **704.33 MB to 0.49 MB (99.93%)**.
Including old quadrature and the same output, legacy boundary payload was
832.89 MB versus compact 3.61 MB. Preparation has 320,000 bytes of staged
events/metadata plus 100,000 bytes of IDs/lengths in this case.
The reported 4.03 MB peak is a conservative NumPy payload envelope: it
includes output even though output allocation follows preparation. It is
not a measured simultaneous allocator/RSS peak. Native per-row expression/
coordinate temporaries and allocator/runtime overhead are excluded.
After evaluation only the 0.56 MB returned output remains, not the 3.61 MB.
Larger q can require private operation-local worker real scratch as described
above; no such allocation scales with unused legacy maxima.

Boundary performance and thread scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests\benchmark_boundary.py`` alternates legacy and compact full boundary
evaluations (allocation, coordinate production, topology and numerical error
evaluation), discards the warm-up and reports seven-sample medians. References
use padded source records outside timing; compact evaluation reads flat
coefficients and numeric CSR fracture records. Every output is exactly equal.
Reported values are legacy / compact milliseconds, with compact/legacy ratio.

.. list-table::
   :header-rows: 1

   * - Threads
     - 128 elements
     - 1,024 elements
     - 10,000 elements
   * - 1
     - 1.104 / 0.949 (0.860)
     - 9.181 / 7.040 (0.767)
     - 94.371 / 71.639 (0.759)
   * - 2
     - 1.420 / 1.057 (0.744)
     - 11.074 / 6.905 (0.624)
     - 119.327 / 71.970 (0.603)
   * - 4
     - 1.904 / 1.243 (0.653)
     - 14.123 / 8.238 (0.583)
     - 143.193 / 91.678 (0.640)
   * - 8
     - 2.271 / 1.590 (0.700)
     - 21.733 / 12.627 (0.581)
     - 209.964 / 120.756 (0.575)
   * - 16
     - 4.061 / 2.884 (0.710)
     - 27.786 / 18.355 (0.661)
     - 297.744 / 193.589 (0.650)
   * - 28 (normal maximum)
     - 5.212 / 4.721 (0.906)
     - 39.773 / 30.601 (0.769)
     - 413.064 / 279.962 (0.678)

No sustained >10% large-model slowdown was observed: compact is faster at
every measured setting. This diagnostic does **not** show useful parallel
scaling: at 10,000 elements, 28-thread legacy/compact times are 4.38/3.91
times their one-thread times. The original serial outer diagnostic loop and
shared scalar numerical dispatch remain; this phase does not redesign their
parallelism or claim bandwidth/cache-counter improvements.

Complete normal-workflow accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below is unique NumPy payload, not RSS. Capacities and 28 worker
slots are held fixed at the actual fixture requirements, with 100,000
calculated affinely from the 10,000-element layout, not executed.
No CSC/LU internals, source Python model objects, compiler/runtime allocations,
native expression temporaries or graphics meshes are included.

.. list-table::
   :header-rows: 1

   * - Category
     - 10,000 bytes
     - 100,000 bytes
   * - Element fixed metadata, layouts and scalar work (249N + 32)
     - 2,490,032
     - 24,900,032
   * - Current/candidate/old coefficient storage (552N)
     - 5,520,000
     - 55,200,000
   * - Retained compact compatibility work coefficients (368N)
     - 3,680,000
     - 36,800,000
   * - Integration/cache (1,536N)
     - 15,360,000
     - 153,600,000
   * - Worker scratch (28 slots; not proportional to N)
     - 64,512
     - 64,512
   * - Solver branch data/layout (73N)
     - 730,000
     - 7,300,000
   * - Fracture metadata and CSR (45N + 8)
     - 450,008
     - 4,500,008
   * - Compact persistent total
     - 28,294,552
     - 282,364,552
   * - Solve-local discharge quadrature (128N)
     - 1,280,000
     - 12,800,000
   * - Sparse matrix inputs / ordered selected IDs, excluding already-counted CSR (93N)
     - 930,000
     - 9,300,000
   * - RHS, discharges and history vectors (18N)
     - 180,000
     - 1,800,000
   * - Solve tracked total (excludes adaptive repack peaks and CSC/LU)
     - 30,684,552
     - 306,264,552
   * - Boundary quadrature (256N)
     - 2,560,000
     - 25,600,000
   * - Boundary work (49N)
     - 490,000
     - 4,900,000
   * - Boundary output (56N)
     - 560,000
     - 5,600,000
   * - Compact persistent + owned boundary operation
     - 31,904,552
     - 318,464,552
   * - Compact persistent + boundary peak envelope
     - 32,324,552
     - 322,664,552
   * - Optional retained padded elements + fractures (14,628N)
     - 146,280,000
     - 1,462,800,000
   * - Compact + retained padded inputs + boundary owned operation
     - 178,184,552
     - 1,781,264,552
   * - Additionally retained legacy solver work, only if legacy execution ran
     - 704,330,000
     - 7,043,300,000

Solve-local and post-solve boundary allocations are separate lifetimes, not
added as though simultaneously resident. The optional compatibility totals
show coexistence, not a required compact import step. The new direct public
workflow allocates **none** of those padded input/work records. For comparison,
the original padded main solve tracked approximately 1,106.61 MB at 10,000
elements before sparse/CSC/LU internals (129.92 MB elements + 16.36 MB
fractures + 704.33 MB work + 256 MB quadrature). Retaining that solver work
and running a legacy boundary diagnostic costs approximately 1,683.50 MB,
versus the compact owned-boundary total of 31.90 MB, on the same accounting
basis. This is payload accounting, not a process-peak claim.

Validation and remaining phase boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``uv run pytest tests`` passed **398 tests**: all 381 previous tests plus
17 compact-workflow regressions. A fresh-cache full run before the last four
extent/scratch cases also passed (394 tests). Coverage includes heterogeneous
ncoef/nint, exact CSR/padded adaptive state and potential/derivative comparisons
at 1/2/28 threads, actual public model solves, boundary counts 1/8/100/401,
all element types, repeated evaluation, forced private event overflow,
scratch sharing/resizing, empty boundary evaluation, direct-allocation guards,
and storage-maximum independence. The existing quadrature dimension,
overflow, ordering and dense/padded trajectory tests remain unchanged.
No tolerance was increased. All numerical comparisons in the new tests and
the 18 boundary benchmark cases are exactly equal (zero differences).

``uv run ruff check .`` and ``uv run ruff format --check .`` were attempted;
both could not start because project Ruff is unavailable. Pinned Ruff 0.16.2
lint/format checks passed on all ten changed Python files. Staged and unstaged
``git diff --check`` checks passed.

Changed files: ``src\andfn\dfn.py``, ``src\andfn\storage.py``,
``src\andfn\hpc\hpc_branch.py``, new ``hpc_compact.py`` and
``hpc_connectivity.py`` in that directory, ``hpc_fracture.py``,
``hpc_solve.py``, ``hpc_work.py``, new ``tests\test_compact_workflow.py``,
new ``tests\benchmark_boundary.py``, and ``docs\hpc.rst``.
The benchmark script reproduces the tables, including storage-only 100,000
projection; it adds no dependency.

The largest compact persistent family is now actual integration/cache data,
then logical coefficient/compatibility state. No measured main allocation
justifies another layout redesign here. Phase 7 should prioritize explicit
legacy-storage lifecycle/removal to eliminate optional 146.28 MB source
coexistence and clarify export boundaries. Error-solver migration should be
separately approved for error-solve workloads; it still has padded work and
quadrature. Phase 6 does not start either migration.

Phase 7: compact separate error solver
-------------------------------------

``DFN.solve_error()`` now follows the active normal storage. The public
compact sequence is::

    dfn.solve(compact=True)
    dfn.solve_error()
    value = dfn.calc_error(z, fracture)
    head = dfn.calc_head(z, fracture)  # still the normal solution
    boundary_errors, worst = dfn.check_boundary_conditions()

Error sampling and error plots also use compact state, including ordered
fracture subsets. No padded consolidation/export is needed. The legacy
default, explicit compact opt-in, padded dtypes, constants and persistence
remain unchanged. The Phase 6 error-solver limitations above describe that
earlier phase and are superseded by this section.

Legacy flow and allocation/lifetime map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The public entry point copies the **complete** padded HPC element array,
zeros its coefficient field, and preserves its other fields. It calls
``hpc_error_solve.solve_error``, which selects discharge elements, constructs
quadrature and the sparse matrix/LU, initializes caches, and iterates the
error problem. Each iteration resets error discharges/constants, evaluates
the RHS from the normal and error potentials, solves/publishes discharges
and error fracture constants, computes element candidates, then commits
damped coefficients at a separate parallel barrier. It reduces convergence
metrics and stops after two successful checks or 20 iterations. The success
counter is not reset between checks; that existing rule is unchanged.

**There is no adaptive coefficient growth in this error algorithm.** Orders
and integration lengths remain those of the normal solution. There are no
error growth requests, capacity transactions, eligibility changes or
coefficient-order repacks. The normal adaptive architecture remains intact.

Let N/F/S be element/fracture/selected-discharge counts, M/E the padded
coefficient/element maxima, q the operation's quadrature count, m_i/n_i the
logical element orders/integration lengths, C/I the sums of coefficient/
integration capacities, K the sparse nonzero count and T the maximum worker
count. Bytes below describe actual NumPy payload, not object overhead.
The benchmark environment has M=200 and E=800 from local configuration.

.. list-table::
   :header-rows: 1
   :widths: 26 20 24 30

   * - Allocation / producer
     - Dtype and physical shape
     - Lifetime / logical requirement
     - Maximum dependence / compact replacement
   * - ``DFN.solve_error`` error records
     - Complete ``element_dtype_hpc[N]``; 12,992N bytes
     - Retained output; fixed geometry plus m_i coefficients and n_i cache entries
     - M padding and redundant inherited old coefficients; shared geometry/layout plus 40N error scalars, capacity coefficients and private correction cache
   * - ``get_discharge_elements`` selection
     - Boolean mask N; **complete structured-record copy** of S rows
     - Mask during selection, copied records throughout solve; only ordered IDs are required
     - M-sized record copy; reuse ordered int64 discharge indices (8S bytes)
   * - ``solve_error`` work
     - Two zeroed ``dtype_work[N]`` arrays; 70,433N bytes each
     - Error work throughout iterations; original work initialized but never actively consumed
     - M/E padding; existing capacity work, worker scratch and private ragged events; omit original work only in compact execution
   * - ``solve_error`` head/discharge/history
     - Three float64 vectors, S+F entries each (legacy discharge initialization uses S+max(frac0)+1)
     - Throughout solve; RHS, current discharge and previous discharge
     - No storage maximum; same numerical vectors retained
   * - ``solve_error/get_z_int_array`` selected grid
     - ``dtype_z_arrays[N]``; two complex128 sides, 2M entries/side
     - Throughout solve; only S*q points per side are read
     - 128MN bytes; existing runtime dense complex128[2,S,q], 32Sq bytes
   * - ``solve_error/get_z_int_array`` error grid
     - Second ``dtype_z_arrays[N]``
     - Throughout solve; populated but never actively read
     - M padding; omitted only in compact path
   * - Legacy grid/branch coordinate producers
     - float64[n_i] or complex128[n_i]; line producers use q+2
     - Element-local coordinate generation
     - Logical sizes already; direct compact branch staging/runtime dense grid reuse the same arithmetic and order
   * - ``build_discharge_matrix`` lookup/count/offset
     - int32[max_id+1], int64[S+F] counts and offsets
     - Matrix construction; ID lookup and row layout
     - No M/E padding; existing indexed CSR-connectivity matrix preparation
   * - ``build_discharge_matrix`` triplets
     - int64[K] rows/columns, float64[K] values
     - Matrix construction, released after CSC construction
     - Actual nonzeros; unchanged matrix formulation and existing compact producer
   * - SciPy CSC / SuperLU
     - Sparse index/value arrays and native factorization storage
     - Matrix/factorization throughout solve
     - Actual sparsity/fill; unchanged, excluded from payload tables below
   * - Error Cauchy real/intersection routines
     - complex128[m_i] integral; two float64[n_i] arrays
     - One element invocation; two intersection calculations run successively
     - Already logical sizes; numerical expression allocations retained
   * - Error imaginary Cauchy routine
     - Two float64[n_i] and complex128[n_i] arrays
     - One bounding-circle invocation
     - Already logical sizes; unchanged
   * - Coefficient metrics and damped commit expressions
     - Complex/real arrays proportional to m_i; fixed scalar metrics
     - Element-local reduction/commit
     - Already logical sizes; same expressions and summation order
   * - Final ``solve_error`` replacement work
     - Fresh zeroed ``dtype_work[N]``
     - Allocated after iterations and returned; not used by an active error diagnostic
     - M/E padding; omitted compact-only
   * - Final ``solve_error`` replacement grid
     - Fresh ``dtype_z_arrays[N]``, populated
     - After iterations; only consumer is a commented-out diagnostic
     - M padding; omitted compact-only
   * - ``get_errors`` post-processing
     - complex128[F_selected,P] errors/working coordinates; float64[F_selected,P,3] mapped points
     - One sampling/plot operation; P requested sample points
     - Actual output shape; numeric fracture views and ordered subset indices, no structured-record gather
   * - Optional public unconsolidation
     - Logical object coefficient/cache arrays
     - Publication back to normal Python objects only when requested
     - Direct compact logical copies, not a padded error-state bridge

The legacy fracture array is not copied by ``solve_error``: its
``error_constant`` is overwritten in place. Normal HPC theta caches are
recomputed in place. The normal DFN ``work_array`` is replaced on return
by the fresh, unused error work. In contrast, the compact path leaves all
normal arrays, normal scratch and the public normal work reference intact.
Both legacy work arrays and the old grids coexist during the solve.
Allocating the final replacement work briefly makes **three** work arrays
live; allocating its replacement grid can briefly make three grid arrays
live. It would be incorrect to add every allocation site as four concurrent
work arrays plus three grids.

Repository-wide consumer searches confirmed that ``work_array_org``,
``z_int_error`` and the final diagnostic grid have no active numerical
reader in error execution: the final ``get_bnd_error`` invocation is inside
a triple-quoted block. Their legacy allocations are retained. The similarly
named ``z_integral`` cache still has diagnostic/export consumers and is
retained/shared, not removed.

Reused architecture and state isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hpc_error.ErrorState`` owns an existing ``AdaptiveState/AdaptiveStorage``
and numeric element/fracture views; it is not another ragged storage format.
The views allow the **same error numerical routines** to consume compact
arrays. No second implementations of coefficient layout, connectivity,
quadrature, cache storage, worker ownership or branch construction are added.
The shared error host solver has storage-specific preparation, selection and
view dispatch, not different numerical algorithms.

.. list-table::
   :header-rows: 1

   * - Normal state / error state
     - Ownership and reason
   * - Fixed element metadata
     - Shared read-only: IDs/types, fracture indices, geometry/endpoints, radius/center, head/phi, ncoef/nint and all other fixed fields
   * - Error q/error/error_old/error_old2/error_coef
     - Separate five-field float64 structured array (40N); initialized from the corresponding normal values, with the original error initialization applied afterward
   * - Normal coefficients
     - Shared read-only numeric view for normal potential evaluation; never reused as error output
   * - Error current/candidate
     - Separate zeroed complex128[C] buffers using the existing starts/capacities; staged updates preserve neighbor reads before commit
   * - Error history
     - One complex128[C] buffer: ``storage.old_coef_data`` aliases ``work.old_coef``; stores pre-candidate coefficients for metrics
   * - Inherited padded element old_coef
     - No separate copy: not an active error-algorithm input
   * - Coefficient metric
     - Actual metric is scalar error_coef; unused work.coef_error has zero entries in the compact error path, without first allocating a capacity array
   * - Coefficient/integration starts and capacities
     - Shared read-only; integration layout preserves the normal owner's exact spare capacity and offsets after adaptive growth
   * - Theta, exp_array_m/exp_array_p, z_integral
     - Shared read-only initialized normal caches; no repeated cache population
   * - dpsi_corr
     - Private copy at integration capacity; initially copies normal corrections, including unsupported rows that the error algorithm never updates
   * - Fracture geometry, normal constants and CSR
     - Shared read-only, exact connectivity order; no full fracture metadata copy or element-record gather
   * - Fracture error_constant
     - Separate float64[F], copied initial values, mutated by error matrix publication
   * - Branch topology
     - Separate existing ragged event arrays/counts/validity/layout; error preparation cannot replace or mutate normal topology
   * - Worker scratch
     - Separate established worker-owned buffers, sized by actual maximum capacities and T; preserves even normal scratch contents
   * - Normal convergence/history/discharge
     - No shared mutable state; every normal buffer remains unchanged after error solving

The error-work lifetime classification is:

.. list-table::
   :header-rows: 1

   * - Field(s)
     - Classification / compact treatment
   * - coef
     - Candidate state; slice of existing separate candidate flat array
   * - old_coef
     - Persistent convergence history, aliased to the one error history family
   * - coef0 / coef1
     - Element-local scratch; existing worker slices, both coexist for intersection assembly
   * - integral
     - Element-local scratch in the standard work view; error Cauchy routines also retain their existing logical local integral expressions
   * - phi / psi / dpsi
     - Worker scratch through existing work-record ownership; no per-element padded rows
   * - exp_array_m / exp_array_p / z_integral
     - Integration cache; shared read-only at integration capacity
   * - thetas
     - Integration cache exposed through compact work/element views; shared read-only
   * - dpsi_corr
     - Persistent error integration correction, private at integration capacity
   * - sign_array / discharge_element / element_pos / len_discharge_element
     - Private branch-event state, existing capacity representation
   * - coef_error
     - Unused compatibility field; empty compact array (element scalar metric retained)
   * - a0_error / work.error / set_zero
     - Compatibility/scalar work slots retained in the shared work representation, not padded coefficient histories

Error branch preparation calls the Phase 4D ``prepare_direct`` producer
once on bounding-circle error rows and publishes only after preparation.
It preserves interval-major, connectivity-major, direct-before-mirror order.
Insufficient capacities are detected by the producer and recovered together
using affected-private staging and a batch repack; no count-only preliminary
pass or MAX_ELEMENTS dimension is introduced. Error topology is private:
although the geometry/grid can match normal topology, correction values
consume normal and error discharges. Valid empty topology remains valid;
the shared validity helper preserves the legacy zero-count behavior for
legacy records.

The selected discharge grid is uniform, operation-local, C-contiguous
complex128[2,S,q]. Positive/negative sides and summation order are unchanged.
Theta/exponential caches remain variable-capacity persistent integration
data, not forced into uniform dense quadrature. Matrix assembly/factorization
reuse existing indexed/CSR normal producers; sparse layout is not redesigned.

All error state stays private until ``DFN.solve_error`` returns successfully.
Failure preserves a previously published error result as well as normal
state. Normal re-solving, reset and model invalidation clear stale compact
error results. ``unconsolidate=True`` retains the existing meaning: publish
the normal solution to Python objects, not a newly invented error export.
No compact coefficient, cache or metadata is temporarily overwritten and
then "restored." Unsupported impermeable circle/line paths still print the
original warning and skip candidate computation; wells retain their original
error behavior. No missing numerical algorithm was implemented.

Measured memory and peak accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests\benchmark_error.py`` deduplicates NumPy array identities, excluding
normal arrays when reporting incremental error state. The heterogeneous
fixture has S=N/2, F=N/4, q=8, T=28 and fixed logical orders/capacities at all
sizes. These are seeded-cache storage/performance fixtures, not a claim that
a 10,000-element physical model was converged. The separate public-model
regression exercises actual normal solve -> error solve -> post-processing.
All units below are bytes; 100,000 is an affine storage-only projection,
not executed or timed.

.. list-table::
   :header-rows: 1

   * - Category
     - 128
     - 1,024
     - 10,000
     - 100,000 projected
   * - Retained normal compact state
     - 425,896
     - 2,955,304
     - 28,294,552
     - 282,364,552
   * - Additional fixed/error scalars + fracture constants
     - 5,376
     - 43,008
     - 420,000
     - 4,200,000
   * - Error current/candidate/history (history included once)
     - 70,656
     - 565,248
     - 5,520,000
     - 55,200,000
   * - Error integration correction (other caches shared)
     - 24,576
     - 196,608
     - 1,920,000
     - 19,200,000
   * - Error worker scratch
     - 64,512
     - 64,512
     - 64,512
     - 64,512
   * - Error branches/layout/validity
     - 9,344
     - 74,752
     - 730,000
     - 7,300,000
   * - Retained shared-work compatibility scalars/layout
     - 3,232
     - 25,632
     - 250,032
     - 2,500,032
   * - Additional steady error total
     - 177,696
     - 969,760
     - 8,904,544
     - 88,464,544
   * - Error operation-local quadrature
     - 16,384
     - 131,072
     - 1,280,000
     - 12,800,000
   * - Matrix inputs/selection (not CSC/LU)
     - 11,904
     - 95,232
     - 930,000
     - 9,300,000
   * - RHS/discharge/history vectors
     - 2,304
     - 18,432
     - 180,000
     - 1,800,000
   * - Branch preparation temporary envelope
     - 5,120
     - 40,960
     - 400,000
     - 4,000,000
   * - Host finite-check temporary envelope
     - 3,072
     - 24,576
     - 240,000
     - 2,400,000
   * - Normal + error conservative tracked envelope
     - 642,376
     - 4,235,336
     - 40,229,096
     - 401,129,096
   * - Legacy error-record full copy
     - 1,662,976
     - 13,303,808
     - 129,920,000
     - 1,299,200,000
   * - Legacy selected-record full copy
     - 831,488
     - 6,651,904
     - 64,960,000
     - 649,600,000
   * - Legacy two initial work arrays
     - 18,030,848
     - 144,246,784
     - 1,408,660,000
     - 14,086,600,000
   * - Legacy two initial padded quadrature arrays
     - 3,276,800
     - 26,214,400
     - 256,000,000
     - 2,560,000,000
   * - Legacy final dead work + grid allocation sites
     - 10,653,824
     - 85,230,592
     - 832,330,000
     - 8,323,300,000
   * - Retained legacy normal elements + fractures
     - 1,872,384
     - 14,979,072
     - 146,280,000
     - 1,462,800,000
   * - Retained legacy normal work during error call
     - 9,015,424
     - 72,123,392
     - 704,330,000
     - 7,043,300,000
   * - Legacy normal + error tracked peak envelope
     - 43,719,552
     - 349,756,416
     - 3,415,590,000
     - 34,155,900,000

At 10,000 elements normal plus steady error state is **37,199,096 bytes**;
adding quadrature gives 38,479,096. The 40,229,096-byte conservative envelope
also adds all matrix inputs/vectors and preparation/validation temporaries,
even though their lifetimes need not overlap. Error orders never grow, so
there is no coefficient transaction peak. Event-overflow replacements use
the existing affected staging/repack; the table's preparation envelope is
for this measured no-overflow fixture, not a universal bound on arbitrary
event counts.

These are tracked NumPy envelopes, not allocator-complete process peaks.
Native element-local Cauchy/geometry/metric/commit expression arrays scale
with actual m_i/n_i and concurrent workers; CSC/SuperLU arrays/fill, NRT,
Python objects and runtime/compiler overhead are not included. The legacy
envelope includes the retained normal work and the third simultaneous work
allocation, not a sum of every replacement allocation. One independent
Windows ``GetProcessMemoryInfo`` run (8 threads, 10,000 elements, paired
warm-up plus one measured sample) reported **1,045,803,008 bytes peak working
set**. That is a whole paired-process RSS measurement, including imports,
JIT/cache loading and both modes, not a compact-only peak. Demand-zero pages
and disjoint lifetimes explain why it is not equal to the sum of payload
allocation sites. No additional dependency was installed.

Storage-independence tests vary runtime legacy maxima through 200/500/1,000
coefficients and 500/5,000 elements with fixed numerical controls/logical
orders. Incremental compact error payload and results are exactly unchanged.
Combined with no-maxima compact constructors and explicit padded-allocation/
legacy-consolidation guards, this checks the actual compact path rather
than claiming monkeypatching changes import-time legacy dtype definitions.

Performance, profiling and thread behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark alternates modes, excludes warm-up, and reports three-sample
medians. Preparation includes the legacy complete error copy or compact
private state/branch preparation; matrix/grid/iteration timings are inside
the shared host solver. Python model construction and explicit legacy test
fixtures are outside timing. Legacy final unused allocations are inside
legacy total because they remain part of the existing public operation.
Final coefficient extraction is outside timing. All 18 comparisons are
finite and exactly equal; all benchmark solves take 19 iterations.
Legacy/compact total seconds and compact/legacy ratios:

.. list-table::
   :header-rows: 1

   * - Threads
     - 128
     - 1,024
     - 10,000
   * - 1
     - 0.0475 / 0.0651 (1.371)
     - 0.3634 / 0.4672 (1.285)
     - 3.6459 / 4.5094 (1.237)
   * - 2
     - 0.0309 / 0.0358 (1.157)
     - 0.2148 / 0.2604 (1.212)
     - 2.1831 / 2.6169 (1.199)
   * - 4
     - 0.0219 / 0.0247 (1.127)
     - 0.1497 / 0.1649 (1.101)
     - 1.3995 / 1.4877 (1.063)
   * - 8
     - 0.0210 / 0.0212 (1.011)
     - 0.1418 / 0.1304 (0.919)
     - 1.3077 / 1.0670 (0.816)
   * - 16
     - 0.0312 / 0.0319 (1.022)
     - 0.1726 / 0.1521 (0.882)
     - 1.5165 / 1.4766 (0.974)
   * - 28 (normal maximum)
     - 0.0493 / 0.0417 (0.847)
     - 0.2326 / 0.2056 (0.884)
     - 1.7471 / 1.5810 (0.905)

The following breakdown is compact, 10,000 elements, seconds. State
preparation includes branch preparation (the branch column is a subset);
host preparation additionally includes quadrature, matrix and setup.
Independent medians do not necessarily add to median total.

.. list-table::
   :header-rows: 1

   * - Threads
     - State prep / branch
     - Grid / matrix+LU
     - Iterations / mean per iteration
     - Post-processing
   * - 1
     - 0.089360 / 0.084615
     - 0.000793 / 0.005387
     - 4.406909 / 0.231943
     - 0.001400
   * - 2
     - 0.048980 / 0.044436
     - 0.000800 / 0.003839
     - 2.554770 / 0.134462
     - 0.000739
   * - 4
     - 0.029765 / 0.024385
     - 0.000826 / 0.003125
     - 1.446811 / 0.076148
     - 0.000429
   * - 8
     - 0.021112 / 0.016163
     - 0.000957 / 0.003242
     - 1.033192 / 0.054379
     - 0.000367
   * - 16
     - 0.019508 / 0.013760
     - 0.001535 / 0.003858
     - 1.435983 / 0.075578
     - 0.000298
   * - 28
     - 0.016959 / 0.009476
     - 0.001627 / 0.003980
     - 1.537718 / 0.080933
     - 0.000411

Full legacy/compact phase breakdowns at all sizes/settings are emitted by
the benchmark command. At one thread/10,000 elements compact element work
accounts for 3.398 seconds of the 4.407-second iteration total. The initial
eager-slice numeric views had a 1.529 large-model ratio. Profiling identified
host per-element validation and native element/RHS evaluation as the
overhead; vectorized finite validation and deferred numeric-view slicing
reduced it to 1.237. A final cProfile run at 1,024/one thread confirmed that
error iteration/element/RHS evaluation, rather than matrix or quadrature
construction, remains dominant. cProfile attributes native kernels to their
host call sites; it is not an instruction-level proof of one accessor's cost.
No unrelated normal-solver optimization or changed formula was used.

**The remaining low-thread regression is not resolved.** At 10,000 elements
compact is 23.7% slower at one thread and 19.9% slower at two; at four threads
it is 6.3% slower, and at 8/16/28 it is faster. Thus the >10% cases were
profiled, not waived or hidden. Default migration should wait for resolution
or explicit acceptance of this tradeoff. Best compact scaling here is at
eight threads (4.23x versus one); increasing to maximum threads is not
monotonic. This is measured behavior, not a bandwidth/cache claim.

Numerical validation and remaining conversion/constant audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Twelve new regressions exercise candidate and committed barriers at
1/2/4/8/16/28 threads, repeat execution, the actual public normal/error
workflow, forced error-event overflow, source integration spare capacity,
nonfinite rejection, failed publication and storage-limit independence.
Snapshots compare every iteration's IDs/orders, current/candidate/history
coefficients, error scalars, convergence counter/discharge error, error
fracture constants, integration caches/corrections, ordered events and
sampled error potentials. Normal arrays, CSR, coefficients/orders,
fracture metadata/constants, discharge, history, branch topology, caches
and even worker scratch are snapshotted and remain unchanged. Actual public
models also compare error potentials and normal heads/discharge/boundary
diagnostics after error solving. Unsupported warning/skip behavior remains.
**Maximum absolute and relative differences are zero** in these comparisons;
IDs, lengths, event order and deterministic iteration counts are exact.
Nonfinite inputs/results are explicitly rejected; no tolerance was relaxed.
There is no active separate error-boundary diagnostic in the legacy flow;
the commented-out diagnostic was not revived as a new algorithm.

There are **no compact -> padded -> compact execution round trips**.
Remaining conversion symbols and reasons are:

.. list-table::
   :header-rows: 1

   * - Location
     - Remaining reason
   * - ``storage.fixed_order_from_hpc/fixed_order_to_hpc/from_hpc/to_hpc``
     - Explicit older compatibility import/export; not called by direct compact normal/error execution
   * - ``hpc_adaptive.adaptive_from_hpc/to_hpc``
     - Explicit reference/import/export helpers, used by tests/benchmarks
   * - ``hpc_fracture.get_heads_from_hpc``
     - Existing name for a dispatcher that extracts metadata/CSR for legacy inputs and references them directly for numeric compact views; not a padded reconstruction
   * - ``storage.connectivity_from_hpc/fracture_metadata_from_hpc`` and ``hpc_solve`` callers
     - Legacy metadata/connectivity extraction or direct numeric-view pass-through; compact error matrix directly consumes CSR
   * - ``Element/Fracture.consolidate_into_hpc`` and legacy ``DFN.consolidate_dfn``
     - Retained per-object/default legacy compatibility conversion; bypassed by compact public solving
   * - Persistence/consolidation load/save paths
     - Existing padded schema and legacy lifecycle, deliberately unchanged

The repository-wide maximum audit remains as in Phase 6 except that
``hpc_error_solve`` allocation references now belong **only to legacy
execution**, not the compact error path:

.. list-table::
   :header-rows: 1

   * - References
     - Classification / Phase 7 result
   * - ``element.py`` maxima, padded dtypes, per-object converters
     - Legacy execution / compatibility export / persistence; retained
   * - ``dfn.py`` padded consolidation dimensions
     - Legacy execution / persistence / explicit compatibility; compact bypasses them
   * - ``hpc_solve`` and ``hpc_error_solve`` work/grid dtypes and legacy allocations
     - Legacy execution; compact normal/error/post-processing allocate none of these records
   * - ``constants.py`` scalar definitions/logging/validation and normal solver scalar MAX_NCOEF
     - Actual numerical stopping control plus legacy physical validation; no maximum-sized compact allocation
   * - Dtype-derived storage length guards and adaptive maximum-request increment
     - Legacy physical compatibility / overflow guard, not a compact allocation dimension; removal/unbinding still needs a deliberate compatibility decision
   * - MAX_DISCHARGE_INT definition/config read
     - Obsolete/dormant storage setting, no active allocation consumer; retained
   * - Tests, benchmarks, examples, documentation and local user scripts
     - Reference/configuration/numerical controls and deliberate padding comparisons; local user files untouched
   * - MAX_ITERATIONS / MAX_ERROR / damping / other numerical controls
     - Actual algorithmic controls; unchanged (including error solver's existing 20-iteration rule)

Validation executed: ``uv run pytest tests`` passed **410 tests**.
``uv run ruff check .`` and ``uv run ruff format --check .`` were attempted
and failed to start because project Ruff is unavailable. Pinned
``uvx ruff@0.16.2`` lint and format checks passed on all nine Phase 7 Python
files. The repository's existing explicit-file pre-commit Ruff lint/format
hooks passed. ``git diff --check`` passed. No dependencies or persistence
formats were changed.

Files changed: ``src\andfn\dfn.py``, ``src\andfn\hpc\hpc_connectivity.py``,
new ``src\andfn\hpc\hpc_error.py``, ``hpc_error_solve.py``,
``hpc_fracture.py``, ``hpc_math_functions.py``, ``hpc_work.py``,
new ``tests\test_compact_error.py``, new ``tests\benchmark_error.py`` and
this documentation. Existing normal numerical algorithms and default paths
remain.

Phase 8 removal inventory (before implementation)
--------------------------------------------------

The pre-removal source audit classified the remaining concepts as follows.
This table is retained as historical review evidence; the current-runtime
disposition is summarized at the start of this document.

.. list-table::
   :header-rows: 1

   * - Classification
     - Locations / concepts
     - Phase 8 disposition
   * - A: obsolete storage artifacts
     - ``element.MAX_NCOEF/MAX_ELEMENTS/MAX_DISCHARGE_INT``, HPC element/fracture dtypes and initializers; ``DFN.consolidate_dfn(hpc=True)`` staging; ``hpc_solve.dtype_work/dtype_z_arrays`` and ``hpc_error_solve`` equivalents; the legacy per-element work/grid allocations
     - Remove from active source/runtime; compact constructors and error/normal workflows already cover the corresponding data.
   * - B: legacy execution only
     - ``hpc_solve.solve`` legacy host loop and legacy-only work wrappers; ``hpc_error_solve`` padded dispatch/allocations; ``DFN`` branches that require retained HPC element/fracture arrays; per-object ``consolidate_hpc/unconsolidate_hpc``
     - Remove after public workflows and low-level compact callers are the sole runtime route; preserve numerical kernels that compact still calls.
   * - C: explicit conversion/export
     - ``storage.from_hpc/to_hpc``, ``fixed_order_from_hpc/to_hpc``, ``hpc_adaptive.adaptive_from_hpc/to_hpc``
     - The input is HPC state, not historical persistence. No public external-use evidence is present in repository code; remove implementation conversions with their source dtypes, while retaining only test-local padded references where regression tests require them.
   * - D: persistence boundary
     - ``element_dtype``, ``fracture_dtype``, object ``consolidate/unconsolidate``, ``DFN.save_dfn/load_dfn``
     - Retain: HDF5 saves object/logical arrays, not the HPC padded dtypes. On save, publish compact solved values to Python objects and serialize the existing logical schema. The HDF5 loader is already a logical compatibility reader.
   * - E: genuine numerical limit
     - Scalar solver ``constants["MAX_NCOEF"]`` used by adaptive eligibility/old-order checks; dtype-derived ``_MAX_REQUEST_INCREMENT`` overflow guard
     - Retain the solver limit under an unambiguous numerical name and preserve its old-order/overshoot behavior. Replace the overflow guard's dependency on a dtype shape with the equivalent int64 arithmetic bound.
   * - F: test/benchmark/reference
     - HPC kernel fixtures and expected-value calculations; storage conversion tests; benchmarks comparing legacy and compact paths
     - Convert to compact-default tests or isolate a small padded reference dtype in tests only. Delete benchmark branches that solely exercise removed legacy execution; retain numerical baselines reproducible from Phase 7 compact measurements.
   * - G: configuration/documentation
     - ``.andfn_config.yaml`` loading of sizing values, examples and migration-history tables/text
     - Remove obsolete sizing options from tracked docs/examples. The local untracked ``.andfn_config.yaml`` is user data and is not edited; unknown historical sizing keys will be ignored by the new runtime.
   * - H: active compact functionality
     - ``hpc_compact``, adaptive compact kernels, numeric CSR, worker work, ragged branches, runtime quadrature, compact error state and compact post-processing
     - Retain as the sole runtime implementation; avoid redesign.

The persistence audit found no need for a historical padded-runtime loader:
the existing save path serializes the non-HPC object dtypes, whose
coefficient/theta/correction fields contain per-element logical arrays.
The loader reads those datasets into Python elements. The HPC-specific
consolidation route is not used by save/load.

Phase 8 final structure
-----------------------

The default public workflow is ``solve()``, ``solve_error()``, and direct
compact post-processing. Coefficients and integration caches use
capacity-sized flat buffers, fracture connectivity uses CSR, branch events use
capacity-based ragged storage, and quadrature is allocated from actual runtime
counts. Adaptive updates publish replacement buffers in batches; storage size
does not depend on former global coefficient, element, or quadrature maxima.
Persistence remains at the logical Python-object boundary.

Compared with the original solver, numerical loops and solver sequencing remain
recognizable in most areas. The following structural changes are limited to
those required by compact storage:

.. list-table::
   :header-rows: 1

   * - Area
     - Structural comparison
     - Reason
   * - Consolidation
     - Substantially changed
     - Construct compact metadata, capacities, and CSR directly rather than first creating padded HPC records.
   * - Potential/head evaluation
     - Minimally changed for storage access
     - Numerical evaluators use compact coefficient views; arithmetic and summation order are preserved.
   * - Coefficient calculation
     - Minimally changed for storage access
     - Flat coefficient segments replace padded rows without changing the numerical calculation.
   * - Adaptive solving
     - Substantially changed
     - Capacity growth and batched publication are required to grow individual elements without global padded arrays or per-element hot-loop reconstruction.
   * - Fracture calculations
     - Minimally changed for storage access
     - Fixed metadata and CSR adjacency replace padded connectivity lists.
   * - Branch calculations
     - Substantially changed
     - Ragged event capacities and worker-owned scratch replace fixed per-element event arrays.
   * - Quadrature
     - Minimally changed for storage access
     - Temporary arrays use the selected rows and runtime quadrature count instead of padded records.
   * - Boundary diagnostics
     - Minimally changed for storage access
     - Diagnostics consume compact state and runtime-sized quadrature directly.
   * - Error solving
     - Substantially changed
     - Error coefficients and solver state are isolated in compact storage while retaining the existing iteration behavior.
   * - Post-processing
     - Minimally changed for storage access
     - Public methods read compact coefficients and connectivity directly; the underlying calculations remain recognizable.

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
