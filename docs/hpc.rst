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
