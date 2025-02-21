HPC
===

This module contains the High Performance Computing (HPC) functions. The module uses Numba to create a compiled version
of the solve and plotter. The code is separated into different files, each containing a different set of functions.

The compiled function utilizes parallel processing to speed up the computation.

These functions are not to be called from the user directly, but are used by the other functions in the package.

Solver functions
----------------

.. automodule:: andfn.hpc.hpc_solve
   :members:
   :undoc-members:
   :show-inheritance

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
