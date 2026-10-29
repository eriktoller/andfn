Example 3: Simulation only a subset fo the DFN model
=======================================

This is an example of how to use the ``andfn.RegionBox`` class to define a region of interest in the DFN model and simulate only the fractures that intersect with the region. The following code creates a rectangular region and assigns constant head boundary conditions to the fractures that intersect with the region. The simulation is then run on the fractures that are inside the region.

.. tab-set::

    .. tab-item:: Static Scene

        .. image:: _static/example3.png
           :width: 100%
           :alt: Static scene

    .. tab-item:: Interactive Scene

        .. raw:: html

           <iframe
               src="_static/example3_3d.html"
               width="100%"
               height="700"
               style="border:none;">
           </iframe>

.. literalinclude:: ../examples/example3.py
   :language: python