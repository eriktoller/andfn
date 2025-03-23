User Guide
==========

This is the user guide and it is currently under development.

The first step is to import the module

     .. code-block:: python

        import andfn

The next step is to create the DFN model. Here a model called 'My DFN Example'

    .. code-block:: python

        my_dfn = andfn.DFN('My DFN example')

The next step is to create the fractures. Fracture can either be created manually or automatically. Once the fractures are created element can be created and assigned to fractures. After that the fractures are loaded on to a defined DFN and prefereably an intersection analysis is run with ``my_dfn.get_fracture_intersections()``. The intersection element will be automatically added to each fracture.

The next step is to solve the groundwater flow with ``my_dfn.solve()``.