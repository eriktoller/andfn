User Guide
==========

This is the user guide.

The first step of creating a DFN model is to create a dfn object with ``my_dfn = andfn.DFN('My DFN example')``.

The next step i to create the fractures. Fracture can either be created manually or automatically. Once the fractures are created element can be created and assigned to fractures. After that the fractures are loaded on to a defined DFN and prefereably an intersection analysis is run with ``my_dfn.get_fracture_intersections()``. The intersection element will be automatically added to each fracture.

The next step is to solve the groundwater flow with ``my_dfn.solve()``.