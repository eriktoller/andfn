User Guide
==========

This is the user guide and it is currently under development.

First make sure that the andfn package is installed. If not, install it with the following command:

.. code-block:: console

    pip install andfn

The first step is to import the module

.. code-block:: python

    import andfn

The next step is to create the DFN model. Here a DFN model called 'My DFN Example' is created

.. code-block:: python

    my_dfn = andfn.DFN('My DFN example')


The next step is to create the fractures. Fracture can either be created or loaded

.. code-block:: python

    # Creating a fracture
    fracture0 = andfn.Fracture('Fracture 0', t=t0, radius=radius, center=center_point, normal=normal_array)

    # Loading file with fractures
    my_dfn.load_fracture(path_to_fracture_file)

Once the fractures are created element can be created and assigned to fractures. After that the fractures are loaded on to a defined DFN an intersection analysis is run with:

.. code-block:: python

    my_dfn.get_fracture_intersections()

which automatically adds the intersection element to each fracture.

The next step is to solve the groundwater flow with ``my_dfn.solve()``.