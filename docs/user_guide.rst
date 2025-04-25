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
    fracture1 = andfn.Fracture('Fracture 0', t=t0, radius=radius, center=center_point, normal=normal_array)
    fracture2 = andfn.Fracture('Fracture 0', t=t0, radius=radius, center=center_point, normal=normal_array)
    fracture3 = andfn.Fracture('Fracture 0', t=t0, radius=radius, center=center_point, normal=normal_array)

    # Add the fracture to the DFN
    my_dfn.add_fracture([fracture0, fracture1, fracture2, fracture3])

Once the fractures are created element can be created and assigned to fractures. After that the fractures are loaded on to a defined DFN an intersection analysis is run with:

.. code-block:: python

    my_dfn.get_fracture_intersections()

which automatically adds the intersection element to each fracture.

It is also possible to import fractures from a file.

.. code-block:: python

    # Load fractures from a file
    my_dfn.import_fractures_from_file('fractures.csv')

This automatically performs an connected fractures analysis and adds the fractures and intersections to the DFN. Currently only CSV files are supported.

The next step is to solve the groundwater flow with ``my_dfn.solve()``.