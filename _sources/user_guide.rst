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


The next step is to create the fractures. Fracture can either be created or loaded. The ``andfn`` package was not developed as a fracture generator, but rather as a tool to analyze and simulate flow in a DFN. Therefore, the fracture generation is limited and it is recommended to use a fracture generator to create the fractures and then load them into the DFN model. However, it is possible to create fractures with the ``andfn`` package. The following code loads a large DFN from a file called 'large_dfn.fracs'.  When fracture are imported from a file, the fracture intersections are automatically computed and assigned to the fractures.

.. Note:: Supported file formats:

    - ``.fracs``: This is the default file format for the ``andfn`` package. It is a JSON file that contains the fracture data.
    - ``.csv``: This is a CSV file that contains the fracture data. The user needs to specify the column names for the fracture data.
    - ``.fab``: This is a FracMan file format that contains the fracture data.

.. code-block:: python

    # Load fractures from a file
    dfn.import_fractures_from_file('large_dfn.fracs')


The next step is to creat flow boundary conditions. The following code creates a region box where specified faces are assigned a constant head boundary condition.

.. code-block:: python

    # Create a rectangular region box
    regbox = andfn.RectangularRegion(
        label="box",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=1000,
        yl=1000,
        zl=1000,
    )

    # Set constant head boundary conditions on the left and right faces of the box
    regbox.frac_intersections(dfn.fractures, face="left", head=100)
    regbox.frac_intersections(dfn.fractures, face="right", head=200)


Now the DFN read almost ready to be solved. First we need to run a connectivity analysis to make sure that the DFN is connected and that there are no unconnected fractures. The following code runs a connectivity analysis on the DFN model.

.. code-block:: python

    # Run connectivity analysis
    dfn.check_connectivity()

It is all set to solve the DFN model., which we do using the following code. We can also adjust the solver parameters, such as the maximum number of iterations and the tolerance.
.. code-block:: python

    # Set solver parameters
    dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)

    # Solve the DFN model
    dfn.solve()

The conefficients and discahrges are now save to the indivudual element and we can plot the results with the following code.

.. code-block:: python

    # This will create a plot window
    p1 = dfn.initiate_plotter(title=True, off_screen=False, scale=1, axis=True)

    # This will plot the fractures colored by their hydraulic head values, with a contour and opacity of 1.
    dfn.plot_fractures_head( p1, 40, 10, opacity=1, contour=True)

    p1.show() # show the plot