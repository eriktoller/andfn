"""
Note
------
This example demonstrate the andfn package for a large DFN with a region box. The fractures outside the box are deleted.

This can be used to compute the effective hydraulic conductivity for a portion of a large DFN.
"""

import andfn

if __name__ == "__main__":
    """
    IMPORT DFN
    This part initiate the dfn and import the fracture from a file. 
    This step will also automatically compute the fracture intersections and add them to the DFN.
    """
    # load the geometry
    dfn = andfn.DFN("My DFN")
    dfn.import_fractures_from_file("large_dfn.fracs")

    """
    ADD CONSTANT HEAD BOUNDARY CONDITIONS
    This part create a region box and add constant head boundary conditions to the fractures that intersect with the specified faces of the box. It also checks the connectivity of the DFN and sets the solver parameters.
    
    In this example fracture outside the box are deleted. 
    """
    # Add Region Box boundary
    head0 = 100
    head1 = 200
    regbox = andfn.RectangularRegion(
        label="box",
        center=[0, 0, 0],
        x_vec=[1, 0, 0],
        y_vec=[0, 1, 0],
        z_vec=[0, 0, 1],
        xl=500,
        yl=500,
        zl=500,
    )
    reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)

    print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
    dfn.delete_fracture(reg_fracs_out)
    print(
        f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}"
    )

    regbox.frac_intersections(
        dfn.fractures, face="left", head=head0
    )  # apply constant head boundary condition to the fractures that intersect with the left face of the box
    regbox.frac_intersections(
        dfn.fractures, face="right", head=head1
    )  # apply constant head boundary condition to the fractures that intersect with the right face of the box

    # check the connectivity of the DFN
    # this is important to ensure that the DFN is properly connected and that the solver can find a solution.
    dfn.check_connectivity()

    # set the solver parameters, I have chosen good values for this example, but you can change them to see how it affects the solution.
    dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)

    """
    SOLVE THE DFN
    
    Everything is set up, now we can solve the DFN. The solver will use the specified parameters to find a solution to the flow problem in the DFN. 
    """
    dfn.solve()

    """
    PLOTTING
    
    The hydraulic head distribution in the DFN can be visualized using the plotting functions provided by the andfn library. The plot_fractures_head function will create a 3D plot of the fractures colored by their hydraulic head values. The region box is also plotted to show the boundaries of the simulation domain.
    """
    # This will create a plot window
    p1 = dfn.initiate_plotter(title=True, off_screen=True, scale=1, axis=True)

    # This will plot the fractures colored by their hydraulic head values, with a contour and opacity of 1.
    dfn.plot_fractures_head(p1, 40, 3, opacity=1, contour=True)

    regbox.plot(p1, opacity=0.2)  # plot the region box with an opacity of 0.1

    p1.export_html("example3_3d.html")
    p1.show()  # show the plot

    print("All done!")
