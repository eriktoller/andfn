"""
Notes
-----
This is an example of a model.
"""
import AnDFN
import AnDFN.geometry_functions as gf

import numpy as np
import datetime

if __name__ == '__main__':
    # Set up the plotter
    animate = False
    light = True
    plot_elements = True
    plot = True
    num_fracs = 2000

    start = datetime.datetime.now()

    print('Generating fractures...')
    frac_surface = AnDFN.Fracture('0', 1, 5, np.array([5, 5, 10]), np.array([0, 0, 1]))
    fracs = gf.generate_fractures(num_fracs, radius_factor=0.5, center_factor=10)

    print('Analyzing intersections...')
    frac_list = gf.get_connected_fractures(fracs, frac_surface)

    print('Create DFN...')
    dfn = AnDFN.DFN('DFN test', 20)
    dfn.add_fracture(frac_list)
    print('Get elements...')
    dfn.get_elements()
    if plot:
        print('Plotting...')
        p1 = dfn.initiate_plotter()
        dfn.plot_fractures(p1)
        if plot_elements:
            dfn.plot_elements(p1)

        if animate:
            p1.camera.zoom(1.5)
            path = p1.generate_orbital_path(factor=20.0, n_points=36 * 2, shift=-20)
            p1.open_gif("orbit.gif")
            p1.orbit_on_path(path, write_frames=True)
            p1.close()
        else:
            p1.show()

    end = datetime.datetime.now()
    print(f'Time elapsed: {end - start}')