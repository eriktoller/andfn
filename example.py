"""
Notes
-----
This is an example of a model.
"""
import AnDFN

import numpy as np
import datetime

if __name__ == '__main__':
    # Set up the plotter
    animate = False
    light = True
    plot_elements = False
    plot = True
    num_fracs = 1000
    radius_factor = 0.65*1.7
    center_factor = 10

    start0 = datetime.datetime.now()
    print('\n---- IMPORTING MODULES ----')
    print(f'Program started at {start0}')
    print('\n---- CREATING DFN ----')
    dfn = AnDFN.DFN('DFN test', 20)
    frac_surface = AnDFN.Fracture('0', 1, 1, np.array([5, 5, 10]), np.array([0, 0, 1]))
    dfn.generate_connected_DFN(num_fracs, radius_factor, center_factor, 20, 40, frac_surface)
    dfn.fractures[0].add_element(AnDFN.Well('well0', .01, np.array([0 + 0*1j]), 1, dfn.fractures[0]))
    dfn.fractures[-1].add_element(AnDFN.Well('well1', .01, np.array([0 + 0*1j]), 0, dfn.fractures[-1]))
    dfn.get_elements()
    start1 = datetime.datetime.now()
    print('\n---- SOLVE THE DFN ----')
    dfn.build_discharge_matrix()
    dfn.build_head_matrix()
    dfn.solve()
    if plot:
        start2 = datetime.datetime.now()
        print('\n---- PLOTTING ----')
        p1 = dfn.initiate_plotter()
        dfn.plot_fractures(p1, opacity=1.0)
        if plot_elements:
            dfn.plot_elements(p1)
        dfn.plot_fractures_flow_net(p1, 40, 100, 0.001)
        end = datetime.datetime.now()
        print(f'\n\nProgram ended at {end}')
        print(f'Time elapsed: {end - start0}')
        print(f'\t-generating: \t{start1 - start0}')
        print(f'\t-solving: \t\t{start2- start1}')
        print(f'\t-plotting: \t\t{end - start2}')
        if animate:
            p1.camera.zoom(1.5)
            path = p1.generate_orbital_path(factor=20.0, n_points=36 * 2, shift=-20)
            p1.open_gif("orbit2.gif")
            p1.orbit_on_path(path, write_frames=True)
            p1.close()
        else:
            p1.show()
    else:
        end = datetime.datetime.now()
        print(f'\n\nProgram ended at {end}')
        print(f'Time elapsed: {end - start0}')
        print(f'\t-generating: {start1 - start0}')

