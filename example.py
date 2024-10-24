"""
Notes
-----
This is an example of a model.
"""
import AnDFN

import numpy as np
import datetime
import pickle

if __name__ == '__main__':
    # Set up the plotter
    animate = False
    light = True
    plot_elements = False
    plot = True
    save = False
    save_fig = False
    scale = 7

    num_fracs = 2000
    radius_factor = 0.25*5
    center_factor = 10

    start0 = datetime.datetime.now()
    print('\n---- IMPORTING MODULES ----')
    print(f'Program started at {start0}')
    print('\n---- CREATING DFN ----')
    dfn = AnDFN.DFN('DFN test', 20)
    frac_surface = AnDFN.Fracture('0', 1, radius_factor, np.array([center_factor/2, center_factor/2, center_factor/2]), np.array([0, 0, 1]))
    dfn.generate_connected_dfn(num_fracs, radius_factor, center_factor, 5, 20, 10, 40, frac_surface)
    dfn.fractures[0].add_element(AnDFN.Well('well0', .001, 0 + 0 * 1j, 20, dfn.fractures[0]))
    dfn.fractures[-1].add_element(AnDFN.Well('well1', .001, 0 + 0 * 1j, 10, dfn.fractures[-1]))
    # dfn.fractures[int(dfn.number_of_fractures()/3)].add_element(AnDFN.Well('well1', .001, 0 + 0 * 1j, 4, dfn.fractures[int(dfn.number_of_fractures()/3)]))
    # frac_surface2 = AnDFN.Fracture('new', 1, 50, np.array([45, 5, 10]), np.array([0, 1, 1]))
    # dfn.add_fracture(frac_surface2)
    # dfn.get_fracture_intersections(20, 40, frac_surface2)
    dfn.get_elements()
    # save the DFN as a pickle file
    if save:
        with open('dfn_large.pkl', 'wb') as f:
            pickle.dump(dfn, f)
        print('DFN saved as dfn_large.pkl')

    start1 = datetime.datetime.now()
    print('\n---- SOLVE THE DFN ----')
    dfn.build_discharge_matrix()
    dfn.build_head_matrix()
    dfn.solve(max_error=1e-2, max_iterations=30, boundary_check=True, tolerance=1e-2, n_boundary_check=10,
              max_iteration_boundary=3, lu_decomp=True)
    if plot:
        start2 = datetime.datetime.now()
        print('\n---- PLOTTING ----')
        if save_fig:
            p1 = dfn.initiate_plotter(title=False, off_screen=True, scale=scale, axis=False)
            dfn.plot_fractures_head(p1, 30, 100, 2 * scale, 0.001, opacity=.7, only_flow=True)
        else:
            p1 = dfn.initiate_plotter(title=True, off_screen=False, scale=1, axis=True)
            dfn.plot_fractures_head(p1, 30, 100, 2, 0.001, opacity=.3, only_flow=False,
                                    color_map='jet')
        if plot_elements:
            dfn.plot_elements(p1)
        end = datetime.datetime.now()
        print(f'\n\nProgram ended at {end}')
        print(f'Time elapsed: {end - start0}')
        print(f'\t-generating: \t{start1 - start0}')
        print(f'\t-solving: \t\t{start2 - start1}')
        print(f'\t-plotting: \t\t{end - start2}')
        if animate:
            p1.camera.zoom(1.5)
            path = p1.generate_orbital_path(factor=20.0, n_points=36 * 2, shift=-20)
            p1.open_gif("orbit2.gif")
            p1.orbit_on_path(path, write_frames=True)
            p1.close()
        elif save_fig:
            p1.screenshot(f'large_DFN_fracs{dfn.number_of_fractures()}.png', scale=scale)
        else:
            p1.show()
    else:
        end = datetime.datetime.now()
        print(f'\n\nProgram ended at {end}')
        print(f'Time elapsed: {end - start0}')
        print(f'\t-generating: {start1 - start0}')
