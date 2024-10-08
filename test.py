import AnDFN
import numpy as np

if __name__ == '__main__':
    # Set up the plotter
    animate = False
    light = True
    plot_elements = False
    plot = True
    num_fracs = 1000
    radius_factor = 0.9
    center_factor = 10

    frac0 = AnDFN.fracture.Fracture('alplha', 1, radius=0.7110514502696412,
                                    center=np.array([-0.68586699,  0.10729588, -0.33259489]),
                                    normal=np.array([-0.22151986,  0.4296706, -0.87539256]))
    frac1 = AnDFN.fracture.Fracture('beta', 1, radius=0.9713630692426233,
                                    center=np.array([-0.61035344, -0.58131561, -0.69512863]),
                                    normal=np.array([-0.63418706, -0.47790732,  0.60779221]))
    frac2 = AnDFN.fracture.Fracture('gamma', 1, radius=0.4755757623772918,
                                    center=np.array([ 0.01075344, -0.0065505,  -0.03596422]),
                                    normal=np.array([ 0.29951863, -0.84592864, -0.44124067]))
    frac3 = AnDFN.fracture.Fracture('delta', 1, radius=0.47061661257300813,
                                    center=np.array([ 0.21129958, -0.81220744,  0.08007986]),
                                    normal=np.array([ 0.77094753, -0.44806996, -0.45262923]))

    well_3d = np.array([ 0.17377122, -1.0649661 ,  0.26637177])
    well_center = AnDFN.geometry_functions.map_3d_to_2d(well_3d, frac3)
    well = AnDFN.well.Well('well', .01, well_center, 1, frac3)
    const_head_3d0 = np.array([-0.26477364,  0.04502505, -0.46971794])
    const_head_3d1 = np.array([-0.28093481,  0.29687769, -0.342011  ])
    const_head_pnt0 = AnDFN.geometry_functions.map_3d_to_2d(const_head_3d0, frac0)
    const_head_pnt1 = AnDFN.geometry_functions.map_3d_to_2d(const_head_3d1, frac0)
    const_head = AnDFN.const_head.ConstantHeadLine('const_head', np.array([-0.43885606715490133 - 0.08605437416676523j, -0.2994522312574155-0.3321571318269438j]), 10, 20, 2.0, frac0)
    frac3.add_element(well)
    frac0.add_element(const_head)


    dfn = AnDFN.DFN('DFN test', 20)
    dfn.add_fracture([frac0, frac1, frac2, frac3])
    dfn.get_fracture_intersections(10,20)
    dfn.get_elements()
    dfn.build_discharge_matrix()
    dfn.build_head_matrix()
    dfn.solve()


    p1 = dfn.initiate_plotter()
    dfn.plot_fractures(p1)
    dfn.plot_elements(p1)
    dfn.plot_fractures_flow_net(p1,20, 200, 0.001)
    p1.show()