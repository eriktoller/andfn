import numpy as np

from andfn.element import fracture_dtype_hpc
from andfn.hpc import hpc_geometry_functions as hgf


def test_line_circle_mappings_roundtrip():
    endpoints = np.array([0.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    z = 1.5 + 0.2j

    chi = hgf.map_z_line_to_chi(z, endpoints)
    z_back = hgf.map_chi_to_z_line(chi, endpoints)

    assert z_back == z

    zc = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex128)
    center = 1.0 + 0.5j
    r = 2.0
    chi_c = hgf.map_z_circle_to_chi(zc, r, center)
    zc_back = hgf.map_chi_to_z_circle(chi_c, r, center)
    assert np.allclose(zc_back, zc)


def test_map_2d_to_3d_and_back():
    frac = np.zeros(1, dtype=fracture_dtype_hpc)
    frac[0]["center"] = np.array([10.0, 20.0, 30.0])
    frac[0]["x_vector"] = np.array([1.0, 0.0, 0.0])
    frac[0]["y_vector"] = np.array([0.0, 1.0, 0.0])

    z = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128)
    pnts = np.zeros((z.size, 3), dtype=np.float64)

    out = hgf.map_2d_to_3d(frac[0], z, pnts)
    assert np.allclose(out[0], np.array([11.0, 22.0, 30.0]))
    assert np.allclose(out[1], np.array([7.0, 20.5, 30.0]))

    z_back0 = hgf.map_3d_to_2d(frac[0], out[0])
    z_back1 = hgf.map_3d_to_2d(frac[0], out[1])
    assert z_back0 == z[0]
    assert z_back1 == z[1]
