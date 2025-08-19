"""
Notes
-----

This module contains the element class that is a parent class to all elements.
"""

import numpy as np
import pyvista as pv
import andfn.geometry_functions as gf
from .constants import load_yaml_config

config = load_yaml_config()
MAX_NCOEF = 200
MAX_ELEMENTS = 300
if config:
    MAX_NCOEF = config.get("MAX_NCOEF", 200)
    MAX_ELEMENTS = config.get("MAX_ELEMENTS", 300)

element_dtype = np.dtype(
    [
        ("_id", np.int_),
        ("_type", np.int_),
        ("frac0", np.int_),
        ("frac1", np.int_),
        ("endpoints0", np.complex128, 2),
        ("endpoints1", np.complex128, 2),
        ("radius", np.float64),
        ("center", np.complex128),
        ("head", np.float64),
        ("phi", np.float64),
        ("ncoef", np.int_),
        ("nint", np.int_),
        ("q", np.float64),
        ("thetas", np.ndarray),
        ("coef", np.ndarray),
        ("old_coef", np.ndarray),
        ("dpsi_corr", np.ndarray),
        ("error", np.float64),
    ]
)

element_dtype_hpc = np.dtype(
    [
        ("_id", np.int_),
        ("_type", np.int_),
        ("frac0", np.int_),
        ("frac1", np.int_),
        ("endpoints0", np.complex128, 2),
        ("endpoints1", np.complex128, 2),
        ("radius", np.float64),
        ("center", np.complex128),
        ("head", np.float64),
        ("phi", np.float64),
        ("ncoef", np.int_),
        ("nint", np.int_),
        ("q", np.float64),
        ("thetas", np.float64, MAX_NCOEF * 2),
        ("coef", np.complex128, MAX_NCOEF),
        ("old_coef", np.complex128, MAX_NCOEF),
        ("dpsi_corr", np.float64, MAX_NCOEF * 2),
        ("error", np.float64),
        ("error_old", np.float64),
    ]
)
"""
The element data type for the HPC solver. Note that not all elements have all properties.

Element Types:
0 = Intersection
1 = Bounding Circle
2 = Well
3 = Constant Head Line
4 = Impermeable Circle
5 = Impermeable Line

Parameters
----------
_id : int
    The id of the element.
_type : int
    The type of the element. (see above)
frac0 : int
    The id of the first fracture.
frac1 : int
    The id of the second fracture.
endpoints0 : np.ndarray
    The endpoints of the element in frac0.
endpoints1 : np.ndarray
    The endpoints of the element in frac1.
radius : float
    The radius of the element.
center : np.ndarray
    The center of the element.
head : float
    The hydraulic head of the element.
phi : float
    The discharge potential of the element.
ncoef : int
    The number of coefficients in the expansion.
nint : int
    The number of integration points.
q : float
    The discharge of the element.
thetas : np.ndarray
    The angles of the integration points in the chi-plane.
coef : np.ndarray
    The coefficients of the expansion.
old_coef : np.ndarray
    The old coefficients of the expansion.
dpsi_corr : np.ndarray
    The correction to the psi values.
error : float
    The error of the element.
    

"""
element_index_dtype = np.dtype(
    [
        ("label", np.str_, 100),
        ("_id", np.int_),
        ("_type", np.int_),
    ]
)


fracture_dtype = np.dtype(
    [
        ("_id", np.int_),
        ("t", np.float64),
        ("radius", np.float64),
        ("center", np.float64, 3),
        ("normal", np.float64, 3),
        ("x_vector", np.float64, 3),
        ("y_vector", np.float64, 3),
        ("elements", np.ndarray),
        ("constant", np.float64),
    ]
)

fracture_dtype_hpc = np.dtype(
    [
        ("_id", np.int_),
        ("t", np.float64),
        ("radius", np.float64),
        ("center", np.float64, 3),
        ("normal", np.float64, 3),
        ("x_vector", np.float64, 3),
        ("y_vector", np.float64, 3),
        ("elements", np.int_, MAX_ELEMENTS),
        ("nelements", np.int_),
        ("constant", np.float64),
    ]
)

fracture_index_dtype = np.dtype([("label", np.str_, 100), ("_id", np.int_)])

ELEMENT_COLORS = {
    0: "#008000",  # Intersection
    1: "#000000",  # Bounding Circle
    2: "#FF0000",  # Well
    3: "#FF0000",  # Constant Head Line
    4: "#0000FF",  # Impermeable Circle
    5: "#0000FF",
}  # Impermeable Line


def initiate_elements_array():
    """
    Function that initiates the elements array.

    Returns
    -------
    elements : np.ndarray
        The elements array.
    """
    elements = np.empty(1, dtype=element_dtype)
    for name in elements.dtype.names:
        if np.issubdtype(element_dtype[name], np.int_):
            elements[name][0] = -1
        elif np.issubdtype(element_dtype[name], np.float64):
            elements[name][0] = np.nan
        elif np.issubdtype(element_dtype[name], np.complex128):
            elements[name][0] = np.nan + 1j * np.nan
        elif np.issubdtype(element_dtype[name], np.ndarray):
            elements[name][0] = np.array([np.nan])

    return elements


def initiate_elements_array_hpc():
    """
    Function that initiates the elements array for HPC.

    Returns
    -------
    elements : np.ndarray
        The elements array.
    """
    elements = np.empty(1, dtype=element_dtype_hpc)
    for name in elements.dtype.names:
        if np.issubdtype(element_dtype_hpc[name], np.int_):
            elements[name][0] = -1
        elif np.issubdtype(element_dtype_hpc[name], np.float64):
            elements[name][0] = np.nan
        elif np.issubdtype(element_dtype_hpc[name], np.complex128):
            elements[name][0] = np.nan + 1j * np.nan
        elif name == "thetas" or name == "dpsi_corr":
            elements[name][0] = np.zeros(MAX_NCOEF * 2, dtype=np.float64)
        elif name == "coef" or name == "old_coef":
            elements[name][0] = np.zeros(MAX_NCOEF, dtype=np.complex128)
        elif name == "endpoints0" or name == "endpoints1":
            elements[name][0] = np.full(2, np.nan + 1j * np.nan, dtype=np.complex128)

    return elements


class Element:
    """
    The parent class for all elements in the andfn model.
    """

    def __init__(self, label, _id, _type, frac0=None, frac1=None):
        """
        Initialize the element.

        Parameters
        ----------
        label : str
            The label of the element.
        _id : int
            The id of the element.
        _type : int
            The type of the element.
            Element Types:
                0 = Intersection
                1 = Bounding Circle
                2 = Well
                3 = Constant Head Line
                4 = Impermeable Circle
                5 = Impermeable Line
        """
        self.label = label
        self._id = _id
        self._type = _type
        self.error = 1.0
        self.ncoef = 5
        self.nint = 10
        self.coef = np.zeros(self.ncoef, dtype=complex)
        self.thetas = np.linspace(
            start=0, stop=2 * np.pi, num=self.nint, endpoint=False
        )

        # Assign the fractures if provided
        self.frac0 = frac0
        self.frac0.add_element(self)



    def __str__(self):
        """
        Returns the string representation of the element.

        Returns
        -------
        str
            The string representation of the element.
        """
        return f"{self.__class__.__name__}: {self.label}"

    def set_id(self, _id):
        """
        Set the id of the element.

        Parameters
        ----------
        _id : int
            The id of the element.

        Returns
        -------
        None. The id is updated in place.
        """
        self._id = _id

    def change_property(self, **kwargs):
        """
        Change a given property/ies of the element.

        Parameters
        ----------
        kwargs : dict
            The properties to change

        Returns
        -------
        None. The element is updated in place.
        """
        assert all(key in element_dtype.names for key in kwargs.keys()), (
            "Invalid property name."
        )
        assert all(key in element_index_dtype.names for key in kwargs.keys()), (
            "Invalid property name."
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def consolidate(self):
        """
        Consolidate into a numpy structures array.

        Returns
        -------
        struc_array : np.ndarray
            The structure array.
        index_array : np.ndarray
            The index array.
        """
        struc_array = initiate_elements_array()

        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key in ["frac0", "frac1"]:
                    struc_array[key][0] = self.__dict__[key]._id
                else:
                    struc_array[key][0] = self.__dict__[key]

        index_array = np.array(
            [(self.label, self._id, self._type)], dtype=element_index_dtype
        )

        return struc_array, index_array

    def consolidate_hpc(self):
        """
        Consolidate into a numpy structures array for HPC solver.

        Returns
        -------
        struc_array : np.ndarray
            The structure array.
        index_array : np.ndarray
            The index array.
        """
        struc_array = initiate_elements_array_hpc()

        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key in ["frac0", "frac1"]:
                    struc_array[key][0] = self.__dict__[key]._id
                elif key in ["thetas", "coef", "old_coef", "dpsi_corr"]:
                    struc_array[key][0][: self.__dict__[key].size] = self.__dict__[key]
                else:
                    struc_array[key][0] = self.__dict__[key]

        index_array = np.array(
            [(self.label, self._id, self._type)], dtype=element_index_dtype
        )

        return struc_array, index_array

    def unconsolidate(self, struc_array, index_array, fracs):
        """
        Unconsolidate from a numpy structures array.

        Returns
        -------
        None. The element is updated in place.
        """
        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key == "frac0" or key == "frac1":
                    self.__dict__[key] = next(
                        frac for frac in fracs if frac._id == struc_array[key]
                    )
                    continue
                self.__dict__[key] = struc_array[key]

        for key in index_array.dtype.names:
            self.__dict__[key] = index_array[key]

    def unconsolidate_hpc(self, struc_array, index_array, fracs):
        """
        Unconsolidate from a numpy structures array for HPC solver.

        Returns
        -------
        None. The element is updated in place.
        """
        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key == "frac0" or key == "frac1":
                    self.__dict__[key] = next(
                        frac for frac in fracs if frac._id == struc_array[key]
                    )
                    continue
                if key == "coef" or key == "old_coef":
                    self.__dict__[key] = struc_array[key][: struc_array["ncoef"]]
                    continue
                if key == "thetas" or key == "dpsi_corr":
                    self.__dict__[key] = struc_array[key][: struc_array["nint"]]
                    continue
                self.__dict__[key] = struc_array[key]

        for key in index_array.dtype.names:
            self.__dict__[key] = index_array[key]

    def plot(self, pl, line_width, color):
        """
        Plot the element using the given plotter.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter to use for plotting.
        line_width : float
            The width of the lines in the plot.
        color : str
            The color of the element in the plot, if None the default color will be used.

        Returns
        -------
        None
            The element is plotted in the plotter.
        """
        if self._type in [
            0,
            3,
            5,
        ]:  # Intersection, Constant Head Line, Impermeable Line
            line = gf.map_2d_to_3d(self.endpoints0, self.frac0)
            pl.add_mesh(
                pv.Line(line[0], line[1]),
                color=color if color is not None else ELEMENT_COLORS[self._type],
                line_width=line_width,
            )
        elif self._type == 1:  # Bounding Circle
            point = gf.map_2d_to_3d(0 + 0j, self.frac0)
            pl.add_mesh(
                pv.Polygon(
                    center=point, radius=self.radius, normal=self.frac0.normal, n_sides=50, fill=False
                ),
                color=color if color is not None else ELEMENT_COLORS[self._type],
                line_width=line_width,
            )
        elif self._type in [1, 2, 4]:  # Bounding Circle, Well, Impermeable Circle
            point = gf.map_2d_to_3d(self.center, self.frac0)
            pl.add_mesh(
                pv.Polygon(
                    center=point, radius=self.radius, normal=self.frac0.normal, n_sides=50, fill=False
                ),
                color=color if color is not None else ELEMENT_COLORS[self._type],
                line_width=line_width,
            )
