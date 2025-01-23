"""
Notes
-----
This module contains the element class that is a parent class to all elements.
"""

import numpy as np
MAX_NCOEF = 3000
MAX_ELEMENTS = 100

element_dtype = np.dtype([
        ('id_', np.int_),
        ('type_', np.int_),
        ('frac0', np.int_),
        ('frac1', np.int_),
        ('endpoints0', np.complex128, 2),
        ('endpoints1', np.complex128, 2),
        ('radius', np.float64),
        ('center', np.complex128),
        ('head', np.float64),
        ('phi', np.float64),
        ('ncoef', np.int_),
        ('nint', np.int_),
        ('q', np.float64),
        ('thetas', np.ndarray),
        ('coef', np.ndarray),
        ('old_coef', np.ndarray),
        ('dpsi_corr', np.ndarray),
        ('error', np.float64)
])

element_dtype_hpc = np.dtype([
        ('id_', np.int_),
        ('type_', np.int_),
        ('frac0', np.int_),
        ('frac1', np.int_),
        ('endpoints0', np.complex128, 2),
        ('endpoints1', np.complex128, 2),
        ('radius', np.float64),
        ('center', np.complex128),
        ('head', np.float64),
        ('phi', np.float64),
        ('ncoef', np.int_),
        ('nint', np.int_),
        ('q', np.float64),
        ('thetas', np.float64, MAX_NCOEF*2),
        ('coef', np.complex128, MAX_NCOEF),
        ('old_coef', np.complex128, MAX_NCOEF),
        ('dpsi_corr', np.float64, MAX_NCOEF*2),
        ('error', np.float64)
])

"""
Element Types:
0 = Intersection
1 = Bounding Circle
2 = Well
3 = Constant Head Line
4 = Impermeable Circle
"""
element_index_dtype = np.dtype([
        ('label', np.str_,100),
        ('id_', np.int_),
        ('type_', np.int_),
])


fracture_dtype = np.dtype([
        ('id_', np.int_),
        ('t', np.float64),
        ('radius', np.float64),
        ('center', np.float64, 3),
        ('normal', np.float64, 3),
        ('x_vector', np.float64, 3),
        ('y_vector', np.float64, 3),
        ('elements', np.ndarray),
        ('constant', np.float64),
])

fracture_dtype_hpc = np.dtype([
        ('id_', np.int_),
        ('t', np.float64),
        ('radius', np.float64),
        ('center', np.float64, 3),
        ('normal', np.float64, 3),
        ('x_vector', np.float64, 3),
        ('y_vector', np.float64, 3),
        ('elements', np.int_, MAX_ELEMENTS),
        ('nelements', np.int_),
        ('constant', np.float64),
])

fracture_index_dtype = np.dtype([
        ('label', np.str_,100),
        ('id_', np.int_)
])

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
        elif np.issubdtype(element_dtype_hpc[name], np.ndarray):
            elements[name][0] = np.array([np.nan])

    return elements

class Element:
    """
    The parent class for all elements in the andfn model.
    """
    def __init__(self, label, id_, type_):
        self.label = label
        self.id_ = id_
        self.type_ = type_
        self.error = 1.0
        self.ncoef = 5
        self.nint = 10
        self.coef = np.zeros(self.ncoef, dtype=complex)
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=self.nint, endpoint=False)

    def __str__(self):
        """
        Returns the string representation of the element.

        Returns
        -------
        str
            The string representation of the element.
        """
        return f'{self.__class__.__name__}: {self.label}'

    def set_id(self, id_):
        """
        Set the id of the element.
        Parameters
        ----------
        id_ : int
            The id of the element.
        """
        self.id_ = id_

    def change_property(self, **kwargs):
        """
        Change a given property/ies of the element.
        Parameters
        ----------
        kwargs : dict
            The properties to change
        """
        assert all(key in element_dtype.names for key in kwargs.keys()), 'Invalid property name.'
        assert all(key in element_index_dtype.names for key in kwargs.keys()), 'Invalid property name.'

        for key, value in kwargs.items():
            setattr(self, key, value)

    def consolidate(self):
        """
        Consolidate into a numpy structures array.
        """
        struc_array = initiate_elements_array()

        for key in self.__dict__.keys():
            if key in element_dtype.names:
                match key:
                    case 'frac0' | 'frac1':
                        struc_array[key][0] = self.__dict__[key].id_
                    case _:
                        struc_array[key][0] = self.__dict__[key]

        index_array = np.array([(
            self.label,
            self.id_,
            self.type_
        )], dtype=element_index_dtype)

        return struc_array, index_array

    def consolidate_hpc(self):
        """
        Consolidate into a numpy structures array for HPC solver.
        """
        struc_array = initiate_elements_array_hpc()

        for key in self.__dict__.keys():
            if key in element_dtype.names:
                match key:
                    case 'frac0' | 'frac1':
                        struc_array[key][0] = self.__dict__[key].id_
                    case 'thetas' | 'coef' | 'old_coef' | 'dpsi_corr':
                        struc_array[key][0][:self.__dict__[key].size] = self.__dict__[key]
                    case _:
                        struc_array[key][0] = self.__dict__[key]

        index_array = np.array([(
            self.label,
            self.id_,
            self.type_
        )], dtype=element_index_dtype)

        return struc_array, index_array

    def unconsolidate(self, struc_array, index_array, fracs):
        """
        Unconsolidate from a numpy structures array.
        """
        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key == 'frac0' or key == 'frac1':
                    self.__dict__[key] = next(frac for frac in fracs if frac.id_ == struc_array[key])
                    continue
                self.__dict__[key] = struc_array[key]

        for key in index_array.dtype.names:
            self.__dict__[key] = index_array[key]

    def unconsolidate_hpc(self, struc_array, index_array, fracs):
        """
        Unconsolidate from a numpy structures array for HPC solver.
        """
        for key in self.__dict__.keys():
            if key in element_dtype.names:
                if key == 'frac0' or key == 'frac1':
                    self.__dict__[key] = next(frac for frac in fracs if frac.id_ == struc_array[key])
                    continue
                if key == 'coef' or key == 'old_coef':
                    self.__dict__[key] = struc_array[key][:struc_array['ncoef']]
                    continue
                if key == 'thetas' or key == 'dpsi_corr':
                    self.__dict__[key] = struc_array[key][:struc_array['nint']]
                    continue
                self.__dict__[key] = struc_array[key]

        for key in index_array.dtype.names:
            self.__dict__[key] = index_array[key]

    def set_new_ncoef(self, n, nint_mult=2):
        """
        Increase the number of coefficients in the asymptotic expansion.

        Parameters
        ----------
        n : int
            The new number of coefficients.
        nint_mult : int
            The multiplier for the number of integration points.
        """
        match self.type_:
            case 0, 3:  # Intersection, Constant Head Line
                self.ncoef = n
                self.coef = np.append(self.coef, np.zeros(n-self.coef.size, dtype=complex))
                self.nint = n * nint_mult
                self.thetas = np.linspace(start=np.pi / (2 * self.nint), stop=np.pi + np.pi / (2 * self.nint),
                                          num=self.nint, endpoint=False)
            case 1:  # Bounding Circle
                self.ncoef = n
                self.coef = np.append(self.coef, np.zeros(n-self.coef.size, dtype=complex))
                self.nint = n * nint_mult
                self.thetas = np.linspace(start=0, stop=2 * np.pi, num=self.nint, endpoint=False)
