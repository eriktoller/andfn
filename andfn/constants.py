"""
Notes
-----
This module contains the constants used in the AnDFN model as a class.
"""

import numpy as np
from numba import set_num_threads

dtype_constants = np.dtype([
    ('RHO', np.float64),
    ('G', np.float64),
    ('PI', np.float64),
    ('MAX_ITERATIONS', np.int32),
    ('MAX_ERROR', np.float64),
    ('MAX_COEF', np.int32),
    ('COEF_INCREASE', np.int32),
    ('COEF_RATIO', np.float64),
    ('MAX_ELEMENTS', np.int32),
    ('NCOEF', np.int32),
    ('NINT', np.int32),
    ('NUM_THREADS', np.int32),
])

class Constants:
    def __init__(self):
        """
        Initialize the constants
        """
        # create the array
        self.constants = np.array((
            1000.0, # Density of water in kg/m^3
            9.81,   # Gravitational acceleration in m/s^2
            np.pi,  # Pi
            50,     # Maximum number of iterations
            1e-6,   # Maximum error
            150,    # Maximum number of coefficients
            5,      # Coefficient increase factor
            0.05,    # Coefficient ratio
            150,    # Maximum number of elements
            5,      # Number of coefficients (default)
            10,      # Number of integration points (default)
            -1      # Number of threads (default -1 = use all available threads)
        ), dtype=dtype_constants)



    def print_constants(self):
        """
        Print the constants
        """
        print("Constants:")
        print(f"            RHO: {self.constants['RHO']}")
        print(f"              G: {self.constants['G']}")
        print(f"             PI: {self.constants['PI']}")
        print(f" MAX_ITERATIONS: {self.constants['MAX_ITERATIONS']}")
        print(f"      MAX_ERROR: {self.constants['MAX_ERROR']}")
        print(f"       MAX_COEF: {self.constants['MAX_COEF']}")
        print(f"  COEF_INCREASE: {self.constants['COEF_INCREASE']}")
        print(f"   MAX_ELEMENTS: {self.constants['MAX_ELEMENTS']}")
        print(f"          NCOEF: {self.constants['NCOEF']}")
        print(f"           NINT: {self.constants['NINT']}")

    def print_solver_constants(self):
        """
        Print the solver constants
        """
        print("Solver Constants:")
        print(f" MAX_ITERATIONS: {self.constants['MAX_ITERATIONS']}")
        print(f"      MAX_ERROR: {self.constants['MAX_ERROR']}")
        print(f"       MAX_COEF: {self.constants['MAX_COEF']}")
        print(f"  COEF_INCREASE: {self.constants['COEF_INCREASE']}")
        print(f"     COEF_RATIO: {self.constants['COEF_RATIO']}")
        print(f"    NUM_THREADS: {'all' if self.constants['NUM_THREADS'] == -1 else self.constants['NUM_THREADS']}")


    def change_constants(self, **kwargs):
        """
        Change the constants

        Parameters
        ----------
        **kwargs : dict
            The constants to change. The keys should be the names of the constants and the values should be the new values.
            The following constants can be changed:
            - RHO: Density of water in kg/m^3
            - G: Gravitational acceleration in m/s^2
            - PI: Pi
            - MAX_ITERATIONS: Maximum number of iterations
            - MAX_ERROR: Maximum error
            - MAX_COEF: Maximum number of coefficients
            - COEF_INCREASE: Coefficient increase factor
            - COEF_RATIO: Coefficient ratio
            - MAX_ELEMENTS: Maximum number of elements
            - NCOEF: Number of coefficients
            - NINT: Number of integration points
            - NUM_THREADS: Number of threads to use for Numba (default -1 = use all available threads)
        """
        for key, value in kwargs.items():
            if key in self.constants.dtype.names:
                if key == 'NUM_THREADS':
                    # Set the number of threads for Numba
                    assert value > 0, "Number of threads must be greater than 0"
                    set_num_threads(value)
                self.constants[key] = value
            else:
                raise ValueError(f"Invalid constant name: {key}")