"""
Notes
-----
This module contains the constants used in the AnDFN model as a class.
"""

import numpy as np

dtype_constants = np.dtype([
    ('RHO', np.float64),
    ('G', np.float64),
    ('PI', np.float64),
    ('MAX_ITERATIONS', np.int32),
    ('MAX_ERROR', np.float64),
    ('MAX_COEF', np.int32),
    ('COEF_INCREASE', np.float64),
    ('MAX_ELEMENTS', np.int32),
    ('NCOEF', np.int32),
    ('NINT', np.int32)
])

class Constants:
    def __init__(self):
        """
        Initialize the constants
        """
        # create the array
        self.constants = np.array([(
            1000.0, # Density of water in kg/m^3
            9.81,   # Gravitational acceleration in m/s^2
            np.pi,  # Pi
            50,     # Maximum number of iterations
            1e-6,   # Maximum error
            150,    # Maximum number of coefficients
            5,      # Coefficient increase factor
            150,    # Maximum number of elements
            5,      # Number of coefficients (default)
            10      # Number of integration points (default)
        )], dtype=dtype_constants)

    def print_constants(self):
        """
        Print the constants
        """
        print("Constants:")
        print(f"            RHO: {self.constants['RHO'][0]}")
        print(f"              G: {self.constants['G'][0]}")
        print(f"             PI: {self.constants['PI'][0]}")
        print(f" MAX_ITERATIONS: {self.constants['MAX_ITERATIONS'][0]}")
        print(f"      MAX_ERROR: {self.constants['MAX_ERROR'][0]}")
        print(f"       MAX_COEF: {self.constants['MAX_COEF'][0]}")
        print(f"  COEF_INCREASE: {self.constants['COEF_INCREASE'][0]}")
        print(f"   MAX_ELEMENTS: {self.constants['MAX_ELEMENTS'][0]}")
        print(f"          NCOEF: {self.constants['NCOEF'][0]}")
        print(f"           NINT: {self.constants['NINT'][0]}")