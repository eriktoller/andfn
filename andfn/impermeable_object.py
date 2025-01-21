"""
Notes
-----
This module contains the impermeable object classes.
"""
import numpy as np

class impermeable_ellipse:
    def __init__(self, label, focis, nu, ncoef, nint, frac):
        """
        Initializes the impermeable ellipse class.
        Parameters
        ----------
        label : str or int
            The label of the impermeable ellipse.
        focis : list
            The focis of the impermeable ellipse.
        nu : float
            The angle of the major axis of the impermeable ellipse.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the impermeable ellipse is associated with.
        """
        self.label = label
        self.focis = focis
        self.nu = nu
        self.ncoef = ncoef
        self.nint = nint
        self.frac = frac

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

    def __str__(self):
        return f'Impermeable ellipse: {self.label}'


class impermeable_circle:
    def __init__(self, label, radius, ncoef, nint, frac):
        """
        Initializes the impermeable circle class.
        Parameters
        ----------
        label : str or int
            The label of the impermeable circle.
        r : float
            The radius of the impermeable circle.
        ncoef : int
            The number of coefficients in the asymptotic expansion.
        nint : int
            The number of integration points.
        frac : Fracture
            The fracture object that the impermeable circle is associated with.
        """
        self.label = label
        self.r = radius
        self.ncoef = ncoef
        self.nint = nint
        self.frac = frac

        # Create the pre-calculation variables
        self.thetas = np.linspace(start=0, stop=2 * np.pi, num=nint, endpoint=False)
        self.coef = np.zeros(ncoef, dtype=complex)
        self.dpsi_corr = np.zeros(self.nint - 1, dtype=float)
        self.error = 1

    def __str__(self):
        return f'Impermeable circle: {self.label}'
