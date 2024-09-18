"""
Notes
-----
This module contains the fracture class.
"""
import elements


class Fracture:
    def __init__(self, label, elements=None):
        if elements is None:
            elements = []
        self.label = label
        self.elements = elements

    def __str__(self):
        return f'Fracture {self.label}'

    def add_element(self, new_element):
        self.elements.append(new_element)

    def set_new_label(self, new_label):
        self.label = new_label

    def calc_omega(self, z, exclude=None):
        """
        Calculates the omega for the fracture excluding element "exclude".

        Parameters
        ----------
        z : complex
            A point in the complex z plane.
        exclude : any
            Id of element to exclude from the omega calculation.

        Returns
        -------
        omega : complex
            The complex potential for the fracture.
        """
        omega = 0
        if exclude is None:
            for e in self.elements:
                omega += e.calc_omega(z)
        else:
            for e in self.elements:
                if e.label != exclude:
                    omega += e.calc_omega(z)
        return omega
