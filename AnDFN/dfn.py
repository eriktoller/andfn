"""
Notes
-----
This module contains the DFN class.
"""


class DFN:
    def __init__(self, label, fractures=None):
        self.label = label
        if fractures is None:
            fractures = []
        self.fractures = fractures

    def number_of_fractures(self):
        return len(self.fractures)

    def add_fractures(self, new_fractures):
        self.fractures.append(new_fractures)
        return None

    def consolidate(self):
        for e in self.elements:
            # some function to consolidate the elements into numpy arrays for the numba solver
            pass
        return None

    def build_discahrge_matrix(self):
        # some function to build the discharge matrix
        pass

    def solve_discarge_matrix(self):
        # some function to solve the discharge matrix
        pass
