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