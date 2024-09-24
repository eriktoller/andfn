"""
Notes
-----
This module contains the elements classes.

pack the element list to a array structure before solving
"""


class Elements:
    def __init__(self, elements=None):
        if elements is None:
            elements = []
        self.elements = elements

        # Set predetermined values
        self.error = 1

    def add_element(self, new_element):
        self.elements.append(new_element)
        return None

    def solve(self):
        for e in self.elements:
            e.solve()
        return None
