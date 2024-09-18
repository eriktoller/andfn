"""
Notes
-----
This module contains the elements classes.
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
