"""
Notes
-----
This is an example of a model.
"""
import AnDFN


# Initialize the DFN
dfn = AnDFN.dfn.DFN('Test')

# Create fractures
frac1 = AnDFN.fracture.Fracture('Fracture 1')
frac2 = AnDFN.fracture.Fracture('Fracture 2')

# Add to DFN
dfn.add_fractures([frac1, frac2])

print('Complete!')