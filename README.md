# AnDFN, Analytical Discrete Fracture Network

## Introduction
AnDFN is a computer program for the modelling of groundwater flow in a discrete fracture network (DFN). The program is based on the Analytic Element Method (AEM) and is distributed as a Python package with various modules and scripts. The original development of this code was funded by BeFo (Stiftelsen Bergteknisk Forskning).

## Functionality
AnDFN currently have the following functionality:
- Generate random DFN
- Compute the intersections of a DFN
- Solve the AEM model for a DFN
- Plot the flow net for the AEM model
- Import DFNs
- Load and save DFNs

## Dependencies
AnDFN requires the following Python packages:
- Numpy
- Pyvista
- Matplotlib
- Numba
- h5py (for load and save)

## Citation
The basic theory for this program is published in:

Otto D.L. Strack, Erik A.L. Toller, An analytic element model for flow in fractured impermeable rock, *Journal of Hydrology*, 2024, 131983, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2024.131983.
