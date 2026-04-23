# AnDFN, Analytical Discrete Fracture Network

<p>
  <a href="https://pypi.org/project/andfn/"><img src="https://img.shields.io/pypi/v/andfn.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/andfn/"><img src="https://img.shields.io/pypi/pyversions/andfn.svg" alt="Python versions"></a>
  <a href="https://github.com/eriktoller/andfn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/eriktoller/andfn.svg" alt="License"></a>
  <a href="https://eriktoller.github.io/andfn/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation"></a>
  <a href="https://github.com/eriktoller/andfn/actions"><img src="https://img.shields.io/github/actions/workflow/status/eriktoller/andfn/.github/workflows/publish_and_release.yml?branch=main" alt="Build status"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/eriktoller/andfn"><img src="https://img.shields.io/github/stars/eriktoller/andfn?style=social" alt="GitHub stars"></a>
</p>

## Introduction
AnDFN is a computer program for the modelling of groundwater flow in a discrete fracture network (DFN). The program is based on the Analytic Element Method (AEM) and is distributed as a Python package with various modules and scripts.

The documentation for AnDFN is available [here](https://eriktoller.github.io/andfn/).

## Installation
AnDFN can be installed from PyPi.

Installation:
```
pip install andfn
```

Update:
```
pip install andfn --upgrade
```

Uninstall
```
pip uninstall andfn
```

### Dependencies
`andfn` has the following required dependencies:
- `numpy`
- `pandas`
- `scipy`
- `pyvista`
- `numba`
- `h5py`

and the following optional dependencies:
- `pyyaml` (for using the YAML configuration file)
- `matplotlib` (for some optional plots)

## Functionality
AnDFN currently have the following functionality:
- Generate random DFN
- Compute the intersections of a DFN
- Solve the AEM model for a DFN
- Plot the flow net for the AEM model
- Import DFNs
- Load and save DFNs

## Getting started
A template for a simple AnDFN model and several examples are available in the `examples` folder (under development).

## Citation
The basic theory for this program is published in:

Otto D.L. Strack, Erik A.L. Toller, An analytic element model for flow in fractured impermeable rock, *Journal of Hydrology*, 2024, 131983, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2024.131983.

## Acknowledgements
The original development of this code was funded by [BeFo](https://www.befo.se) (Stiftelsen Bergteknisk Forskning) grant number 529.
