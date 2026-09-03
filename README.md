# AnDFN, Analytical Discrete Fracture Network

<p>
  <a href="https://pypi.org/project/andfn/"><img src="https://img.shields.io/pypi/v/andfn.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/andfn/"><img src="https://img.shields.io/pypi/pyversions/andfn.svg" alt="Python versions"></a>
  <a href="https://github.com/eriktoller/andfn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/eriktoller/andfn.svg" alt="License"></a>
  <a href="https://eriktoller.github.io/andfn/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation"></a>
  <a href="https://doi.org/10.5281/zenodo.22275898"><img src="https://zenodo.org/badge/857873613.svg" alt="DOI"></a> 
  <a href=""><img src="https://img.shields.io/github/actions/workflow/status/eriktoller/andfn/.github/workflows/publish_and_release.yml?branch=main" alt="Build status"></a>
  <a href="https://github.com/eriktoller/andfn/actions/workflows/format_lint.yml"><img src="https://github.com/eriktoller/andfn/actions/workflows/format_lint.yml/badge.svg" alt="CI RUFF"></a>
  <a href="https://github.com/eriktoller/andfn/actions/workflows/run_tests.yml"><img src="https://github.com/eriktoller/andfn/actions/workflows/run_tests.yml/badge.svg" alt="CI PYTESTS"></a>
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
`andfn` depends on a number of Python packages, which will be installed automatically when installing `andfn`. 

To install all dependencies run:
```
pip install andfn[all]
```

## Getting started
A template for a simple AnDFN model and several examples are available in the `examples` folder (under development).

## Citation
The basic theory for this program is published in:

Otto D.L. Strack, Erik A.L. Toller, An analytic element model for flow in fractured impermeable rock, *Journal of Hydrology*, 2024, 131983, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2024.131983.

You can also cite the software itself from Zenodo: https://doi.org/10.5281/zenodo.22275899

## Acknowledgements
The original development of this code was funded by [BeFo](https://www.befo.se) (Stiftelsen Bergteknisk Forskning) grant number 529.
