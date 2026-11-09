"""Shared numerical fixtures for compact-solver tests and benchmarks."""

import numpy as np
from test_hpc_coefficients import network


def fixture(copies):
    f0, e0 = network()
    e0["ncoef"] = [8, 12, 0, 16, 20, 24, 12, 0]
    e0["nint"] = np.maximum(2 * e0["ncoef"], 2)
    fractures = np.tile(f0, copies)
    elements = np.tile(e0, copies)
    for i in range(copies):
        f = fractures[i * 2 : (i + 1) * 2]
        e = elements[i * 8 : (i + 1) * 8]
        f["_id"] += i * 2
        for row in f:
            row["elements"][: row["nelements"]] += i * 8
        e["_id"] += i * 8
        e["frac0"] += i * 2
        e["frac1"][e["frac1"] >= 0] += i * 2
    return fractures, elements


def differences(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Nonfinite numerical comparison; fixture must be investigated")
    absolute = np.abs(a - b)
    return {
        "absolute": float(np.max(absolute, initial=0)),
        "relative": float(np.max(absolute / np.maximum(np.abs(a), 1e-30), initial=0)),
    }
