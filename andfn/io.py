import json

import numpy as np
from .fracture import fracture_from_dict


def numpy_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.complexfloating):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [{"real": x.real, "imag": x.imag} for x in obj.tolist()]
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_fractures(dfn, filename):
    """
    Export the fractures of a DFN to a JSON file.

    Parameters
    ----------
    dfn : DFN
        The DFN object containing the fractures.
    filename : str
        The name of the output JSON file.
    """

    # Check the ending of the filename and add .fracs if necessary
    if filename.split(".")[-1] != "fracs":
        filename = filename.split(".")[0] + ".fracs"
    fracs = [frac.to_dict(fracs_file=True) for frac in dfn.fractures]
    with open(filename, "w") as f:
        json.dump(fracs, f, default=numpy_converter, indent=4)


def import_fractures_from_json(filename):
    """
    Import fractures from a JSON file and create a DFN object.

    Parameters
    ----------
    filename : str
        The name of the input JSON file.

    Returns
    -------
    DFN
        The DFN object containing the imported fractures.
    """
    with open(filename, "r") as f:
        fracs_data = json.load(f)

    fracs = [fracture_from_dict(fd) for fd in fracs_data]

    return fracs
