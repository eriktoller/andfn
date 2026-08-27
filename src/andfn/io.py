import json
import logging
import os
import time

import numpy as np
import scipy as sp

import andfn.geometry_functions as gf
from andfn.fracture import Fracture, fracture_from_dict

logger = logging.getLogger("andfn")

COLUMN_ALIASES = {
    "radius_str": ["r", "radius", "EquivRadius[m]"],
    "x_str": ["x", "east", "easting", "FractureX[m]"],
    "y_str": ["y", "north", "northing", "FractureY[m]"],
    "z_str": ["z", "elevation", "depth", "FractureZ[m]"],
    "t_str": ["t", "transmissivity", "Transmissivity[m2/s]"],
    "e_str": ["e", "aperture", "Aperture[m]"],
    "strike_str": ["strike", "Strike[deg]"],
    "dip_str": ["dip", "Dip_Angle[deg]"],
    "trend_str": ["trend", "Trend[deg]"],
    "plunge_str": ["plunge", "Plunge[deg]"],
}


def infer_columns(df, **kwargs):
    for param, aliases in COLUMN_ALIASES.items():
        if kwargs.get(param) is None:
            kwargs[param] = find_column(df, aliases)

    return kwargs


def find_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]

    return None


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


def import_fractures_from_csv(
    path,
    **kwargs,
):
    # Check if pandas is installed
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Pandas is required to import fractures from a file. Please install pandas."
        )

    data_file = pd.read_csv(path)
    cols = infer_columns(data_file, **kwargs)

    # Check that all required columns are present
    for col in ["radius_str", "x_str", "y_str", "z_str", "t_str", "e_str"]:
        if cols[col] is None:
            raise ValueError(
                f"Column for '{col}' not found in the data file. Please specify the column name in the function call."
            )

    if cols["trend_str"] is not None and cols["plunge_str"] is not None:
        orientation_method = gf.convert_trend_plunge_to_normal
        st_str = cols["trend_str"]
        dp_str = cols["plunge_str"]
    elif cols["strike_str"] is not None and cols["dip_str"] is not None:
        orientation_method = gf.convert_strike_dip_to_normal
        st_str = cols["strike_str"]
        dp_str = cols["dip_str"]
    else:
        raise ValueError(
            "Columns for either 'trend' and 'plunge' or 'strike' and 'dip' must be present in the data file. Please specify the column names in the function call."
        )

    # Extract the data from the file
    radius_arr = data_file[cols["radius_str"]].to_numpy()
    st_arr = data_file[st_str].to_numpy()
    dp_arr = data_file[dp_str].to_numpy()
    center_arr = data_file[[cols["x_str"], cols["y_str"], cols["z_str"]]].to_numpy()
    transmissivity_arr = data_file[cols["t_str"]].to_numpy()
    aperture_arr = data_file[cols["e_str"]].to_numpy()

    normals = np.array([orientation_method(st, dp) for st, dp in zip(st_arr, dp_arr)])

    frac = [
        Fracture(
            f"{i}",
            transmissivity_arr[i],
            radius_arr[i],
            center_arr[i],
            normals[i],
            aperture_arr[i],
        )
        for i in range(len(data_file))
    ]

    return frac


def import_fractures_from_fab(filename):
    """
    Import fractures from a FAB file and create a DFN object.

    Parameters
    ----------
    filename : str
        The name of the input FAB file.

    Returns
    -------
    fractures : list of dict
        A list of dictionaries, each representing a fracture with its properties..
    """

    # Read property names from BEGIN PROPERTIES section
    property_names = []

    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]

    in_properties = False
    in_format = False

    format_info = {}

    for line in lines:
        if line == "BEGIN FORMAT":
            in_format = True
            continue

        if line == "END FORMAT":
            in_format = False

        if in_format:
            format_info[line.split("=")[0].strip()] = line.split("=")[1].strip()

        if line == "BEGIN PROPERTIES":
            in_properties = True
            continue

        if line == "END PROPERTIES":
            break

        if in_properties:
            property_names.append(line.split('"')[1])

    # Replace "Transmissivity" with "t"
    property_names = [
        "t" if name == "Transmissivity" else name for name in property_names
    ]

    # Replace "Aperture" with "e"
    property_names = [
        "aperture" if name == "Aperture" else name for name in property_names
    ]

    fractures = []
    vertices = []

    i = 0

    while i < len(lines):
        if lines[i] == "BEGIN FRACTURE":
            break

        i += 1  # Move to the next line after "BEGIN FRACTURE"

    for num in range(int(format_info["No_Fractures"])):
        print(f" \rParsing fracture {num + 1}/{format_info['No_Fractures']}", end="")

        # ---------------------
        # Fracture header
        # ---------------------
        i += 1  # Calculate the index of the fracture header line
        header = lines[i].split()

        fracture_id = int(header[0])
        n_vertices = int(header[1])

        fracture = {
            "label": fracture_id,
        }

        # Add properties
        for j, prop in enumerate(property_names):
            fracture[prop] = float(header[3 + j])

        # ---------------------
        # Vertices
        # ---------------------
        vertices = np.empty((n_vertices, 3), dtype=float)
        for _ in range(n_vertices):
            i += 1
            row = lines[i].split()

            vertices[_] = [float(row[1]), float(row[2]), float(row[3])]

        area = polygon_area_3d(vertices)
        fracture["radius"] = equivalent_radius(area)
        fracture["center"] = center_from_vertices(vertices)

        # ---------------------
        # Normal vector
        # ---------------------
        i += 1
        normal = lines[i].split()

        fracture["normal"] = np.array(
            [float(normal[1]), float(normal[2]), float(normal[3])]
        )

        property_remove = [p for p in property_names if p not in ["t", "aperture"]]
        for key in property_remove:
            fracture.pop(key, None)

        fractures.append(fracture)

    fracs = [fracture_from_dict(fd) for fd in fractures]

    return fracs


def polygon_area_3d(vertices):
    """
    vertices: (n, 3) array of polygon vertices ordered around perimeter.

    Returns
    -------
    area : float
        Polygon area.
    """

    vertices = np.asarray(vertices)

    nx = ny = nz = 0.0
    n = len(vertices)

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]

        nx += (p1[1] - p2[1]) * (p1[2] + p2[2])
        ny += (p1[2] - p2[2]) * (p1[0] + p2[0])
        nz += (p1[0] - p2[0]) * (p1[1] + p2[1])

    return 0.5 * np.sqrt(nx**2 + ny**2 + nz**2)


def center_from_vertices(vertices):
    """
    vertices: (n, 3) array of polygon vertices ordered around perimeter.

    Returns
    -------
    center : (3,) array
        Polygon center.
    """
    return np.mean(vertices, axis=0)


def equivalent_radius(area):
    return np.sqrt(area / np.pi)


class IO:
    """
    Class for importing fractures from a file into a DFN model.
    """

    def import_fractures_from_file(
        self,
        path,
        starting_frac=None,
        remove_isolated=True,
        remove_tolerance=-1,
        **kwargs,
    ):
        """
        Imports fractures from a csv file. More formatting options can be added later.

        Parameters
        ----------
        path : str
            The path to the file containing the fractures. Supported file types are .csv, .fracs, and .fab.
        starting_frac : int, optional
            The fracture to use as the starting point for the connected fractures. The default is None.
        remove_isolated : bool, optional
            If True, removes isolated fractures from the DFN. The default is True.
        remove_tolerance : float, optional
            The tolerance to use when removing isolated fractures. The default is -1 (no tolerance).
        **kwargs : dict
            Additional keyword arguments to pass to the import_fractures_from_csv function.

            For example, you can specify the column names for radius, x, y, z, transmissivity, aperture, strike, dip, trend, and plunge when importing from a CSV file. If not specified, the function will try to infer the column names from the file.
            radius_str : str
                The name of the column containing the radius of the fractures.
            x_str : str
                The name of the column containing the x coordinate of the center of the fractures.
            y_str : str
                The name of the column containing the y coordinate of the center of the fractures.
            z_str : str
                The name of the column containing the z coordinate of the center of the fractures.
            t_str : str
                The name of the column containing the transmissivity of the fractures.
            e_str : str, optional
                The name of the column containing the aperture of the fractures. The default is None.
            strike_str : str, optional
                The name of the column containing the strike of the fractures. The default is None.
            dip_str : str, optional
                The name of the column containing the dip of the fractures. The default is None.
            trend_str : str, optional
                The name of the column containing the trend of the fractures. The default is None.
            plunge_str : str, optional
                The name of the column containing the plunge of the fractures. The default is None.

        Returns
        -------
        None
            The fractures are added to the DFN.
        """
        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".csv", ".fracs", ".fab"]:
            raise ValueError(
                f"The file {path} is not a valid fracture file. Only .csv and .fracs files are supported."
            )

        start = time.time()

        if ext == ".fracs":
            frac = import_fractures_from_json(path)
        elif ext == ".csv":
            frac = import_fractures_from_csv(
                path,
                **kwargs,
            )
        elif ext == ".fab":
            frac = import_fractures_from_fab(path)

        end = time.time()
        logger.info(
            f"Imported {len(frac)} fractures from {path} in {end - start:.2f} seconds."
        )

        # sort the fracture by radius, starting with the largest
        frac.sort(key=lambda f: f.radius, reverse=True)
        centers = np.array([f.center for f in frac])
        tree = sp.spatial.KDTree(centers)

        if starting_frac is not None:
            fracs = gf.get_connected_fractures(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
                fracture_surface=frac[starting_frac],
                tolerance=remove_tolerance,
            )
        else:
            fracs = gf.get_fracture_intersections(
                frac,
                self.constants["SE_FACTOR"],
                ncoef=self.constants["NCOEF"],
                nint=self.constants["NINT"],
                tolerance=remove_tolerance,
                tree=tree,
            )

        if remove_isolated:
            # Remove isolated fractures
            len_before = len(fracs)
            fracs = gf.remove_isolated_fractures(fracs)
            removed = len_before - len(fracs)
            if removed > 0:
                logger.info(
                    f"Removed {len_before - len(fracs)} isolated fractures from the DFN."
                )

        self.add_fracture(fracs)

        logger.info(
            f"Added and filtered {len(fracs)} fractures to the DFN in {time.time() - end:.2f} seconds."
        )


if __name__ == "__main__":
    filename = "../../data/test_fab.csv"

    model = IO.import_fractures_from_file(filename, filename)

# if __name__ == "__main__":
#
# Import fractures from a JSON file
#  imported_fracs = import_fractures_from_fab(r"../data/fab_fracs_test.fab")
