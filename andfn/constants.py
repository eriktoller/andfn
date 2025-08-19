"""
Notes
-----
This module contains the constants used in the AnDFN model as a class.
"""

import numpy as np
from numba import set_num_threads
import logging
import sys
import os

dtype_constants = np.dtype(
    [
        ("RHO", np.float64),
        ("G", np.float64),
        ("SE_FACTOR", np.float64),
        ("MAX_ITERATIONS", np.int32),
        ("MAX_ERROR", np.float64),
        ("MAX_NCOEF", np.int32),
        ("COEF_INCREASE", np.int32),
        ("COEF_RATIO", np.float64),
        ("MAX_ELEMENTS", np.int32),
        ("NCOEF", np.int32),
        ("NINT", np.int32),
        ("NUM_THREADS", np.int32),
    ]
)


def load_yaml_config():
    """
    Load the constants from a YAML configuration file.
    """
    # Check if the .andfn_config.yaml file exists
    if not os.path.exists(".andfn_config.yaml"):
        return

    # Check if the yaml package is installed
    try:
        import yaml
    except ImportError:
        yaml = None

    # If yaml is not installed, raise an ImportError
    if yaml is None:
        raise ImportError(
            "The 'pyyaml' package is required to read the .andfn_config.yaml file. Install it with `pip install pyyaml`."
        )

    # Load the configuration from the YAML file
    with open(".andfn_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


# set the packages to WARNING level to avoid too much output
logging.getLogger("matplotlib").propagate = False
logging.getLogger("numba").propagate = False
logging.getLogger("PIL").propagate = False

# Configure the logging
logging.basicConfig(level="INFO", format="%(message)s", stream=sys.stdout)

# Set up the logger
logger = logging.getLogger("andfn")
if os.path.exists(".andfn_config.yaml"):
    config = load_yaml_config()
    if config.get("LOG_LEVEL"):
        logger.setLevel(config["LOG_LEVEL"])
    if config.get("LOG_FILE"):
        file_handler = logging.FileHandler(config["LOG_FILE"], mode="w")
        formatter = logging.Formatter(
            "%(asctime)s [%(module)s] %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class Constants:
    def __init__(self):
        """
        Initialize the constants
        """
        # create the array
        self.constants = np.array(
            (
                1000.0,  # Density of water in kg/m^3
                9.81,  # Gravitational acceleration in m/s^2
                1.0,  # SE factor (shortening element length)
                50,  # Maximum number of iterations
                1e-6,  # Maximum error
                150,  # Maximum number of coefficients
                5,  # Coefficient increase factor
                0.05,  # Coefficient ratio
                150,  # Maximum number of elements
                5,  # Number of coefficients (default)
                10,  # Number of integration points (default)
                -1,  # Number of threads (default -1 = use all available threads)
            ),
            dtype=dtype_constants,
        )

        # Load the YAML configuration if available
        self.configure_constants()

    def print_constants(self):
        """
        Print the constants
        """
        logger.info("Constants:")
        logger.info(f"            RHO: {self.constants['RHO']}")
        logger.info(f"              G: {self.constants['G']}")
        logger.info(f"      SE_FACTOR: {self.constants['SE_FACTOR']}")
        logger.info(f" MAX_ITERATIONS: {self.constants['MAX_ITERATIONS']}")
        logger.info(f"      MAX_ERROR: {self.constants['MAX_ERROR']}")
        logger.info(f"      MAX_NCOEF: {self.constants['MAX_NCOEF']}")
        logger.info(f"  COEF_INCREASE: {self.constants['COEF_INCREASE']}")
        logger.info(f"     COEF_RATIO: {self.constants['COEF_RATIO']}")
        logger.info(f"   MAX_ELEMENTS: {self.constants['MAX_ELEMENTS']}")
        logger.info(f"          NCOEF: {self.constants['NCOEF']}")
        logger.info(f"           NINT: {self.constants['NINT']}")

    def print_solver_constants(self):
        """
        Print the solver constants
        """
        logger.info("Solver Constants:")
        logger.info(f" MAX_ITERATIONS: {self.constants['MAX_ITERATIONS']}")
        logger.info(f"      MAX_ERROR: {self.constants['MAX_ERROR']}")
        logger.info(f"       MAX_NCOEF: {self.constants['MAX_NCOEF']}")
        logger.info(f"  COEF_INCREASE: {self.constants['COEF_INCREASE']}")
        logger.info(f"     COEF_RATIO: {self.constants['COEF_RATIO']}")
        logger.info(
            f"    NUM_THREADS: {'all' if self.constants['NUM_THREADS'] == -1 else self.constants['NUM_THREADS']}"
        )

    def change_constants(self, **kwargs):
        """
        Function that changes the constants of the model. This function can either be called with keyword arguments or
        by loading a YAML configuration file named `.andfn_config.yaml` in the current directory.

        The following constants can be changed:
            - RHO: Density of water in kg/m^3
            - G: Gravitational acceleration in m/s^2
            - PI: Pi
            - SE_FACTOR: Shortening element length factor
            - MAX_ITERATIONS: Maximum number of iterations
            - MAX_ERROR: Maximum error
            - MAX_NCOEF: Maximum number of coefficients
            - COEF_INCREASE: Coefficient increase factor
            - COEF_RATIO: Coefficient ratio
            - MAX_ELEMENTS: Maximum number of elements
            - NCOEF: Number of coefficients
            - NINT: Number of integration points
            - NUM_THREADS: Number of threads to use for Numba (default -1 = use all available threads)

        Parameters
        ----------
        **kwargs : dict
            The constants to change. The keys should be the names of the constants and the values should be the new values.

        """
        for key, value in kwargs.items():
            if key in self.constants.dtype.names:
                if key == "NUM_THREADS":
                    # Set the number of threads for Numba
                    if value <= 0:
                        raise ValueError("Number of threads must be greater than 0")
                    set_num_threads(value)
                if key == "MAX_NCOEF":
                    from .element import MAX_NCOEF

                    # Ensure MAX_NCOEF is not greater than what is set in the code
                    if value > MAX_NCOEF:
                        raise ValueError(
                            f"MAX_NCOEF cannot be greater than {MAX_NCOEF}. I you need a higher value set the MAX_NCOEF using the .andfn_config.yaml file instead."
                        )
                if key == "MAX_ELEMENTS":
                    from .element import MAX_ELEMENTS

                    # Ensure MAX_ELEMENTS is not greater than what is set in the code
                    if value > MAX_ELEMENTS:
                        raise ValueError(
                            f"MAX_ELEMENTS cannot be greater than {MAX_ELEMENTS}. If you need a higher value set the MAX_ELEMENTS using the .andfn_config.yaml file instead."
                        )
                self.constants[key] = value

    def configure_constants(self):
        if os.path.exists(".andfn_config.yaml"):
            config = load_yaml_config()
            self.change_constants(**config)
