CONFIGURATION
-------------

It is possible to configure the behaviour of the ``andfn`` package by setting environment variables using the ``.andfn_config.yaml`` file. The following environment variables are available:

CONSTANTS
---------
``RHO: (float)``

The density of the fluid in the fractures. Deafult is 1000 kg/m³.

``G: (float)``

The gravitational acceleration. Default is 9.81 m/s².

``SE_FACTOR: (float)``

The shortening of element length when automatic generation of constant head lines and intersections. Default is 1.0.

``MAX_ITERATIONS: (int)``

The maximum number of iterations for the solver. Default is 50.

``MAX_ERROR: (float)``

The maximum error allowed in the solver. Default is 1e-6.

``MAX_COEF: (int)``

The maximum number of coefficients for a single element. Default is 150.

``COEF_INCREASE: (int)``

The value for which to increase the number of coefficients in the solver when the COEF_RATION is not met. Default is 5.

``COEF_RATIO: (float)``

The ration betwenn the maximum of the first two coefficients nad the last two of an element. Default is 0.05.

``MAX_ELEMENTS: (int)``

Maximum number of elements in a single fracture. Default is 150.

``NCOEF: (int)``

Default number of coefficients for a single element. Default is 5.

``NINT: (int)``

Default number of integration points for a single element. Default is 10.

``NUM_THREADS: (int)``

The number of threads to use for parallel processing. If set to -1, it will use all available threads.


LOGGING
-------
``LOG_LEVEL:``

This sets the logging level for the `andfn` package. Possible values are DEBUG, INFO, WARNING, ERROR, CRITICAL, and PROGRESS (which is a custom level for progress updates).

``LOG_FILE:``

This sets the file where logs will be written. If not set, logs will be written to the console.