# -*- coding: utf-8 -*-
"""Default parameters."""

import multiprocessing
import numpy as np


# Constants
# ------------------------------------------------------------------------------
# For parallel computation, the maximum number of available CPUs is taken.
NUM_CORES = multiprocessing.cpu_count()

# Wavelet parameters (Gabor filter)
# ------------------------------------------------------------------------------
# Parameter search space for wavelet fitting.
wavelet_params = {'sigma': 5.0,  # sigma of gaussian
                  'lambda': np.linspace(1.0, 5.0, 3),  # wavelength of cosine
                  'ori': np.linspace(0, 2 * np.pi, 3),  # direction of cosine
                  'phase': np.linspace(0, 2 * np.pi, 3),  # phase of cosine
                  }
