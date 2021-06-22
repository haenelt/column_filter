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
sigma = 2.0  # standard deviation of gaussian envelope
n_lambda = 25  # number of spatial frequency steps
n_theta = 12  # number of orientation steps
lambda_min = 1.0  # smallest spatial frequency
lambda_max = 5.0  # largest spatial frequency

wavelet_params = {'sigma': sigma,
                  'lambda': np.linspace(lambda_min, lambda_max, n_lambda),
                  'ori': np.linspace(0, np.pi-np.pi/n_theta, n_theta),
                  }
