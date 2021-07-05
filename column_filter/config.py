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
sigma = 1.0  # standard deviation of gaussian envelope
n_lambda = 16  # number of wavelengths
n_theta = 12  # number of orientation steps
lambda_min = 1.0  # smallest wavelength
lambda_max = 8.0  # largest wavelength
hull_min = 0.1  # cutoff value of gaussian hull

wavelet_params = {'sigma': sigma,
                  'lambda': np.logspace(np.log(lambda_min),
                                        np.log(lambda_max),
                                        n_lambda, base=np.e),
                  'ori': np.linspace(0, np.pi-np.pi/n_theta, n_theta),
                  'hull': hull_min,
                  }
