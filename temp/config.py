# -*- coding: utf-8 -*-

import numpy as np

# BrainSMASH default parameters for :class:`brainsmash.mapgen.sampled.Sampled`.
# See there to get more information about parameters.
brainsmash_params = {}
brainsmash_params['ns'] = 1000
brainsmash_params['pv'] = 25
brainsmash_params['nh'] = 10
brainsmash_params['knn'] = 100
brainsmash_params['b'] = None
brainsmash_params['deltas'] = np.arange(0.1,1.0,0.1) 
brainsmash_params['kernel'] = "exp"
brainsmash_params['resample'] = True
brainsmash_params['verbose'] = True
brainsmash_params['seed'] = None
brainsmash_params['n_jobs'] = 1

