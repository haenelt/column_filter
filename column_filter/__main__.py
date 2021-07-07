# -*- coding: utf-8 -*-
"""Python package for filtering cortical columns on a surface mesh. Execute via
the command line with python -m column_filter <args>."""

import argparse
import numpy as np
import column_filter
from .io import load_roi, load_mesh, load_mmap, load_overlay, save_overlay
from .config import wavelet_params
from .util import linear_scale, log_scale
from .filter import Filter


# description
parser_description = "This program will apply a filter bank to a scalar " \
                     "field defined on a curved triangle mesh. The filter " \
                     "bank consists of complex-valued Gabor wavelets with " \
                     "various wavelengths and orientations. A freesurfer " \
                     "overlay containing the filtered scalar field is " \
                     "written to disk. Optionally, a pandas dataframe with " \
                     "computed filter parameters can be saved as well."

m_help = "file name of input surface mesh. The mesh is expected to be in " \
         "freesurfer file format."
r_help = "file name of input region of interest. The file is expected to be " \
         "in freesurfer label format."
d_help = "file name of input distance matrix. The file is expected to be a " \
         "memory-mapped numpy array which can be computed beforehand with " \
         "the function column_filter.filter.dist_matrix()."
i_help = "file name of input overlay. The file is expected to be a " \
         "freesurfer overlay (*.mgh)."
o_help = "file name of written output overlay (*.mgh). The file contains the " \
         "filtered input overlay. An already existing file will be overwritten."

# parse arguments from command line
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument('mesh_in', metavar='mesh_in', type=str, help=m_help)
parser.add_argument('roi_in', metavar='roi_in', type=str, help=r_help)
parser.add_argument('dist_in', metavar='dist_in', type=str, help=d_help)
parser.add_argument('overlay_in', metavar='overlay_in', type=str, help=i_help)
parser.add_argument('overlay_out', metavar='overlay_out', type=str, help=o_help)

df_help = "file name to store filter parameters as pandas dataframe"
s_help = "standard deviation of Gaussian hull (default: " + \
         str(wavelet_params['sigma']) + ")"
hmin_help = "cutoff threshold of Gaussian hull (default: " + \
            str(wavelet_params['hull']) + ")"
lmin_help = "minimum filter wavelength (default: " + \
            str(wavelet_params['lambda'][0]) + ")"
lmax_help = "maximum filter wavelength (default: " + \
            str(wavelet_params['lambda'][-1]) + ")"
nl_help = "number of filter wavelengths (default: " + \
          str(len(wavelet_params['lambda'])) + ")"
nt_help = "number of filter orientations (default: " + \
          str(len(wavelet_params['ori'])) + ")"

parser.add_argument('-df', '--data_frame', type=str, help=df_help)
parser.add_argument('-s', '--sigma', type=float, help=s_help)
parser.add_argument('-hmin', '--hull_min', type=float, help=hmin_help)
parser.add_argument('-lmin', '--lambda_min', type=float, help=lmin_help)
parser.add_argument('-lmax', '--lambda_max', type=float, help=lmax_help)
parser.add_argument('-nl', '--n_lambda', type=int, help=nl_help)
parser.add_argument('-nt', '--n_theta', type=int, help=nt_help)
args = parser.parse_args()

# run
print("-----------------------------------------------------------------------")
print("Column filter "+"(v"+str(column_filter.__version__)+")")
print("author: "+str(column_filter.__author__))
print("-----------------------------------------------------------------------")

# sigma
if args.sigma is not None:
    wavelet_params['sigma'] = args.sigma

# hull
if args.hull_min is not None:
    wavelet_params['hull'] = args.hull_min

# wavelength
if all([args.lambda_min, args.lambda_max, args.n_lambda]):
    wavelet_params['lambda'] = log_scale(args.lambda_min,
                                         args.lambda_max,
                                         args.n_lambda)
elif all([args.lambda_min, args.lambda_max]):
    wavelet_params['lambda'] = log_scale(args.lambda_min,
                                         args.lambda_max,
                                         len(wavelet_params['lambda']))
elif all([args.lambda_min, args.n_lambda]):
    wavelet_params['lambda'] = log_scale(args.lambda_min,
                                         wavelet_params['lambda'][-1],
                                         args.n_lambda)
elif all([args.lambda_max, args.n_lambda]):
    wavelet_params['lambda'] = log_scale(wavelet_params['lambda'][0],
                                         args.lambda_max,
                                         args.n_lambda)
elif args.lambda_min:
    wavelet_params['lambda'] = log_scale(args.lambda_min,
                                         wavelet_params['lambda'][-1],
                                         len(wavelet_params['lambda']))
elif args.lambda_max:
    wavelet_params['lambda'] = log_scale(wavelet_params['lambda'][0],
                                         args.lambda_max,
                                         len(wavelet_params['lambda']))
elif args.n_lambda:
    wavelet_params['lambda'] = log_scale(wavelet_params['lambda'][0],
                                         wavelet_params['lambda'][-1],
                                         args.n_lambda)

# orientations
if args.n_theta is not None:
    wavelet_params['ori'] = linear_scale(0,
                                         np.pi-np.pi/args.n_theta,
                                         args.n_theta)

# load data
surf = load_mesh(args.mesh_in)
label = load_roi(args.roi_in)
dist = load_mmap(args.dist_in)
arr = load_overlay(args.overlay_in)['arr']

vtx = surf['vtx']
fac = surf['fac']

# initialize filter
filter_bank = Filter(vtx, fac, label, dist)

# fit
res = filter_bank.fit(arr, file_out=args.data_frame)

# save overlay (real part of best gabor wavelets)
y = np.zeros(len(vtx))
y[label] = res['y_real'].to_numpy()
save_overlay(args.overlay_out, y)

# print expectation value of column width
res_abs = np.sqrt(res['y_real'].to_numpy()**2+res['y_imag'].to_numpy()**2)
res_lambda = res['lambda'].to_numpy()
column_width = np.sum(res_lambda * res_abs) / (2 * np.sum(res_abs))

print("Expected value of column width (half wavelenth): "+str(column_width))
