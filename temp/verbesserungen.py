from column.utils import _convert_memmap

_convert_memmap("/home/daniel/Schreibtisch/test.npy", 
                "/home/daniel/Schreibtisch/bla")

#%%

import numpy as np

distmat = np.load("/home/daniel/Schreibtisch/bla/distmat.npy", mmap_mode="r")
index = np.load("/home/daniel/Schreibtisch/bla/index.npy", mmap_mode="r")


#distmat = distmat[:,:500]
#index = index[:,:500]

#%%

from column.utils import surrogate_maps
from nibabel.freesurfer.io import read_label
from fmri_tools.io import read_mgh


label_in = "/home/daniel/Schreibtisch/data/lh.v1.label"
data_in = "/home/daniel/Schreibtisch/data/lh.Z_all_left_right_GE_EPI1_layer_5.mgh"
data,_,_ = read_mgh(data_in)
label = read_label(label_in)
file_dist = "/home/daniel/Schreibtisch/test.npy"
file_out = "/home/daniel/Schreibtisch/sdfsdf.npy"

AAA = surrogate_maps(data[label], file_dist, file_out, n=10)

#%%

from fmri_tools.io import write_mgh

res = np.zeros(len(data))
res[label] = AAA[5,:]

write_mgh("/home/daniel/Schreibtisch/menno.mgh", res)
