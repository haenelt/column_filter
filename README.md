# Column filter

<p align="center">
  <img src="https://github.com/haenelt/GBB/blob/master/gbb_logo.gif?raw=true" width=75% height=75% alt="Illustration of GBB"/>
</p>

Cortical columns are often thought as the fundamental building blocks for cortical information processing (but see).

[[1]](#1)

## Installation
I recommend to use `Anaconda` to create a new python environment with `python=3.8`.

- clone this repository
- run the following line from the directory in which the repository was cloned
```shell
python setup.py install
```
After the installation you should be able to import the package with `import column_filter`.


## Example

```python
import numpy as np
from column_filter.io import load_roi, load_mesh, load_mmap, load_overlay
from column_filter.filt import Filter

# calculate distance
surf_in = "/home/daniel/Schreibtisch/data2/lh.cortex"
label_in = "/home/daniel/Schreibtisch/data2/lh.v1.label"
dist_in = "/home/daniel/Schreibtisch/data2/dist.npy"
arr_in = "/home/daniel/Schreibtisch/data2/lh.contrast.mgh"

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)
arr = load_overlay(arr_in)['arr']
vtx = surf['vtx']
fac = surf['fac']

arr = np.vstack((arr, arr)).T

filter = Filter(vtx, fac, label, dist)
bla = filter.fit(arr, file_out="/home/daniel/Schreibtisch/test.parquet")
```

```python
import numpy as np
from column_filter.io import load_roi, load_mesh, save_overlay, load_mmap
from column_filter.filt import Filter
from column_filter.config import wavelet_params

# calculate distance
surf_in = "/home/daniel/Schreibtisch/data2/lh.cortex"
label_in = "/home/daniel/Schreibtisch/data2/lh.v1.label"
dist_in = "/home/daniel/Schreibtisch/data2/dist.npy"

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)
vtx = surf['vtx']
fac = surf['fac']

filt = Filter(vtx, fac, label, dist)
coords = filt.get_coordinates(4219)
sf = wavelet_params['lambda']

for i in sf:
    A = filt.generate_wavelet(coords[0], coords[1],
                              1,  # sigma
                              i,  # length
                              0, 0.1)

    A = np.real(A)
    save_overlay("/home/daniel/Schreibtisch/bla_" + str(i) + "_.mgh", A)
```

## Acknowledgements
I thank [Denis Chaimow](https://www.cbs.mpg.de/person/dchaimow/374227) who had the initial idea which finally led to this project. 

## References
<a id="1">[1]</a> Brodmann K, Vergleichende Lokalisationslehre der Grosshirnrinde, Barth-Verlag (1909).<br/>

## Contact
If you have questions, problems or suggestions regarding the column_filter package, please feel free to contact [me](mailto:daniel.haenelt@gmail.com).
