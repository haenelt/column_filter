# Column filter

<p align="center">
  <img src="https://github.com/haenelt/column_filter/blob/main/img/odc.gif?raw=true" width=75% height=75% alt="Illustration of GBB"/>
</p>

Cortical columns are often thought as the fundamental building blocks for cortical information processing. Recent advances in ultrahigh field functional magnetic resonance imaging (fMRI) allow the mapping of cortical column in living humans. However, fMRI suffers from various noise sources which degrade the specificity of the signal and challenges the visualization of fine-grained cortical structures.

This package aims to uncover the underlying repetitive structure using a wavelet approach. A filter bank consisting of complex-valued Gabor wavelets with different spatial frequencies and orientations is applied to fMRI contrasts sampled on a surface mesh. This is similar to [[1]](#1). However, this implementation works completely on the curved irregular triangle mesh. This is advantageous since no spatial distortion have to be applied to the data, e.g. by flattening the mesh. In brief, for each vertex in a region of interest, the wavelet which correlates most with the underlying structure is found. From the filtered response, the preferred spatial frequency can be computed. The orientation information is lost because of the irregular mesh structure.

The figure above shows human ocular dominance columns (contrast: left eye response > right eye response color coded yellow > blue) in the primary visual cortex (V1) acquired at 7 T. The data is shown for one hemisphere on an inflated surface mesh which only covers the occipital lobe. Please not that while the data is visualized on an inflated mesh, the filtering procedure was performed on the original curved mesh. The repetitive structure can be clearly seen after filtering. The column width (half the wavelength expectation value) is 1.5 mm in line with the known size of human ocular dominance columns. 

Another example in the *img* folder shows thin stripes in the secondary visual cortex which were mapped by exploiting their color sensitivity. The expectation value of column width is 2.77 mm which again is in line with known sizes of human stripes.

This method can potentially be used to explore other cortical areas for fine-grained repetitive structures.

## Installation
I recommend to use `Miniconda` to create a new python environment with `python=3.8`. To install the package, clone the repository and run the following line from the directory in which the repository was cloned

```shell
python setup.py install
```

After the installation you should be able to import the package with `import column_filter`.

## Run the package
The package can be executed from the terminal with the following command.

```shell
python -m column_filter <args>
```

An overview of obligatory and optional arguments can be printed out in the terminal with

```shell
python -m column_filter -h
```

## Example data
Data to reproduce the results which are shown in the figure above can be downloaded from my personal google drive. This should also be checked to know which files are expected.

```python
from column_filter.data import download_data
dir_out = ""  # directory where downloaded files are saved
download_data(dir_out)
```

## Some snippets

###  Filter data
```python
import numpy as np
from column_filter.io import load_roi, load_mesh, load_mmap, load_overlay
from column_filter.filt import Filter

surf_in = ""  # input surface mesh
label_in = ""  # input region of interest
dist_in = ""  # input distance matrix
arr_in = ""  # input overlay
file_out = ""  # output overlay

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)
arr = load_overlay(arr_in)['arr']

filter_bank = Filter(surf['vtx'], surf['fac'], label, dist)
_ = filter_bank.fit(arr, file_out=file_out)
```

### Visualize Gabor wavelet
```python
import numpy as np
from column_filter.io import load_roi, load_mesh, save_overlay, load_mmap
from column_filter.filt import Filter

# calculate distance
surf_in = ""  # input surface mesh
label_in = ""  # input region of interest
dist_in = ""  # input distance matrix
file_out = ""  # output overlay

sigma = 1.0  # standard deviation of gaussian hull
sf = 2.0  # spatial frequency
i = 4219  # vertex index in region of interest

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)

filter_bank = Filter(surf['vtx'], surf['fac'], label, dist)
coords = filter_bank.get_coordinates(i)
res = filter_bank.generate_wavelet(coords[0], coords[1], sigma, sf, 0, 0.1)

# save gabor wavelet as mgh overlay
save_overlay(file_out, np.real(res))
```

## Acknowledgements
I thank [Denis Chaimow](https://www.cbs.mpg.de/person/dchaimow/374227) who had the initial idea which finally led to this project. 

## References
<a id="1">[1]</a> de Hollander, G. D., van der Zwaag, W., Qian, C., Zhang, P. &  Knapen, T. Ultra-high field fMRI reveals origins of feedforward and feedback activity within laminae of human ocular dominance columns. *NeuroImage* **228** (2021). 

## Contact
If you have questions, problems or suggestions regarding the column_filter package, please feel free to contact [me](mailto:daniel.haenelt@gmail.com).
