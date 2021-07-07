# -*- coding: utf-8 -*-
"""I/O functions."""

import os
import numpy as np
import nibabel as nb
from nibabel.freesurfer.mghformat import MGHHeader
from nibabel.freesurfer.io import read_geometry, read_label

__all__ = ['load_mesh', 'load_overlay', 'load_roi', 'load_mmap', 'save_overlay']


def load_mesh(file_in):
    """Load vertex coordinates and associated faces of surface mesh from file.
    Currently, only meshs in freesurfer file format are supported.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    ValueError
        If `file_in` is not a string.

    Returns
    -------
    dict
        Dictionary collecting the output under the following keys

        * vtx : np.ndarray, shape=(N,3)
            Vertex coordinates.
        * fac : np.ndarray, shape=(M,3)
            Vertex indices of each triangle.

    """

    # check file name
    if not isinstance(file_in, str):
        raise ValueError("File name must be a string!")

    surf = read_geometry(file_in)

    return {'vtx': surf[0],
            'fac': surf[1],
            }


def load_overlay(file_in):
    """Load a surface overlay from file which contains one value per vertex.
    Currently, only overlays in freesurfer mgh format are supported.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    ValueError
        If `file_in` is not a string or has a file extension which is not
        supported.

    Returns
    -------
    dict
        Dictionary collecting the output under the following keys

        * arr : np.ndarray, shape=(N,)
            Overlay data.
        * affine : np.ndarray, shape=(4,4)
            Affine transformation matrix.
        * header : MGHHeader
            Header information.

    """

    # check filename
    if isinstance(file_in, str):
        if not file_in.endswith("mgh"):
            raise ValueError("Currently supported file format is mgh.")
    else:
        raise ValueError("File name must be a string!")

    # get header
    header = nb.load(file_in).header
    affine = nb.load(file_in).affine

    # get data
    arr = nb.load(file_in).get_fdata()
    arr = np.squeeze(arr)

    return {'arr': arr,
            'affine': affine,
            'header': header,
            }


def load_roi(file_in):
    """Load vertex indices in region-of-interest (ROI) from file. Currently,
    only freesurfer label files are supported.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    ValueError
        If `file_in` is not a string.

    Returns
    -------
    np.ndarray, shape=(N,)
        Array of vertex indices in ROI.

    """

    # check filename
    if not isinstance(file_in, str):
        raise ValueError("File name must be a string!")

    return np.sort(read_label(file_in))


def load_mmap(file_in):
    """Load a memory-mapped array.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    ValueError
        If `file_in` is not a string or has wrong file extension.

    Returns
    -------
    np.memmap
        Memory-mapped array.

    """

    # check filename
    if isinstance(file_in, str):
        if not file_in.endswith("npy"):
            raise ValueError("Currently supported file format is npy.")
    else:
        raise ValueError("File name must be a string!")

    return np.load(file_in, mmap_mode='r')


def save_overlay(file_out, arr, affine=None, header=None):
    """Write a 1D array as surface overlay to disk. Currently, only overlays in
    freesurfer mgh format are supported.

    Parameters
    ----------
    file_out : str
        File name of output file.
    arr : np.ndarray, shape=(N,)
        Overlay data.
    affine : np.ndarray, shape=(4,4), optional
        Affine transformation matrix.
    header : MGHHeader, optional
        Header information.

    Raises
    ------
    ValueError
        If `file_out` is not a string or has a file extension which is not
        supported.

    Returns
    -------
    None.

    """

    # check filename
    if isinstance(file_out, str):
        if not file_out.endswith("mgh"):
            raise ValueError("Currently supported file format is mgh.")
    else:
        raise ValueError("Filename must be a string!")

    # make output folder
    dir_out = os.path.dirname(file_out)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # add empty dimensions
    arr = np.expand_dims(arr, axis=1)
    arr = np.expand_dims(arr, axis=1)

    if affine is None:
        affine = np.eye(4)

    if header is None:
        header = MGHHeader()

    # write output
    output = nb.Nifti1Image(arr, affine, header)
    nb.save(output, file_out)
