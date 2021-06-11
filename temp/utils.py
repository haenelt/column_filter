# -*- coding: utf-8 -*-

import os
import multiprocessing
import shutil as sh
import numpy as np
from brainsmash.mapgen.sampled import Sampled
from surfdist.analysis import dist_calc
from joblib import Parallel, delayed
from column.config import brainsmash_params


def dist_matrix(file_out, vtx, fac, label):    
    """Dist matrix.

    This function creates a memory-mapped file which contains the distance
    matrix from a connected region of interest on a triangular surface mesh.
    The computation of matrix elements takes a while. Therefore, joblib is used
    to execute the computation on all available CPUs in parallel.

    Parameters
    ----------
    file_out : str
        Filename of memory-mapped distance matrix.
    vtx : (nvtx,3) np.ndarray
        Array of vertex points.
    fac : (nfac,3) np.ndarray
        Array of corresponging faces.
    label : (N,) np.ndarray
        Array of label indices.

    Raises
    ------
    FileExistsError
        If `file_out` already exists.

    Returns
    -------
    None.

    """
    
    # number of cores
    num_cores = multiprocessing.cpu_count()        
    
    # check if file already exists
    if os.path.exists(file_out):
        raise FileExistsError("File already exists!")
    
    # create output folder
    if not os.path.exists(os.path.dirname(file_out)):
        os.makedirs(os.path.dirname(file_out))

    # create binary file
    D = np.lib.format.open_memmap(file_out,
                                  mode='w+', 
                                  dtype=np.float32, 
                                  shape=(len(label), len(label)),
                                  )

    # fill distance matrix
    Parallel(n_jobs=num_cores)(
        delayed(_map_array)(
            i,
            D,
            label, 
            vtx, 
            fac) for i in range(len(label))
        )


def surrogate_maps(data, file_dist, file_out="", n=1000, **params):
    """Surrogate maps

    This function uses the BrainSMASH package [1] to compute n surrogate maps.
    These maps consist of randomly distributed map values with matched spatial 
    autocorrelation to the target map.    

    Parameters
    ----------
    data : (N,) np.ndarray
        Target map.
    file_dist : str
        Filename of memory-mapped distance matrix.
    file_out : str, optional
        Filename of the output file containing the data of all generated 
        surrogate maps. The default is "".
    n : int, optional
        Number of generated surrogate maps. The default is 1000.
    **brainsmash_params : dict
        Keyword arguments for :class:`brainsmash.mapgen.sampled.Sampled`. If no
        arguments are set, default values are used which are defined in
        `column.config`.

    Raises
    ------
    ValueError
        If 'file_out' is defined but without npy file extension.

    Returns
    -------
    surrogates : (n, N) np.ndarray
        Data of all generate surrogate maps.
    
    References
    -------
    .. [1] Burt, J, et al. Generative modeling of brain maps with spatial 
    autocorrelation, Neuroimage 220, 1--17 (2020).

    """
    
    # set default arguments if no brainsmash parameters are given
    params = { **brainsmash_params, **params }
    
    # output filename
    if file_out and not file_out.endswith('npy'):
        raise ValueError("Output file must have npy file extension!")   
        
    # create temporary folder
    tmp = np.random.randint(0, 10, 10)
    tmp = ''.join(str(i) for i in tmp)
    path_output = os.path.dirname(file_out) if file_out else os.getcwd()
    path_tmp = os.path.join(path_output, tmp)
    
    # convert distance matrix
    mmap = _convert_memmap(file_dist, path_tmp)
    
    # create instance and create random surrogates
    sampled = Sampled(data, mmap['D'], mmap['index'], **params)
    surrogates = sampled(n)
    
    # write output
    if file_out:
        np.save(file_out, surrogates)
    
    # delete temporary folder
    sh.rmtree(path_tmp, ignore_errors=True)   
    
    return surrogates


def _map_array(i, D, label, vtx, fac):
    """Map array.

    This helper function computes nearest geodesic distances from index 
    label[n] to all other indices in the label array and write these distances 
    into row n and column n of the distance matrix.

    Parameters
    ----------
    i : int
        Position within label array.
    D : (N,N) np.ndarray
        Distance matrix.
    label : (N,) np.ndarray
        Array of label indices.
    vtx : (nvtx,3) np.ndarray
        Array of vertex points.
    fac : (nfac,3) np.ndarray
        Array of corresponding faces.

    Returns
    -------
    None.

    """
    
    # compute geodesic distances
    tmp = dist_calc((vtx, fac), label, label[i])
    D[i:, i] = tmp[label[i:]]
    D[i, i:] = tmp[label[i:]]  
   
    # print current status
    loop_length = len(label)  
    counter = np.floor(i / loop_length * 100)
    counter2 = np.floor((i-1) / loop_length * 100)
    if counter != counter2:
        print("Loop status: "+str(counter)+" %")
    
    del D


def _convert_memmap(file_dist, path_output):
    """Convert memmap.
    
    This helper function sorts a memory-mapped large distance matrix for 
    memory-efficient data retrieval. In addition to the sorted distance matrix, 
    an index file is saved which contains the original location of matrix
    elements in the i-th row. The function is adapted from 
    brainsmash.mapgen.memmap.txt2memmap and allows memory-mapped binary files 
    as input.

    Parameters
    ----------
    file_dist : str
        Filename of memory-mapped distance matrix.
    path_output : str
        Path where output is written.

    Raises
    ------
    ValueError
        If 'file_dist' does not contain a 2D array with square shape.

    Returns
    -------
    dict
        Absolute paths to the corresponding binary files on disk. Keys are 'D'
        and 'index'.

    """

    # make output folder
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # load distfile
    D = np.load(file_dist, mmap_mode='r')
    
    # check if matrix is square
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix has no square shape.")    
    
    # Build memory-mapped arrays
    npydfile = os.path.join(path_output, "distmat.npy")
    npyifile = os.path.join(path_output, "index.npy")
    #fpd = np.lib.format.open_memmap(
    #    npydfile, mode='w+', dtype=np.float32, shape=(len(D), len(D)))
    #fpi = np.lib.format.open_memmap(
    #    npyifile, mode='w+', dtype=np.int32, shape=(len(D), len(D)))

    #idx = np.arange(len(D))
    #for il, l in enumerate(D):  # Loop over rows
    #    d = l[idx]
    #    sort_idx = np.argsort(d)
    #    fpd[il, :] = d[sort_idx]  # sorted row of distances
    #    fpi[il, :] = sort_idx  # sort indexes

    n = 500
    fpd = np.lib.format.open_memmap(
        npydfile, mode='w+', dtype=np.float32, shape=(len(D), n))
    fpi = np.lib.format.open_memmap(
        npyifile, mode='w+', dtype=np.int32, shape=(len(D), n))

    idx = np.arange(len(D))
    for il, l in enumerate(D):  # Loop over rows
        d = l[idx]
        sort_idx = np.argsort(d)
        sort_idx = sort_idx[:500]
        fpd[il, :] = d[sort_idx]  # sorted row of distances
        fpi[il, :] = sort_idx  # sort indexes

    # flush memoray changes to disk
    del fpd
    del fpi

    return {'D': npydfile, 'index': npyifile}
