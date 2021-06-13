# -*- coding: utf-8 -*-

import os
import multiprocessing
import numpy as np
from numpy.linalg import norm
from surfdist.analysis import dist_calc
from joblib import Parallel, delayed


def _rotation_matrix(f, t):
    """Rotation matrix.

    This helper function computes a 3x3 rotation matrix that rotates a unit
    vector f into another unit vector t. The algorithm is taken from [1] and
    uses the implementation found in [2].

    Parameters
    ----------
    f : List[float]
        source unit vector.
    t : List[float]
        target unit vector.

    Returns
    -------
    rot : (3,3) np.ndarray
        Rotation matrix.

    References
    -------
    .. [1] Moeller, T, et al. Efficiently building a matrix to rotate one vector
    to another, Journal of Graphics Tools 4(4), 1--3 (2018).
    .. [2] https://math.stackexchange.com/questions/180418/calculate-rotation-
    matrix-to-align-vector-a-to-vector-b-in-3d

    """

    if not np.isclose(np.linalg.norm(f), 1):
        raise ValueError("Source vector must be a unit vector!")

    if not np.isclose(np.linalg.norm(t), 1):
        raise ValueError("Target vector must be a unit vector!")

    v = np.cross(f, t)
    c = np.dot(f, t)
    h = (1 - c) / (1 - c ** 2)

    vx, vy, vz = v
    rot = np.array([[c + h * vx ** 2, h * vx * vy - vz, h * vx * vz + vy],
                    [h * vx * vy + vz, c + h * vy ** 2, h * vy * vz - vx],
                    [h * vx * vz - vy, h * vy * vz + vx, c + h * vz ** 2]])

    return rot


def _angle_between_vectors(v1, v2, n):
    """Angle between vectors.

    This helper function computes the angle between two 3D vectors in the range
    (-pi, +pi].

    Parameters
    ----------
    v1 : List[float]
        Vector 1.
    v2 : List[float]
        Vector 2.
    n : List[float]
        Normal vector.

    Returns
    -------
    ang : float
        Signed angle between both vectors in radians.

    """

    # compute angle in the range [0, +pi]
    c = np.cross(v1, v2)
    d = np.dot(v1, v2)
    ang = np.arctan2(np.linalg.norm(c), d)

    # set sign
    if np.dot(n, c) < 0:
        ang *= -1

    return ang
















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
    counter2 = np.floor((i - 1) / loop_length * 100)
    if counter != counter2:
        print("Loop status: " + str(counter) + " %")

    del D























"""
stuff to compute gradient
"""

def _face_area(v, f):
    """
    Helper function to compute face areas.
    """

    # indexed view into the vertex array
    tris = v[f]

    A = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    A = norm(A, axis=1)
    A /= 2

    return A


def _face_normal(v, f):
    """
    Helper function to compute face-wise normals.
    """

    # indexed view into the vertex array
    tris = v[f]

    # calculate the normal for all triangles by taking the cross product of
    # the vectors v1-v0 and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n_norm = norm(n, axis=1)

    # normalize
    n[:, 0] /= n_norm
    n[:, 1] /= n_norm
    n[:, 2] /= n_norm

    return n


def _f2v(f, gf, a):
    """
    Helper function to transform face- to vertex-wise expressions.
    """
    nv = np.max(f) + 1  # number of vertices
    nf = len(f)  # number of faces
    gv = np.zeros((nv, 3))
    magn = np.zeros(nv)
    for i in range(nf):
        gv[f[i, 0], :] += a[i] * gf[i, :]
        gv[f[i, 1], :] += a[i] * gf[i, :]
        gv[f[i, 2], :] += a[i] * gf[i, :]

        magn[f[i, 0]] += a[i]
        magn[f[i, 1]] += a[i]
        magn[f[i, 2]] += a[i]

    gv[:, 0] /= magn
    gv[:, 1] /= magn
    gv[:, 2] /= magn

    return gv


def gradient(vtx, fac, arr_scalar, normalize=True):
    """ Gradient

    This function computes the vertex-wise gradient of a scalar field sampled
    on a triangular mesh. The calculation is taken from [1].

    Parameters
    ----------
    vtx : ndarray
        Array of vertex coordinates.
    fac : ndarray
        Corresponding faces.
    arr_scalar : ndarray
        Scalar field values per vertex.
    normalize : bool, optional
        Normalize gradient vectors. The default is True.

    Returns
    -------
    gv : ndarray
        Vertex-wise gradient vector.
    gv_magn : ndarray
        Vertex-wise gradient magnitude.

    References
    -------
    .. [1] Mancinelli, C. et al. Gradient field estimation on triangle meshes.
    Eurographics Proceedings (2018).

    Notes
    -------
    created by Daniel Haenelt
    Date created: 25-08-2020
    Last modified: 19-11-2020

    """

    # face areas and normals
    arr_A = _face_area(vtx, fac)
    arr_n = _face_normal(vtx, fac)

    # face-wise gradient
    gf_ji = arr_scalar[fac[:, 1]] - arr_scalar[fac[:, 0]]
    gf_ki = arr_scalar[fac[:, 2]] - arr_scalar[fac[:, 0]]

    v_ik = vtx[fac[:, 0], :] - vtx[fac[:, 2], :]
    v_ji = vtx[fac[:, 1], :] - vtx[fac[:, 0], :]

    # rotate
    v_ik_rot = np.cross(v_ik, arr_n)
    v_ji_rot = np.cross(v_ji, arr_n)

    gf = np.zeros_like(fac).astype(float)
    gf[:, 0] = (gf_ji * v_ik_rot[:, 0] + gf_ki * v_ji_rot[:, 0]) / (2 * arr_A)
    gf[:, 1] = (gf_ji * v_ik_rot[:, 1] + gf_ki * v_ji_rot[:, 1]) / (2 * arr_A)
    gf[:, 2] = (gf_ji * v_ik_rot[:, 2] + gf_ki * v_ji_rot[:, 2]) / (2 * arr_A)

    # vertex-wise gradient
    gv = _f2v(fac, gf, arr_A)
    gv_magn = norm(gv, axis=1)

    # normalize
    if normalize:
        gv_norm = norm(gv, axis=1)
        gv_norm[gv_norm == 0] = np.nan

        gv[:, 0] /= gv_norm
        gv[:, 1] /= gv_norm
        gv[:, 2] /= gv_norm
        pole = np.argwhere(np.isnan(gv))[:, 0]
        gv[pole, :] = 0

    return gv, gv_magn
