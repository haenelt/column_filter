# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy.linalg import norm
from surfdist.analysis import dist_calc
from joblib import Parallel, delayed
from .mesh import Mesh
from .config import NUM_CORES
from tqdm import tqdm


__all__ = ['Filter', 'dist_matrix']


class Filter:

    def __init__(self, vtx, fac, roi, arr, dist):
        self.vtx = vtx
        self.fac = fac
        self.roi = roi
        self.arr = arr
        self.dist = dist

    def gradient(self, arr_s):
        """This function computes the vertex-wise gradient of a scalar field sampled
        on a triangular mesh. The calculation is taken from [1].

        Parameters
        ----------
        arr_s : ndarray
            Scalar field values per vertex.

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

        """

        mesh = Mesh(self.vtx, self.fac)
        arr_a = mesh.face_areas
        arr_n = mesh.face_normals

        # face-wise gradient
        gf_ji = arr_s[self.fac[:, 1]] - arr_s[self.fac[:, 0]]
        gf_ki = arr_s[self.fac[:, 2]] - arr_s[self.fac[:, 0]]

        v_ik = self.vtx[self.fac[:, 0], :] - self.vtx[self.fac[:, 2], :]
        v_ji = self.vtx[self.fac[:, 1], :] - self.vtx[self.fac[:, 0], :]

        # rotate
        v_ik_rot = np.cross(v_ik, arr_n)
        v_ji_rot = np.cross(v_ji, arr_n)

        gf = np.zeros_like(self.fac).astype(float)
        gf[:, 0] = (gf_ji * v_ik_rot[:, 0] + gf_ki * v_ji_rot[:, 0]) / (
                    2 * arr_a)
        gf[:, 1] = (gf_ji * v_ik_rot[:, 1] + gf_ki * v_ji_rot[:, 1]) / (
                    2 * arr_a)
        gf[:, 2] = (gf_ji * v_ik_rot[:, 2] + gf_ki * v_ji_rot[:, 2]) / (
                    2 * arr_a)

        # vertex-wise gradient
        gv = self._f2v(gf, arr_a)
        gv_magn = norm(gv, axis=1)

        # normalize
        gv = self._normalize(gv)

        pole = np.argwhere(np.isnan(gv))[:, 0]
        gv[pole, :] = 0

        return gv, gv_magn

    def wavelet(self):
        pass

    def fit(self):
        pass

    def _convolution(self):
        pass

    def _f2v(self, gf, a):
        """Helper function to transform face- to vertex-wise expressions."""
        gv = np.zeros((len(self.vtx), 3))
        magn = np.zeros(len(self.vtx))
        for i, f in enumerate(self.fac):
            gv[f[0], :] += a[i] * gf[i, :]
            gv[f[1], :] += a[i] * gf[i, :]
            gv[f[2], :] += a[i] * gf[i, :]

            magn[f[0]] += a[i]
            magn[f[1]] += a[i]
            magn[f[2]] += a[i]

        gv[:, 0] /= magn
        gv[:, 1] /= magn
        gv[:, 2] /= magn

        return gv

    @staticmethod
    def _normalize(arr):
        """Normalize a numpy array of shape=(n,3) along axis=1."""
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        lens[lens == 0] = np.nan
        res = np.zeros_like(arr)
        res[:, 0] = arr[:, 0] / lens
        res[:, 1] = arr[:, 1] / lens
        res[:, 2] = arr[:, 2] / lens
        res[~np.isfinite(res)] = 0

        return res

    @staticmethod
    def _rotation_matrix(f, t):
        """This helper function computes a 3x3 rotation matrix that rotates a
        unit vector f into another unit vector t. The algorithm is taken from
        [1] and uses the implementation found in [2].

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

    @staticmethod
    def _angle_between_vectors(v1, v2, n):
        """Helper function computes the angle between two 3D vectors in the
        range (-pi, +pi].

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


def dist_matrix(file_out, vtx, fac, roi):
    """Create a memory-mapped file which contains the distance matrix from a
    connected region of interest on a triangle surface mesh. The computation of
    matrix elements takes a while. Therefore, joblib is used to execute the
    computation in parallel.

    Parameters
    ----------
    file_out : str
        File name of output file.
    vtx : (N,3) np.ndarray
        Vertex coordinates.
    fac : (M,3) np.ndarray
        Vertex indices of each triangle.
    roi : (U,) np.ndarray
        Array of vertex indices in ROI.

    Raises
    ------
    FileExistsError
        If `file_out` already exists.

    Returns
    -------
    None.

    """

    # check if file already exists
    if os.path.exists(file_out):
        raise FileExistsError("File already exists!")

    # create output folder
    dir_out = os.path.dirname(file_out)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # create binary file
    d = np.lib.format.open_memmap(file_out,
                                  mode='w+',
                                  dtype=np.float32,
                                  shape=(len(roi), len(roi)),
                                  )

    # fill distance matrix
    Parallel(n_jobs=NUM_CORES)(
        delayed(_map_array)(
            i,
            d,
            vtx,
            fac,
            roi) for i in tqdm(range(len(roi)))
    )


def _map_array(i, d, vtx, fac, roi):
    """Helper function to compute nearest geodesic distances from index roi[n]
    to all other indices in roi. Distances are written into row n and column n
    of the distance matrix.

    Parameters
    ----------
    i : int
        Element in roi array.
    d : (N,N) np.ndarray
        Distance matrix.
    vtx : (N,3) np.ndarray
        Vertex coordinates.
    fac : (M,3) np.ndarray
        Vertex indices of each triangle.
    roi : (U,) np.ndarray
        Array of vertex indices in ROI.

    Returns
    -------
    None.

    """

    # compute geodesic distances
    tmp = dist_calc((vtx, fac), roi, roi[i])
    d[i:, i] = tmp[roi[i:]]
    d[i, i:] = tmp[roi[i:]]

    del d