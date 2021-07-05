# -*- coding: utf-8 -*-
"""Wavelet filtering."""

import os
import itertools
import numpy as np
from numpy.linalg import norm
from surfdist.analysis import dist_calc
from tqdm import tqdm
from joblib import Parallel, delayed
from .mesh import Mesh
from .config import NUM_CORES, wavelet_params
import pandas as pd

__all__ = ['Filter', 'dist_matrix']


class Filter:
    """Filter overlay on triangle surface mesh with a filter bank consisting of
    complex-valued Gabor filters differing in spatial frequency and orientation.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N,3)
        Vertex coordinates.
    fac : np.ndarray, shape=(M,3)
        Vertex indices of each triangle.
    roi : np.ndarray, shape=(U,)
        Array of vertex indices in ROI.
    dist : np.memmap, shape=(U,U)
        Memory-mapped geodesic distances for each ROI vertex to all other
        vertices in ROI.

    Raises
    ------
    ValueError :
        If input parameters have inconsistent shapes.

    """

    def __init__(self, vtx, fac, roi, dist):
        self.vtx = vtx
        self.fac = fac
        self.roi = roi
        self.dist = dist

        self.mesh = Mesh(self.vtx, self.fac)
        self.n = self.mesh.vertex_normals

        self.r = None
        self.ang = None

    def get_coordinates(self, i):
        """Create local polar coordinates system on triangle mesh. Since the
        triangular mesh lacks an intrinsic order, no global coordinate system
        can be used without distorting the mesh. Therefore, the polar angle has
        no global meaning. To compute the corresponding angle for distance
        vertices, the orientation between normal vectors are compared. This is
        a fast method but only results in correct angles in the local
        neighboorhood of the source vertex if the mesh is curved.

        Parameters
        ----------
        i : int
            Array index in ROI.

        Returns
        -------
        r : np.ndarray, shape=(N,)
            Geodesic distance to origin.
        ang : np.ndarray, shape=(N,)
            Polar angle.
        v0 : np.ndarray, shape=(3,)
            Vertex coordinates of the origin of the coordinate system.
        v1 : np.ndarray, shape=(3,)
            Vertex coordinates pointing to one 1-ring neighbor to establish the
            coordinate origin for ang.

        References
        -------
        .. [1] Masci, J. et al. Geodesic convolutional neural networks on
        Riemannian manifolds. Proc. - 2015 Int. Conf. Comput. Vis. Workshop
        ICCVW (2015).

        """

        r = np.zeros(len(self.vtx))
        r[self.roi] = np.nan_to_num(self.dist[i])
        src = self.roi[i]

        g, _ = self._gradient(r)
        v = self.vtx[self.mesh.neighborhood(src)[0], :] - self.vtx[src, :]
        v /= np.linalg.norm(v)

        n_target = np.zeros_like(self.n)
        n_target[:, 0] = self.n[src, 0]
        n_target[:, 1] = self.n[src, 1]
        n_target[:, 2] = self.n[src, 2]

        v_target = np.zeros_like(self.vtx)
        v_target[:, 0] = v[0]
        v_target[:, 1] = v[1]
        v_target[:, 2] = v[2]

        m = self._rotation_matrix(self.n, n_target)
        gv1 = np.einsum('...ij,...j', m, g)
        ang = self._angle_between_vectors(gv1, v_target, n_target)

        v0 = self.vtx[src]
        v1 = v

        return r, ang, v0, v1

    def generate_wavelet(self, r, ang, sigma=1.0, length=1.0, ori=0, hull=0.1):
        """Complex-valued Gabor filter. The amplitude is set to 1 and the hull
        scales with the wavelength to exhibit the number of side lobes at each
        spatial frequency.

        Parameters
        ----------
        r : np.ndarray, shape=(N,)
            Geodesic distance to origin.
        ang : np.ndarray, shape=(N,)
            Polar angle.
        sigma : float
            Standard deviation of the gaussian hull.
        length : float
            Wavelength of the sinusoidal oscillation.
        ori : float
            Orientation of the sinusoidal oscillation (rad).
        hull : float
            Cutoff value of gaussian hull.

        Returns
        -------
        np.ndarray, shape=(N,)
            Gabor wavelet.

        References
        -------
        .. [1] Tortorici, C. et al. Performing Image-like Convolution on
        Triangular Meshes. Eurogr. Workshop 3D Object Retr. EG 3DOR (2018).

        """

        size = length ** 2 / (2 * sigma ** 2)
        psi = np.exp(-r ** 2 / size) * np.exp(
            1j * 2 * np.pi / length * r * np.cos(ang + ori))

        if hull is not None:
            psi_hull = np.exp(-r ** 2 / size)
            psi[psi_hull < hull] = 0

        y = np.zeros(len(self.vtx), dtype='complex')
        y[self.roi] = psi[self.roi]

        return np.nan_to_num(y)

    def fit(self, arr, file_out=None, **params):
        """Fit filter bank to data array in each point of the ROI. For each
        point, the parameters for the wavelet with the highest response are
        stored and saved as pandas dataframe.

        Parameters
        ----------
        arr : np.ndarray, shape=(N,)
            Data array.
        file_out : str
            File name of saved pandas dataframe.
        params : dict
            Further optional parameters to change default setting.

        Returns
        -------
        dict
            Dictionary collecting the output under the following keys

            * ind (U,) : Vertex index.
            * v0 (U,3) : Vertex.
            * v1 (U,3) : Neighbor Vertex for spanning the coordinate system.
            * y (U,) : Filtered response (complex-valued).
            * lambda (U,) : Best wavelength.
            * ori (U,) : Best orientation.

        """

        # set default arguments if no parameters are given
        params = {**wavelet_params, **params}

        res = Parallel(n_jobs=NUM_CORES)(
            delayed(self._fit)(
                i,
                arr,
                params,
                ) for i in tqdm(range(len(self.roi)))
        )

        res = np.asarray(res)

        data = {
            'ind': res[:, 0],
            'v0': res[:, 1],
            'v1': res[:, 2],
            'y': res[:, 3],
            'lambda': res[:, 4],
            'ori': res[:, 5],
        }

        # create pandas dataframe
        df = pd.DataFrame(data=data)

        if file_out:

            dir_out = os.path.dirname(file_out)
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)

            df.to_parquet(file_out, engine="pyarrow")

        return df

    def _fit(self, i, arr, params):
        """Helper function used by .fit() to run the fitting procedure on
        multiple CPUs in parallel.

        Parameters
        ----------
        i : int
            Array index in ROI.
        arr : np.ndarray, shape=(N,)
            Data array.
        params : dict
            Further optional parameters to change default setting.

        Returns
        -------
        list
            Output for one array index.

        """

        r, phi, v0, v1 = self.get_coordinates(i)
        tmp = 0
        filt = 0
        wave = 0
        ori = 0
        for m, n in list(itertools.product(params['lambda'],
                                           params['ori'])):
            y = self.generate_wavelet(r, phi, params['sigma'], m, n,
                                      params['hull'])
            tmp2 = self.convolution(y, arr)
            tmp2_abs = np.abs(tmp2)
            if tmp2_abs > tmp:
                tmp = tmp2_abs
                filt = tmp2
                wave = m
                ori = n

        return [self.roi[i], v0, v1+v0, filt, wave, ori]

    @staticmethod
    def convolution(arr_kernel, arr):
        """Convolve array with one kernel. The kernel is not shifted along the
        array. Therefore, the output is not the array convolved with the kernel
        array but just the result after applying one particular kernel to the
        array.

        Parameters
        ----------
        arr_kernel : np.ndarray, shape=(N,)
            Kernel array.
        arr : np.ndarray, shape=(N,)
            Data array.

        Returns
        -------
        float
            Filtered data point

        """

        return np.sum(arr_kernel * arr) / len(arr_kernel[arr_kernel != 0])

    def _gradient(self, arr_s):
        """Compute vertex-wise gradients of a scalar field sampled on a
        triangular mesh. The calculation is taken from [1].

        Parameters
        ----------
        arr_s : np.ndarray, shape=(N,)
            Scalar field values per vertex.

        Returns
        -------
        gv : np.ndarray, shape=(N,3)
            Vertex-wise gradient vector.
        gv_magn : np.ndarray, shape=(N,)
            Vertex-wise gradient magnitude.

        References
        -------
        .. [1] Mancinelli, C. et al. Gradient field estimation on triangle
        meshes. Eurographics Proceedings (2018).

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

        return gv, gv_magn

    def _f2v(self, gf, a):
        """Transform face- to vertex-wise expressions.

        Parameters
        ----------
        gf : np.ndarray, shape=(M,3)
            Face-wise gradients.
        a : np.ndarray, shape=(M,)
            Triangle areas.

        Returns
        -------
        gv : np.ndarray, shape=(N,3)
            Vertex-wise gradients.

        """

        vf = self.mesh.vfm
        tmp = a[:, np.newaxis] * gf
        gv = vf.dot(np.nan_to_num(tmp))
        magn = vf.dot(np.nan_to_num(a))

        gv[:, 0] /= magn
        gv[:, 1] /= magn
        gv[:, 2] /= magn

        return gv

    @staticmethod
    def _normalize(arr):
        """Normalize a numpy array of shape=(n,3) along axis=1.

        Parameters
        ----------
        arr : np.ndarray, shape=(N,3)
            Data array

        Returns
        -------
        res : np.ndarray, shape=(N,3)
            Normalized data array.

        """

        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        lens[lens == 0] = np.nan
        res = np.zeros_like(arr)
        res[:, 0] = arr[:, 0] / lens
        res[:, 1] = arr[:, 1] / lens
        res[:, 2] = arr[:, 2] / lens
        res_sum = np.sum(res, axis=1)
        res[~np.isfinite(res_sum), :] = 0

        return res

    @staticmethod
    def _rotation_matrix(f, t):
        """Computes a 3x3 rotation matrix that rotates a unit vector f into
        another unit vector t. The algorithm is taken from [1] and uses the
        implementation found in [2].

        Parameters
        ----------
        f : List[float]
            source unit vector.
        t : List[float]
            target unit vector.

        Returns
        -------
        rot : np.ndarray, shape=(3,3)
            Rotation matrix.

        References
        -------
        .. [1] Moeller, T, et al. Efficiently building a matrix to rotate one
        vector to another, Journal of Graphics Tools 4(4), 1--3 (2018).
        .. [2] https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        """

        if not np.all(np.isclose(np.linalg.norm(f, axis=1), 1)):
            raise ValueError("Source vectors must be a unit vectors!")

        if not np.all(np.isclose(np.linalg.norm(t, axis=1), 1)):
            raise ValueError("Target vectors must be a unit vectors!")

        v = np.cross(f, t)
        c = np.einsum('ij,ij->i', f, t)  # dot product
        c[c == 1] = np.nan  # set to zero since h has a singularity there
        h = (1 - c) / (1 - c ** 2)

        vx = v[:, 0]
        vy = v[:, 1]
        vz = v[:, 2]

        rot = np.array([[c + h * vx ** 2, h * vx * vy - vz, h * vx * vz + vy],
                        [h * vx * vy + vz, c + h * vy ** 2, h * vy * vz - vx],
                        [h * vx * vz - vy, h * vy * vz + vx, c + h * vz ** 2]])

        rot = np.moveaxis(rot, 2, 0)

        return rot

    @staticmethod
    def _angle_between_vectors(v1, v2, n):
        """Compute the angle between two 3D vectors in the range (-pi, +pi].

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
        d = np.einsum('ij,ij->i', v1, v2)  # dot product
        ang = np.arctan2(np.linalg.norm(c, axis=1), d)

        # set sign
        ang_sign = np.einsum('ij,ij->i', n, c)  # dot product
        ang_sign[ang_sign >= 0] = 1
        ang_sign[ang_sign != 1] = -1
        ang *= ang_sign

        return ang

    @property
    def vtx(self):
        return self._vtx

    @vtx.setter
    def vtx(self, v):
        v = np.asarray(v)
        if v.ndim != 2 or np.shape(v)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        self._vtx = v

    @property
    def fac(self):
        return self._fac

    @fac.setter
    def fac(self, f):
        f = np.asarray(f)
        if f.ndim != 2 or np.shape(f)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        if np.max(f) != len(self.vtx) - 1:
            raise ValueError("Faces do not match vertex array!")

        self._fac = f

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, r):
        r = np.asarray(r)
        if r.ndim != 1:
            raise ValueError("ROI array has wrong shape!")

        self._roi = r

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, d):
        d = np.asarray(d)
        if d.ndim != 2 or len(d[0]) != len(d[1]) or len(d[0]) != len(self.roi):
            raise ValueError("Distance matrix has wrong shape!")

        self._dist = d


def dist_matrix(file_out, vtx, fac, roi):
    """Create a memory-mapped file which contains the distance matrix from a
    connected region of interest on a triangle surface mesh. The computation of
    matrix elements takes a while. Therefore, joblib is used to execute the
    computation in parallel.

    Parameters
    ----------
    file_out : str
        File name of output file.
    vtx : np.ndarray, shape=(N,3)
        Vertex coordinates.
    fac : np.ndarray, shape=(M,3)
        Vertex indices of each triangle.
    roi : np.ndarray, shape=(U,)
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
    """Compute nearest geodesic distances from index roi[n] to all other indices
    in roi. Distances are written into row n and column n of the distance
    matrix.

    Parameters
    ----------
    i : int
        Element in roi array.
    d : np.ndarray, shape=(N,N)
        Distance matrix.
    vtx : np.ndarray, shape=(N,3)
        Vertex coordinates.
    fac : np.ndarray, shape=(M,3)
        Vertex indices of each triangle.
    roi : np.ndarray, shape=(U,)
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
