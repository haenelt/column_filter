# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
from scipy.sparse import csr_matrix

__all__ = ['Mesh']


class Mesh:
    """Implementation of some useful functions to work with a triangle surface
    mesh.

    Parameters
    ----------
    vtx : (N,3) np.ndarray
        Vertex coordinates.
    fac : (M,3) np.ndarray
        Vertex indices of each triangle.

    Raises
    ------
    ValueError :
        If `vtx` has a wrong shape or `fac` does not match the vertex array.

    """

    def __init__(self, vtx, fac):
        self.vtx = vtx
        self.fac = fac

    @property
    @functools.lru_cache
    def adjm(self):
        """Compute a sparse adjacency matrix. The matrix has the size
        (nvertex, nvertex). Each matrix entry with value 1 stands for an edge of
        the surface mesh.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse adjacency matrix.

        """

        # get number of vertices and faces
        nvtx = len(self.vtx)
        nfac = len(self.fac)

        # initialise
        row = []
        col = []

        # get rows and columns of edges
        row.extend([self.fac[i, 0] for i in range(nfac)])
        col.extend([self.fac[i, 1] for i in range(nfac)])

        row.extend([self.fac[i, 1] for i in range(nfac)])
        col.extend([self.fac[i, 2] for i in range(nfac)])

        row.extend([self.fac[i, 2] for i in range(nfac)])
        col.extend([self.fac[i, 0] for i in range(nfac)])

        # make sure that all edges are symmetric
        row.extend([self.fac[i, 1] for i in range(nfac)])
        col.extend([self.fac[i, 0] for i in range(nfac)])

        row.extend([self.fac[i, 2] for i in range(nfac)])
        col.extend([self.fac[i, 1] for i in range(nfac)])

        row.extend([self.fac[i, 0] for i in range(nfac)])
        col.extend([self.fac[i, 2] for i in range(nfac)])

        # adjacency entries get value 1
        data = np.ones(len(row), dtype=np.int8)

        return csr_matrix((data, (row, col)), shape=(nvtx, nvtx))

    @property
    @functools.lru_cache
    def face_normals(self):
        """Face-wise surfaces normals.

        Returns
        -------
        n : (M,) np.ndarray
            Array of face-wise normal vectors.

        """

        # indexed view into the vertex array
        tris = self.vtx[self.fac]

        # calculate the normal for all triangles by taking the cross product of
        # the vectors v1-v0 and v2-v0 in each triangle and normalize
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        n = self._normalize(n)

        return n

    @property
    @functools.lru_cache
    def vertex_normals(self):
        """Vertex-wise surfaces normals. The code is taken from [1]_ and adapted
        to my own purposes.

        Returns
        -------
        norm : (N,) np.ndarray
            Array of vertex-wise normal vectors.

        References
        -------
        .. [1] https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy

        """

        # face normals
        n = self.face_normals

        # calculate vertex-wise normals from face normals and normalize
        n = self._f2v(n)
        n = self._normalize(n)

        return n

    @property
    @functools.lru_cache
    def face_areas(self):
        """Triangle areas.

        Returns
        -------
        n : (M,) np.ndarray
            Array of face areas.

        """

        # indexed view into the vertex array
        tris = self.vtx[self.fac]

        # calculate the normal for all triangles by taking the cross product of
        # the vectors v1-v0 and v2-v0 in each triangle and get face area from
        # length
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        n = np.sqrt((n ** 2).sum(-1)) / 2

        return n

    def neighborhood(self, ind):
        """Compute 1-ring neighborhood for one vertex.

        Parameters
        ----------
        ind : int
            Vertex index.

        Returns
        -------
        (N,) np.ndarray
            Array of neighborhood indices.

        """

        return self.adjm[ind, :].indices

    def _f2v(self, nf_arr):
        """Get average vertex-wise normal by adding up all face-wise normals
        around vertex."""
        nv_arr = np.zeros_like(self.vtx)
        for i in range(len(self.fac)):
            nv_arr[self.fac[i, 0], :] += nf_arr[i, :]
            nv_arr[self.fac[i, 1], :] += nf_arr[i, :]
            nv_arr[self.fac[i, 2], :] += nf_arr[i, :]

        return nv_arr

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
