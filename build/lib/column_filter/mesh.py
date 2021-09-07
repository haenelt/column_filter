# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
from scipy.sparse import csr_matrix
from .util import normalize_array

__all__ = ['Mesh']


class Mesh:
    """Implementation of some useful functions to work with a triangle surface
    mesh.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N,3)
        Vertex coordinates.
    fac : np.ndarray, shape=(M,3)
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

        # number of vertices
        nvtx = len(self.vtx)

        # initialise
        row = []
        col = []

        # get rows and columns of edges
        row.extend(list(self.fac[:, 0]))
        col.extend(list(self.fac[:, 1]))

        row.extend(list(self.fac[:, 1]))
        col.extend(list(self.fac[:, 2]))

        row.extend(list(self.fac[:, 2]))
        col.extend(list(self.fac[:, 0]))

        # make sure that all edges are symmetric
        row.extend(list(self.fac[:, 1]))
        col.extend(list(self.fac[:, 0]))

        row.extend(list(self.fac[:, 2]))
        col.extend(list(self.fac[:, 1]))

        row.extend(list(self.fac[:, 0]))
        col.extend(list(self.fac[:, 2]))

        # adjacency entries get value 1
        data = np.ones(len(row), dtype=np.int8)

        return csr_matrix((data, (row, col)), shape=(nvtx, nvtx))

    @property
    @functools.lru_cache
    def vfm(self):
        """Compute a sparse matrix of vertex-face associations. The matrix has
        the size (nvertex, nface). For each vertex index, all associated faces
        are listed.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse adjacency matrix.

        """

        # number of vertices and faces
        nvtx = len(self.vtx)
        nfac = len(self.fac)

        row = np.hstack(self.fac.T)
        col = np.tile(range(nfac), (1, 3)).squeeze()

        # vertex-face associations get value 1
        data = np.ones(len(row), dtype=np.int8)

        return csr_matrix((data, (row, col)), shape=(nvtx, nfac))

    @property
    @functools.lru_cache
    def face_normals(self):
        """Face-wise surfaces normals.

        Returns
        -------
        n : np.ndarray, shape=(M,3)
            Array of face-wise normal vectors.

        """

        # indexed view into the vertex array
        tris = self.vtx[self.fac]

        # calculate the normal for all triangles by taking the cross product of
        # the vectors v1-v0 and v2-v0 in each triangle and normalize
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        n = normalize_array(n)

        return n

    @property
    @functools.lru_cache
    def vertex_normals(self):
        """Vertex-wise surfaces normals. The code is taken from [1]_ and adapted
        to my own purposes.

        Returns
        -------
        norm : np.ndarray, shape=(N,)
            Array of vertex-wise normal vectors.

        References
        -------
        .. [1] https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy

        """

        # face normals
        n = self.face_normals

        # calculate vertex-wise normals from face normals and normalize
        n = self._f2v(n)
        n = normalize_array(n)

        return n

    @property
    @functools.lru_cache
    def face_areas(self):
        """Triangle areas.

        Returns
        -------
        n : np.ndarray, shape=(M,)
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
        np.ndarray, shape=(N,)
            Array of neighborhood indices.

        """

        return self.adjm[ind, :].indices

    def _f2v(self, nf_arr):
        """Transform face- to vertex-wise expression.

        Parameters
        ----------
        nf_arr : np.ndarray, shape=(M,3)
            Face-wise array.

        Returns
        -------
        np.ndarray, shape=(N,3)
            Vertex-wise array.

        """

        return self.vfm.dot(nf_arr)

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
