# -*- coding: utf-8 -*-

# external inputs
import numpy as np
from nibabel.freesurfer.io import read_geometry
from nibabel.freesurfer.io import read_label
from surfdist.analysis import dist_calc
from gbb.neighbor import nn_2d
from gbb.normal import get_normal
from gbb.utils import get_adjm
from fmri_tools.surface import gradient
from fmri_tools.io import write_mgh


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


# def wavelet(vtx, vtx_smooth, fac, i, label)


# calculate distance
surf_in = "/data/pt_01880/Experiment1_ODC/p4/anatomy/layer/lh.layer_5"
label_in = "/data/pt_01880/Experiment1_ODC/p4/anatomy/label/lh.v1.label"
label = read_label(label_in)
# label=None

vtx, fac = read_geometry(surf_in)
src = 99742

if label is None:
    label = np.arange(len(vtx))

dist = dist_calc((vtx, fac), label, src)
n = get_normal(vtx, fac)
g, _ = gradient(vtx, fac, dist, normalize=True)
adjm = get_adjm(vtx, fac)

v = vtx[nn_2d(src, adjm, 0)[0], :] - vtx[src, :]
v /= np.linalg.norm(v)

# %%

r = np.zeros(len(vtx))
for i in label:
    M = _rotation_matrix(n[i], n[src])
    gv1 = np.matmul(M, g[i])
    ang = _angle_between_vectors(gv1, v, n[src])
    r[i] = ang

write_mgh("/data/pt_01880/d.mgh", dist)
write_mgh("/data/pt_01880/ang.mgh", r)

# %%

# wavelength
l = 1.0  # warum ist das doppelt so lang?
k = 2 * np.pi / l

dist[np.isinf(dist)] = 0
psi = np.exp(-dist ** 2 / 2) * np.exp(1j * k * dist * np.cos(r))
psi_real = np.zeros(len(vtx))
psi_real[label] = np.real(psi[label])
# psi_real[np.abs(psi_real) < 0.01] = 0
psi_real[np.isnan(psi_real)] = 0

write_mgh("/data/pt_01880/bla.mgh", psi_real)
