"""
Quaternion utilities (scalar–first convention: q = [w, x, y, z]).

This module is self-contained so that other scripts can simply
    import quaternion_utils as quat
and reuse the mathematical helpers.
"""
import numpy as np


# ------------------------------------------------------------------ #
# Basic quaternion algebra
# ------------------------------------------------------------------ #
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product  q = q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def rotvec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the 3-vector *v* by quaternion *q*."""
    qv = np.concatenate(([0.0], v))
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


# ------------------------------------------------------------------ #
# Optional helper (not used by the FK demo, but handy elsewhere)
# ------------------------------------------------------------------ #
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3 × 3 rotation matrix into a unit quaternion."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)
