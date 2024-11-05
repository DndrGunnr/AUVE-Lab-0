import numpy as np
import numpy.linalg as LA
from einops import rearrange
from typing import List, Union, Tuple
import numpy as np
from pyquaternion import Quaternion


Vector = Union[List[float], List[int], np.ndarray]


def to_numpy(l: Vector) -> np.ndarray:
    if isinstance(l, list):
        return np.array(l)
    elif isinstance(l, np.ndarray):
        return l
    else:
        raise NotImplementedError(f"{type(l)} is not supported")


def to_quaternion(q: Union[Vector, Quaternion]) -> Quaternion:
    if isinstance(q, list):
        assert len(q) == 4, f"{len(q)} != 4"
        return Quaternion(q)
    elif isinstance(q, np.ndarray):
        if q.shape == (4,):
            return Quaternion(q)
        elif q.shape == (3, 3) or q.shape == (4, 4):
            return Quaternion(matrix=q)
        else:
            raise ValueError(f"{q.shape} is neither a quaternion nor a rotation matrix")
    elif isinstance(q, Quaternion):
        return q
    else:
        raise NotImplementedError(f"{type(q)} is not supported")


def make_tf(translation: Vector, rotation: Union[Vector, Quaternion, np.ndarray]) -> np.ndarray:
    """
    Create a homogeneous transformation matrix
    :param translation: (3) - t_x, t_y, t_z
    :param rotation: either 4 number representing a quaternion, a Quaternion, or a rotation matrix
    :return: (4, 4)
    """
    tf = np.eye(4)
    tf[:3, -1] = to_numpy(translation)
    if isinstance(rotation, np.ndarray):
        if rotation.shape == (3, 3) or rotation.shape == (4, 4):
            tf[:3, :3] = rotation[:3, :3]
        else:
            raise ValueError(f"rotation has an invalid shape {rotation.shape}")
    else:
        tf[:3, :3] = to_quaternion(rotation).rotation_matrix
    return tf


def apply_tf(tf: np.ndarray, points: np.ndarray, in_place=False) -> Union[np.ndarray, None]:
    """
    Apply a homogeneous transformation to a set pof points
    :param tf: (4, 4) - transformation matrix
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param in_place: to overwrite points' coordinate with the output or not.
        If True, this function doesn't return anything. Default: False
    :return:  (N, 3) - transformed coordinate
    """
    assert tf.shape == (4, 4), f"{tf.shape} is not a homogeneous transfomration matrix"
    assert points.shape[1] >= 3, f'expect points has at least 3 coord, get: {points.shape[1]}'
    xyz1 = np.pad(points[:, :3], pad_width=[(0, 0), (0, 1)], constant_values=1)  # (N, 4)
    xyz1 = rearrange(tf @ rearrange(xyz1, 'N C -> C N', C=4), 'C N -> N C')
    if in_place:
        points[:, :3] = xyz1[:, :3]
        return
    else:
        return xyz1[:, :3]


