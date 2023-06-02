import numpy as np

from numpy import ndarray
from typing import Union


def compute_normals(data: ndarray,
                    normal_value: int = 1) -> tuple:
    min_val = min(data)
    max_val = max(data)
    normalized = np.empty(len(data))

    for key, value in enumerate(data):
        normalized[key] = (value - min_val) / (max_val - min_val) * normal_value

    return normalized, min_val, max_val


def revert_normals(data: ndarray,
                   normal_value: int,
                   min_value: Union[int, float],
                   max_value: Union[int, float]) -> ndarray:
    reverted = np.empty(len(data))

    for key, value in enumerate(data):
        reverted[key] = value * (max_value - min_value) + min_value

    return reverted


def normalize_pointset(data: ndarray,
                       normal_value: int = 1) -> tuple:
    xset = data[:, 0]
    yset = data[:, 1]
    zset = data[:, 2]

    normalized_set = np.empty((len(data), 3))
    normalized_set[:, 0], x_min, x_max = compute_normals(xset, normal_value=normal_value)
    normalized_set[:, 1], y_min, y_max = compute_normals(yset, normal_value=normal_value)
    normalized_set[:, 2], z_min, z_max = compute_normals(zset, normal_value=normal_value)

    return normalized_set, [x_min, y_min, z_min], [x_max, y_max, z_max]


def revert_normalization(data: ndarray,
                         min_values: list,
                         max_values: list,
                         normal_value: int) -> ndarray:
    xset = data[:, 0]
    yset = data[:, 1]
    zset = data[:, 2]

    x_min = min_values[0]
    y_min = min_values[1]
    z_min = min_values[2]

    x_max = max_values[0]
    y_max = max_values[1]
    z_max = max_values[2]

    reverted_set = np.empty((len(data), 3))
    reverted_set[:, 0] = revert_normals(xset, normal_value=normal_value, min_value=x_min, max_value=x_max)
    reverted_set[:, 1] = revert_normals(yset, normal_value=normal_value, min_value=y_min, max_value=y_max)
    reverted_set[:, 2] = revert_normals(zset, normal_value=normal_value, min_value=z_min, max_value=z_max)
    return reverted_set

if __name__ == "__main__":
    pass