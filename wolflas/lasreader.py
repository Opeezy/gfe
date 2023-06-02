import numpy as np
import laspy
import logging

from numpy import ndarray

logging.basicConfig(level=logging.INFO)


def read(file: str) -> ndarray:
    # Reading our file
    logging.info(f"Reading file {file}")
    with laspy.open(file) as fh:
        las = fh.read()
    # Converting las to v 1.4
    las = laspy.convert(las, file_version='1.4')

    # GRABBING METADATA FROM LAS FILE
    points = np.vstack((las.x, las.y, las.z)).transpose()
    intensity = np.array(las.intensity)
    return_number = np.array(las.return_number)
    number_of_returns = np.array(las.number_of_returns)
    scan_direction_flag = np.array(las.scan_direction_flag)
    edge_of_flight_line = np.array(las.edge_of_flight_line)
    classification = np.array(las.classification)
    synthetic = np.array(las.synthetic)
    key_point = np.array(las.key_point)
    withheld = np.array(las.withheld)
    user_data = np.array(las.user_data)
    point_source_id = np.array(las.point_source_id)
    gps_time = np.array(las.gps_time)

    # MASTER ARRAY WITH POINTS AND CORRESPONDING METADATA
    las_data = np.empty([len(points), 16], dtype=np.float64)
    las_data[:, 0] = points[:, 0]
    las_data[:, 1] = points[:, 1]
    las_data[:, 2] = points[:, 2]
    las_data[:, 3] = intensity
    las_data[:, 4] = return_number
    las_data[:, 5] = number_of_returns
    las_data[:, 6] = scan_direction_flag
    las_data[:, 7] = edge_of_flight_line
    las_data[:, 8] = classification
    las_data[:, 9] = synthetic
    las_data[:, 10] = key_point
    las_data[:, 11] = withheld
    las_data[:, 12] = user_data
    las_data[:, 13] = point_source_id
    las_data[:, 14] = gps_time

    logging.info(f"{len(las_data)} points read from file")
    return las_data


if __name__ == "__main__":
    pass
