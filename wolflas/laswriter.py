import numpy as np
import os
import laspy

from numpy import ndarray


def write(data: ndarray,
          point_format: int = 3,
          version: str = "1.2",
          filename: str = "default",
          path: str = os.getcwd()):
    
    if version == "1.2":
        print(f"Writing to {path}\{filename}.las")
        # Changing classes over 31 to 0
        data[:, 8][data[:, 8] > 31] = 0

        new_header = laspy.LasHeader(point_format=point_format, version=version)
        new_las = laspy.LasData(new_header)
        new_las.x = data[:, 0]
        new_las.y = data[:, 1]
        new_las.z = data[:, 2]
        new_las.intensity = data[:, 3]
        new_las.return_number = data[:, 4]
        new_las.number_of_returns = data[:, 5]
        new_las.scan_direction_flag = data[:, 6]
        new_las.edge_of_flight_line = data[:, 7]
        new_las.classification = data[:, 8]
        new_las.synthetic = data[:, 9]
        new_las.key_point = data[:, 10]
        new_las.withheld = data[:, 11]
        new_las.user_data = data[:, 12]
        new_las.point_source_id = data[:, 13]
        new_las.gps_time = data[:, 14]

        new_las.write(f"{path}/{filename}.las")
        print(f"{filename}.las created")

    else:
        print(f"Writing to {path}\{filename}.las")
        new_header = laspy.LasHeader(point_format=point_format, version=version)

        new_las = laspy.LasData(new_header)
        new_las.x = data[:, 0]
        new_las.y = data[:, 1]
        new_las.z = data[:, 2]
        new_las.intensity = data[:, 3]
        new_las.return_number = data[:, 4]
        new_las.number_of_returns = data[:, 5]
        new_las.scan_direction_flag = data[:, 6]
        new_las.edge_of_flight_line = data[:, 7]
        new_las.classification = data[:, 8]
        new_las.synthetic = data[:, 9]
        new_las.key_point = data[:, 10]
        new_las.withheld = data[:, 11]
        new_las.user_data = data[:, 12]
        new_las.point_source_id = data[:, 13]
        new_las.gps_time = data[:, 14]

        new_las.write(f"{path}/{filename}.las")
        print(f"{filename}.las created")


if __name__ == "__main__":
    pass
