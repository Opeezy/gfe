import sys

import numpy as np
import pptk
import logging

from wolflas.alphashape import alpha_shape
from wolflas.lasreader import read
from wolflas.laswriter import write
from typing import Union, List
from wolflas.normalization import normalize_pointset, revert_normalization
from wolflas.clustering import dbscan, cubic_clustering, sk_dbscan
from numpy import ndarray
from pyautocad import Autocad, APoint
from shapely import MultiPolygon, Polygon, delaunay_triangles
from scipy.spatial.distance import cdist
from wolflas.exceptions import InvalidClassError, VersionError


logging.basicConfig(level=logging.INFO)


class Cloud:
    def __init__(self,
                 file: Union[str, None]) -> None:

        if file is not None:
            logging.info("Reading points")

            self.data = read(file)
            self.points = self.data[:, [0, 1, 2]]
            self.intensity = self.data[:, 3]
            self.return_number = self.data[:, 4]
            self.number_of_returns = self.data[:, 5]
            self.scan_direction_flag = self.data[:, 6]
            self.edge_of_flight_line = self.data[:, 7]
            self.classification = self.data[:, 8]
            self.synthetic = self.data[:, 9]
            self.key_point = self.data[:, 10]
            self.withheld = self.data[:, 11]
            self.user_data = self.data[:, 12]
            self.point_source_id = self.data[:, 13]
            self.gps_time = self.data[:, 14]
            self.unique_classes = np.unique(self.classification)
            self.version = "1.4"
            self.point_count = len(self.data)

    def load_points(self,
                    data: ndarray,
                    file: str) -> None:
        logging.info("Reading points")

        self.data = read(las_file)
        self.points = self.data[:, [0, 1, 2]]
        self.intensity = self.data[:, 3]
        self.return_number = self.data[:, 4]
        self.number_of_returns = self.data[:, 5]
        self.scan_direction_flag = self.data[:, 6]
        self.edge_of_flight_line = self.data[:, 7]
        self.classification = self.data[:, 8]
        self.synthetic = self.data[:, 9]
        self.key_point = self.data[:, 10]
        self.withheld = self.data[:, 11]
        self.user_data = self.data[:, 12]
        self.point_source_id = self.data[:, 13]
        self.gps_time = self.data[:, 14]
        self.unique_classes = np.unique(self.classification)
        self.version = "1.4"
        self.point_count = len(self.data)

    def view(self,
             points: ndarray = None) -> None:
        if points is None:
            _points = self.points
            v = pptk.viewer(_points)
        else:
            v = pptk.viewer(points)

    def points_in_class(self,
                        classification: int) -> ndarray:
        """Returns all points of a corresponding class"""

        _points_in_class = self.data[self.classification == classification]
        return _points_in_class

    def _draw_shapely_polygons(self,
                               poly: Polygon,
                               acad: Autocad):
        print(poly, end="\r")
        _x, _y = poly.exterior.xy
        _xy = np.empty((len(_x), 2))
        _xy[:, 0] = _x
        _xy[:, 1] = _y
        for key, coord in enumerate(_xy):
            if key == len(_xy) - 1:
                _draw_to = _xy[0]
                p1 = APoint(coord[0], coord[1])
                p2 = APoint(_draw_to[0], _draw_to[1])
            else:
                _draw_to = _xy[key + 1]
                p1 = APoint(coord[0], coord[1])
                p2 = APoint(_draw_to[0], _draw_to[1])
            acad.model.AddLine(p1, p2)
    def draw_polygons(self,
                      data: ndarray,
                      alpha: float = 0.3,
                      tolerance: float = 0.5) -> None:
        logging.info("Clustering points")
        _points: ndarray = data[::5]
        _clusters: List = dbscan(points=_points, eps=10)
        _hulls: List = []
        logging.info("Finding polygons")

        for key, cluster in enumerate(_clusters):
            print(f"({key}/{len(_clusters)})", end="\r")
            _concave_hull, _ = alpha_shape(points=cluster, alpha=alpha)
            _concave_hull = _concave_hull.simplify(tolerance=tolerance)
            _hulls.append(_concave_hull)


        acad = Autocad()
        acad.prompt("Hello, Autocad from Python\n")

        logging.info("Drawing polys to CAD")
        for key, hull in enumerate(_hulls):
            print(f"({key}/{len(_hulls)})", end="\r")
            if type(hull) is MultiPolygon:
                pass
            else:
                self._draw_shapely_polygons(hull, acad)

        sys.exit()


        plt.plot(_points[:, 0], _points[:, 1], 'b.')

        for hull in _hulls:
            if type(hull) is MultiPolygon:
                for geom in hull.geoms:
                    _x, _y = geom.exterior.xy
                    plt.plot(_x, _y, 'r-')
            else:
                _x, _y = hull.exterior.xy
                print(type(_x[1]))
                plt.plot(_x, _y, 'r-')

        plt.show()
        sys.exit()

    def find_bottoms(self,
                     classification: int,
                     write_to_file: bool = False) -> ndarray:
        def points_in_window(tolerance: int,
                             points: ndarray) -> ndarray:
            _lowest_height = min(points[:, 2])
            _points_in_window = points[points[:, 2] < _lowest_height + tolerance]
            return _points_in_window

        logging.info("Finding bottoms")

        if classification in self.unique_classes:
            _points = self.points[self.classification == classification]
            _clusters = dbscan(_points)

            if len(_clusters) == 0:
                _clusters = [_points]

            _bottoms = []
            _bar = _clusters
            for cluster in _clusters:
                """Finding our lowest point and then finding all points within a tolerance"""
                _points_in_window = points_in_window(points=cluster, tolerance=5)

                _bases = cubic_clustering(_points_in_window, 2)
                for base in _bases:
                    base = np.vstack(base)
                    _lowest_point = min(base[:, 2])

                    if len(base) < 4:
                        _bottoms.append(base[base[:, 2] ==_lowest_point])
                    else:
                        _poly = Polygon(base)
                        _centroid = _poly.centroid
                        _bottom = np.array([_centroid.x, _centroid.y, _lowest_point])
                        _bottoms.append(_bottom)

            _bottoms = np.vstack(_bottoms)

            if write_to_file:
                with open("bottoms.txt", "w") as f:
                    for b in _bottoms:
                        f.write(f"{b[0]},{b[1]},{b[2]}\n")
                return _bottoms
            else:
                return _bottoms
        else:
            raise InvalidClassError("Class not found in data")

    def find_single_tops(self,
                         classification: int,
                         write_to_file: bool = False) -> ndarray:
        def points_in_window(tolerance: int,
                             points: ndarray) -> ndarray:
            _highest_height = max(points[:, 2])
            _points_in_window = points[points[:, 2] > _highest_height - tolerance]
            return _points_in_window

        logging.info("Finding tops")

        if classification in self.unique_classes:
            _points = self.points[self.classification == classification]
            _clusters = sk_dbscan(_points, plot=True)

            if len(_clusters) == 0:
                _clusters = [_points]

            _tops = []
            _bar = _clusters
            for cluster in _clusters:
                """Finding our lowest point and then finding all points within a tolerance"""
                _points_in_window = points_in_window(points=cluster, tolerance=2)
                if len(_points_in_window) > 0:
                    _highest_z = max(_points_in_window[:, 2])
                    _top = _points_in_window[_points_in_window[:, 2] == _highest_z]
                    _tops.append(_top)

            _tops = np.vstack(_tops)

            if write_to_file:
                with open("tops.txt", "w") as f:
                    for b in _tops:
                        f.write(f"{b[0]},{b[1]},{b[2]}\n")
                return _tops
            else:
                return _tops
        else:
            raise InvalidClassError("Class not found in data")

    def convert_class(self,
                      classification: int,
                      new_class: int) -> None:
        if classification in self.unique_classes:
            self.data[:, 8][self.data[:, 8] == classification] = new_class
            self.unique_classes = np.unique(self.data[:, 8])
        else:
            raise InvalidClassError("Class not found in data")

    def update_version(self,
                       version: str) -> None:
        _version_list = ["1.2", "1.4", "1.6"]
        if version in _version_list:
            self.version = version
        else:
            raise VersionError("Not a valid version")


if __name__ == "__main__":
    import os

    file = f"D:\\Projects\\Dominion TL 539\\review\\dist_poles.las"
    cld = Cloud(file)
    print(cld.unique_classes)
    cld.find_single_tops(write_to_file=True, classification=46)





