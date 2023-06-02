import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import logging
import time

from typing import List
from sklearn.cluster import DBSCAN
from shapely import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from numpy import ndarray
from scipy.spatial.distance import cdist
from wolflas.exceptions import ScanError



'''Takes an ndarray of las data and attempts to separate points into clusters
    using o3d's dbscan.'''


logging.basicConfig(level=logging.INFO)


def dbscan(points: ndarray,
           eps: float = 15,
           min_count: int = 15,
           print_progress: bool = False,
           plot: bool = False) -> List:
    """The difference between 'log' and 'print_progress'
     is that 'log' is for the WolfLas print statements while
      'print_progress' is for the built-in log feature of o3d's
      dbscan method"""

    logging.info("Performing dbscan")
    # Our list for the clusters
    clusters = []

    pcd_points = points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    logging.info("Extracting labels")
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_count, print_progress=print_progress))
    except Exception:
        raise ScanError("Dbscan failed. Try using different eps or min_count params")

    num_of_clusters = int(labels.max())
    labeled_points = np.empty((len(labels), 4), dtype=np.float64)
    labeled_points[:, 0] = labels
    labeled_points[:, 1] = pcd_points[:, 0]
    labeled_points[:, 2] = pcd_points[:, 1]
    labeled_points[:, 3] = pcd_points[:, 2]

    logging.info("Clustering points")

    for i in range(num_of_clusters):
        print(f"({i}/{num_of_clusters})", end="\r")
        cluster = labeled_points[labeled_points[:, 0] == i]
        # Discarding the label
        cluster = cluster[:, [1, 2, 3]]
        clusters.append(cluster)


    if plot:
        for cluster in clusters:
            plt.plot(cluster[:, 0], cluster[:, 1], ".")
        plt.show()

    return clusters


def sk_dbscan(points: ndarray,
              eps: float = 15,
              min_count: int = 15,
              plot: bool = False):
    flattened_points = points[:, [0, 1]]
    clustering = DBSCAN(eps=eps, min_samples=min_count).fit(flattened_points)
    labels = clustering.labels_
    print(labels)
    labels_max = max(labels)
    labeled_points = np.empty((len(labels), 4), dtype=np.float64)
    labeled_points[:, 0] = labels
    labeled_points[:, 1] = points[:, 0]
    labeled_points[:, 2] = points[:, 1]
    labeled_points[:, 3] = points[:, 2]
    clusters = []

    for i in range(labels_max):
        cluster = labeled_points[labeled_points[:, 0] == i]
        cluster = cluster[:, [1, 2, 3]]
        clusters.append(cluster)

    if plot:
        for cluster in clusters:
            plt.plot(cluster[:, 0], cluster[:, 1], ".")
        plt.show()

    return clusters







def cubic_clustering(points: ndarray,
                     length: float = 1.00) -> List:
    """Rather slow but accurate way of clustering points into groups.
       Works well on smaller data sets. (Ex. finding individual bases
       of a tower).
    """
    squares = set()

    def find_square(midpoint: ndarray) -> Polygon:
        """Finding square vertexes around given midpoint"""
        x = midpoint[0]
        y = midpoint[1]

        v1 = [x+length, y]
        v2 = [x, y+length]
        v3 = [x-length, y]
        v4 = [x, y-length]

        return Polygon((v1, v2, v3, v4))

    logging.info("Finding squares")
    for point in points:
        squares.add(find_square(point))

    logging.info("Merging squares")
    merged_squares = unary_union(list(squares))

    clusters = []

    if type(merged_squares) is MultiPolygon:
        polys = set(merged_squares.geoms)
        for p in polys:
            clusters.append([c for c in points if p.contains(Point(c))])
    else:
        clusters.append([c for c in points if merged_squares.contains(Point(c))])

    return clusters


if __name__ == "__main__":
    import os
    from cloud import Cloud

    file = f"{os.path.dirname(os.getcwd())}\\__data__\\buildings.las"
    cld = Cloud(file)
    p = cld.points
    p = p[::20]
    dbscan(p)

