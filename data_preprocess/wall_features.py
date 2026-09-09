"""Wall-aware feature utilities for vascular CFD preprocessing."""

import os

import numpy as np
import vtk
from vtk.util import numpy_support as nps

try:
    import fcpw
except Exception:
    fcpw = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError(f"Boolean value expected, got {value!r}")


def geometry_normalization_stats(points, target_length):
    points = np.asarray(points, dtype=np.float64)
    bound_max = np.max(points, axis=0)
    bound_min = np.min(points, axis=0)
    length = float(np.max(bound_max - bound_min))
    if length < 1e-12:
        raise RuntimeError("Degenerate geometry extent.")
    scale = float(target_length) / length
    center = (points * scale).mean(axis=0)
    return scale, center


def apply_geometry_normalization(points, scale, center):
    return np.asarray(points, dtype=np.float64) * scale - center


def _read_polydata(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".stl":
        reader = vtk.vtkSTLReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == ".vtk":
        reader = vtk.vtkGenericDataObjectReader()
    else:
        raise ValueError(f"Unsupported wall surface extension: {path}")

    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()

    if isinstance(output, vtk.vtkPolyData):
        poly = output
    else:
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(output)
        geometry.Update()
        poly = geometry.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        raise RuntimeError(f"Empty wall surface: {path}")

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(tri.GetOutput())
    clean.Update()
    poly = clean.GetOutput()

    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        raise RuntimeError(f"Wall surface became empty after cleaning: {path}")
    return poly


def _polydata_to_arrays(polydata):
    points = nps.vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            continue
        faces.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])
    faces = np.asarray(faces, dtype=np.int32)
    if len(points) == 0 or len(faces) == 0:
        raise RuntimeError("Wall surface has no triangular faces.")
    return points, faces


class WallFeatureLocator:
    """Closest-point queries against a wall surface."""

    def __init__(self, surface_path, backend="auto"):
        self.surface_path = str(surface_path)
        self.polydata = _read_polydata(surface_path)
        self.backend = backend
        self._fcpw_scene = None
        self.locator = None
        if backend not in {"auto", "fcpw", "vtk"}:
            raise ValueError(f"Unsupported wall feature backend: {backend}")
        if backend in {"auto", "fcpw"} and fcpw is not None:
            points, faces = _polydata_to_arrays(self.polydata)
            self._fcpw_scene = fcpw.scene_3D()
            self._fcpw_scene.set_object_count(1)
            self._fcpw_scene.set_object_vertices(np.asfortranarray(points), 0)
            self._fcpw_scene.set_object_triangles(np.asfortranarray(faces), 0)
            self._fcpw_scene.build(fcpw.aggregate_type.bvh_surface_area, False)
            self.backend = "fcpw"
        else:
            if backend == "fcpw":
                raise ImportError("fcpw is not installed")
            self.locator = vtk.vtkStaticCellLocator()
            self.locator.SetDataSet(self.polydata)
            self.locator.BuildLocator()
            self.backend = "vtk"

    @property
    def n_points(self):
        return int(self.polydata.GetNumberOfPoints())

    @property
    def n_cells(self):
        return int(self.polydata.GetNumberOfCells())

    def features(self, points, scale=1.0, eps=1e-8):
        points = np.asarray(points, dtype=np.float64)
        if self._fcpw_scene is not None:
            query = np.asfortranarray(points.astype(np.float32))
            radii = np.full((len(query),), np.finfo(np.float32).max, dtype=np.float32)
            interactions = fcpw.interaction_3D_list()
            self._fcpw_scene.find_closest_points(query, radii, interactions, True)
            closest = np.empty_like(points, dtype=np.float64)
            raw_dist = np.empty(points.shape[0], dtype=np.float64)
            for i in range(len(points)):
                interaction = interactions[i]
                if interaction.primitive_index < 0:
                    raise RuntimeError("FCPW closest point query missed the wall surface")
                closest[i] = interaction.p
                raw_dist[i] = interaction.d
        else:
            closest = np.empty_like(points, dtype=np.float64)
            raw_dist = np.empty(points.shape[0], dtype=np.float64)

            c = [0.0, 0.0, 0.0]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)
            for i, point in enumerate(points):
                self.locator.FindClosestPoint(point, c, cell_id, sub_id, dist2)
                closest[i] = c
                raw_dist[i] = float(dist2) ** 0.5

        diff = closest - points
        denom = np.maximum(raw_dist[:, None], eps)
        direction = diff / denom
        near_zero = raw_dist < eps
        if np.any(near_zero):
            direction[near_zero] = 0.0

        dist = raw_dist * float(scale)
        features = np.concatenate([dist[:, None], direction], axis=1)
        stats = {
            "wall_dist_min": float(dist.min()) if len(dist) else 0.0,
            "wall_dist_mean": float(dist.mean()) if len(dist) else 0.0,
            "wall_dist_max": float(dist.max()) if len(dist) else 0.0,
            "wall_near_zero_count": int(near_zero.sum()),
            "wall_surface_points": self.n_points,
            "wall_surface_cells": self.n_cells,
            "wall_feature_backend": self.backend,
        }
        return features, stats
