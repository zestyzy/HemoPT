#!/usr/bin/env python3
"""Generate vascular pretraining data for HemoPT.

The generator:
- samples points inside the vessel lumen and on the wall;
- uses VTK vtkImplicitPolyDataDistance for inside/outside and distance queries;
- uses VTK vtkSelectEnclosedPoints for fast batch inside/outside testing;
- adds boundary type features for capped faces when available.

Input: Vascular STL files from Vascular_STL/
Output per mesh:
  x.npy: (36864, 7) = [x, y, z, dist_to_wall, nx, ny, nz]
  supervise_j.npy: (36864, 9) = 3-step displacement to nearest wall
  condition_j.npy: (36864, 4) = [dx, dy, dz, step_length]
"""
import os
import sys
import glob
import argparse
import json
import numpy as np
import trimesh
import vtk
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

try:
    import fcpw
except ImportError:
    fcpw = None

N_SURF = 4096
N_VOL = 32768
N_TOTAL = N_SURF + N_VOL
TARGET_LENGTH = 5.0
N_RANDOM_WALKS = 20
BASE_WALKS = 20
PERTURB_SIGMA = 0.05
WALK_STEPS = 3
MIN_STEP = 0.0
MAX_STEP = 2.0
DEFAULT_QC_MANIFEST = "./HemoData/Vascular_STL_QC/qc_manifest.jsonl"


def dtype_from_name(name):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}")


def expected_output_files(save_dir, n_random_walks=N_RANDOM_WALKS):
    files = [os.path.join(save_dir, "x.npy")]
    for j in range(n_random_walks):
        files.append(os.path.join(save_dir, f"supervise_{j}.npy"))
        files.append(os.path.join(save_dir, f"condition_{j}.npy"))
    return files


def is_complete_output(save_dir, n_random_walks=N_RANDOM_WALKS):
    return all(os.path.exists(path) for path in expected_output_files(save_dir, n_random_walks))


def _float_close(a, b, tol=1e-8):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _meta_matches_expected(meta, expected):
    if not meta or meta.get("success") is not True:
        return False, "missing_or_unsuccessful_meta"

    for key, expected_value in expected.items():
        actual = meta.get(key)
        if isinstance(expected_value, float):
            if not _float_close(actual, expected_value):
                return False, f"{key}: expected {expected_value}, got {actual}"
        elif actual != expected_value:
            return False, f"{key}: expected {expected_value}, got {actual}"
    return True, "match"


def expected_generation_meta(stl_path, dataset_name, n_random_walks, base_walks,
                             perturb_sigma, geometry_backend, collision_backend,
                             save_dtype):
    return {
        "dataset": dataset_name,
        "source_path": os.path.abspath(stl_path),
        "normalized_target_length": TARGET_LENGTH,
        "n_surface_points": N_SURF,
        "n_volume_points": N_VOL,
        "n_random_walks": int(n_random_walks),
        "base_walks": int(min(base_walks, n_random_walks)),
        "perturb_sigma": float(perturb_sigma),
        "walk_steps": WALK_STEPS,
        "geometry_backend_requested": geometry_backend,
        "collision_backend": collision_backend,
        "save_dtype": save_dtype,
    }


def complete_output_matches_meta(save_dir, expected):
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False, "missing_meta_json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as exc:
        return False, f"unreadable_meta_json:{str(exc)[:120]}"
    return _meta_matches_expected(meta, expected)


def remove_expected_output_files(save_dir, n_random_walks=N_RANDOM_WALKS):
    paths = [os.path.join(save_dir, "x.npy"),
             os.path.join(save_dir, "meta.json")]
    paths.extend(glob.glob(os.path.join(save_dir, "condition_*.npy")))
    paths.extend(glob.glob(os.path.join(save_dir, "supervise_*.npy")))
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def write_meta(save_dir, **kwargs):
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(kwargs, f, indent=2)


def update_existing_meta_qc(save_dir, stl_path, dataset_name, qc_row):
    if not qc_row:
        return
    meta_path = os.path.join(save_dir, "meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        meta = {
            "dataset": dataset_name,
            "source_path": os.path.abspath(stl_path),
            "success": True,
            "status": "already exists",
        }
    meta["qc_status"] = qc_row.get("status")
    meta["qc_reasons"] = qc_row.get("reasons")
    meta["qc_relative_path"] = qc_row.get("relative_path")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_reserved_vmr_cases(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("reserved_case_ids", []))


def load_include_list(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    include_paths = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            include_paths.add(os.path.abspath(line))
    return include_paths


def _normalize_qc_key(path):
    return path.replace("\\", "/").lstrip("./")


def _normalize_statuses(statuses):
    result = []
    for item in statuses:
        for status in str(item).split(","):
            status = status.strip().lower()
            if status:
                result.append(status)
    return set(result)


def load_qc_manifest(path):
    """Load STL QC rows keyed by relative path and dataset/filename."""
    if not path or not os.path.exists(path):
        return {}, {}

    qc_by_key = {}
    status_counts = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            status = str(row.get("status", "unknown")).lower()
            row["status"] = status
            status_counts[status] = status_counts.get(status, 0) + 1

            rel_path = row.get("relative_path")
            if rel_path:
                qc_by_key[_normalize_qc_key(rel_path)] = row

            dataset = row.get("dataset")
            filename = row.get("filename")
            if dataset and filename:
                qc_by_key[_normalize_qc_key(os.path.join(dataset, filename))] = row

            abs_path = row.get("path")
            if abs_path:
                qc_by_key[_normalize_qc_key(os.path.abspath(abs_path))] = row

    return qc_by_key, status_counts


def qc_row_for_file(path, base_dir, qc_by_key):
    keys = [
        _normalize_qc_key(os.path.relpath(path, base_dir)),
        _normalize_qc_key(os.path.join(os.path.basename(os.path.dirname(path)),
                                       os.path.basename(path))),
        _normalize_qc_key(os.path.abspath(path)),
    ]
    for key in keys:
        row = qc_by_key.get(key)
        if row is not None:
            return row
    return None


def filter_files_by_qc(files, base_dir, qc_by_key, allowed_statuses, require_match=False):
    kept = []
    skipped = []
    unmatched = 0
    skipped_by_status = {}

    for path in files:
        row = qc_row_for_file(path, base_dir, qc_by_key)
        if row is None:
            unmatched += 1
            if require_match:
                skipped.append((path, "unmatched_qc"))
                skipped_by_status["unmatched_qc"] = skipped_by_status.get("unmatched_qc", 0) + 1
            else:
                kept.append(path)
            continue

        status = row.get("status", "unknown")
        if status in allowed_statuses:
            kept.append(path)
        else:
            reason = row.get("reasons", "")
            skipped.append((path, f"{status}:{reason}"))
            skipped_by_status[status] = skipped_by_status.get(status, 0) + 1

    return kept, skipped, unmatched, skipped_by_status


def vmr_case_id_from_stl(path, reserved_cases=None):
    name = os.path.basename(path)
    if not name.startswith("VMR_"):
        return None
    stem = os.path.splitext(name)[0][4:]
    if reserved_cases:
        for case_id in sorted(reserved_cases, key=len, reverse=True):
            if stem == case_id or stem.startswith(case_id + "_"):
                return case_id
    parts = os.path.splitext(name)[0].split("_")
    if len(parts) < 5:
        return None
    return "_".join(parts[1:5])


def aneumo_case_id_from_stl(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].lower() == "aneumo":
        return parts[1]
    if stem.isdigit():
        return stem
    return None


def trimesh_to_vtk_polydata(mesh):
    """Convert trimesh.Trimesh to vtkPolyData."""
    points = vtk.vtkPoints()
    for v in mesh.vertices:
        points.InsertNextPoint(v[0], v[1], v[2])

    triangles = vtk.vtkCellArray()
    for f in mesh.faces:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, f[0])
        tri.GetPointIds().SetId(1, f[1])
        tri.GetPointIds().SetId(2, f[2])
        triangles.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    return polydata


class VTKDistanceField:
    """VTK-based distance field for inside/outside testing and closest point queries."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.polydata = trimesh_to_vtk_polydata(mesh)

        # Build implicit distance (handles inside/outside for non-watertight meshes)
        self.implicit = vtk.vtkImplicitPolyDataDistance()
        self.implicit.SetInput(self.polydata)

        # Build cell locator for closest point queries
        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(self.polydata)
        self.locator.BuildLocator()

    def evaluate(self, points):
        """Evaluate signed distance and closest point for an array of points.

        Returns:
            signed_dist: (N,) negative inside, positive outside
            closest_points: (N, 3) closest point on surface
            normals: (N, 3) surface normal at closest point
        """
        n = len(points)
        signed_dist = np.zeros(n, dtype=np.float32)
        closest_pts = np.zeros((n, 3), dtype=np.float32)
        normals = np.zeros((n, 3), dtype=np.float32)

        closest_point = [0.0, 0.0, 0.0]
        # Process in batches using vectorized approach
        for i in range(n):
            val = self.implicit.FunctionValue(points[i])
            signed_dist[i] = val

        return signed_dist

    def closest_points(self, query_points):
        """Find closest points on surface.

        Returns:
            distances: (N,) distances to closest point
            closest_points: (N, 3) closest points on surface
            normals: (N, 3) normals at closest points
        """
        n = len(query_points)
        distances = np.zeros(n, dtype=np.float32)
        closest = np.zeros((n, 3), dtype=np.float32)
        normals = np.zeros((n, 3), dtype=np.float32)

        # Precompute face normals
        face_normals = np.zeros((self.polydata.GetNumberOfCells(), 3), dtype=np.float32)
        for ci in range(self.polydata.GetNumberOfCells()):
            cell = self.polydata.GetCell(ci)
            p0 = np.array(cell.GetPoints().GetPoint(0))
            p1 = np.array(cell.GetPoints().GetPoint(1))
            p2 = np.array(cell.GetPoints().GetPoint(2))
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            face_normals[ci] = normal

        for i in range(n):
            cp = [0.0, 0.0, 0.0]
            cellId = vtk.mutable(0)
            subId = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)
            self.locator.FindClosestPoint(query_points[i].tolist(), cp, cellId, subId, dist2)
            d = np.sqrt(dist2.get())
            distances[i] = d
            closest[i] = cp

            ci = cellId.get()
            if 0 <= ci < len(face_normals):
                normals[i] = face_normals[ci]

        return distances, closest, normals

    def contains_batch(self, points):
        """Fast batch inside/outside test using vtkSelectEnclosedPoints.

        Returns:
            inside: (N,) boolean array
        """
        pts_pd = vtk.vtkPolyData()
        pts_vtk = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for p in points:
            pid = pts_vtk.InsertNextPoint(p[0], p[1], p[2])
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
        pts_pd.SetPoints(pts_vtk)
        pts_pd.SetVerts(verts)

        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetInputData(pts_pd)
        selector.SetSurfaceData(self.polydata)
        selector.Update()

        inside = np.array([selector.IsInside(i) for i in range(len(points))], dtype=bool)
        return inside

    def inside_slow(self, points):
        """Slower but more reliable inside test for small batches.

        Returns:
            inside: (N,) boolean array
        """
        signed_dist = np.array([self.implicit.FunctionValue(p) for p in points])
        return signed_dist < 0


class FCPWGeometryQuery:
    """FCPW-accelerated closest-point and ray-intersection queries."""

    def __init__(self, mesh):
        if fcpw is None:
            raise ImportError("fcpw is not installed")

        vertices = np.asfortranarray(np.asarray(mesh.vertices, dtype=np.float32))
        faces = np.asfortranarray(np.asarray(mesh.faces, dtype=np.int32))

        self.scene = fcpw.scene_3D()
        self.scene.set_object_count(1)
        self.scene.set_object_vertices(vertices, 0)
        self.scene.set_object_triangles(faces, 0)
        self.scene.build(fcpw.aggregate_type.bvh_surface_area, False)
        self.backend = "fcpw"

    def closest_points(self, query_points):
        points = np.asfortranarray(np.asarray(query_points, dtype=np.float32))
        radii = np.full((len(points),), np.finfo(np.float32).max, dtype=np.float32)
        interactions = fcpw.interaction_3D_list()
        self.scene.find_closest_points(points, radii, interactions, True)

        closest = np.empty((len(points), 3), dtype=np.float32)
        distances = np.empty((len(points),), dtype=np.float32)
        normals = np.empty((len(points), 3), dtype=np.float32)
        for i in range(len(points)):
            interaction = interactions[i]
            if interaction.primitive_index < 0:
                raise RuntimeError("FCPW closest point query missed the surface")
            closest[i] = interaction.p
            distances[i] = interaction.d
            normals[i] = interaction.n
        return distances, closest, normals

    def first_ray_hits(self, origins, directions, max_distances):
        origins = np.asfortranarray(np.asarray(origins, dtype=np.float32))
        directions = np.asfortranarray(np.asarray(directions, dtype=np.float32))
        bounds = np.asarray(max_distances, dtype=np.float32)
        interactions = fcpw.interaction_3D_list()
        self.scene.intersect(origins, directions, bounds, interactions, False)

        hit = np.zeros((len(origins),), dtype=bool)
        distances = np.full((len(origins),), np.inf, dtype=np.float32)
        points = np.zeros((len(origins), 3), dtype=np.float32)
        for i in range(len(origins)):
            interaction = interactions[i]
            if interaction.primitive_index >= 0 and interaction.d <= bounds[i]:
                hit[i] = True
                distances[i] = interaction.d
                points[i] = interaction.p
        return hit, distances, points


class TrimeshGeometryQuery:
    """Fallback closest-point query using trimesh."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.backend = "trimesh"

    def closest_points(self, query_points):
        from trimesh.proximity import closest_point
        closest, distances, _ = closest_point(self.mesh, query_points)
        return distances.astype(np.float32), closest.astype(np.float32), None


def build_geometry_query(mesh, backend="auto"):
    backend = backend.lower()
    if backend not in {"auto", "fcpw", "trimesh"}:
        raise ValueError(f"Unsupported geometry backend: {backend}")

    if backend in {"auto", "fcpw"}:
        try:
            return FCPWGeometryQuery(mesh)
        except Exception as exc:
            if backend == "fcpw":
                raise
            print(f"[WARN] FCPW unavailable for this mesh, falling back to trimesh: {exc}")

    return TrimeshGeometryQuery(mesh)


def sample_volume_inside_mesh(vtk_dist, mesh, N=N_VOL, max_iter=80):
    """Sample points INSIDE the vessel mesh."""
    bounds = mesh.bounds
    collected = []
    total = 0
    batch_size = 65536

    for it in range(max_iter):
        if total >= N:
            break

        pts = np.random.uniform(bounds[0], bounds[1], (batch_size, 3)).astype(np.float32)

        # Use fast batch test
        if batch_size <= 100000:
            inside = vtk_dist.contains_batch(pts)
        else:
            inside = vtk_dist.inside_slow(pts)

        new_pts = pts[inside]
        if len(new_pts) > 0:
            collected.append(new_pts)
            total += len(new_pts)

    if not collected:
        return None

    result = np.concatenate(collected, axis=0)
    return result[:N] if len(result) >= N else result


def transform_mesh(mesh, target_length=TARGET_LENGTH):
    """Normalize mesh: center + scale to target length."""
    V = mesh.vertices.copy()
    centroid = V.mean(axis=0)
    V -= centroid

    extents = np.ptp(V, axis=0)
    scale = target_length / extents.max()
    V *= scale

    mesh.vertices = V
    return mesh, scale


def get_wall_distance_and_direction(geometry_query, points):
    """Compute distance and direction to nearest wall."""
    distances, closest, _ = geometry_query.closest_points(points)
    diff = closest - points
    norms = distances.reshape(-1, 1) + 1e-8
    directions = diff / norms
    return distances, directions


def vtk_inside_mask(vtk_implicit, points):
    pts_pd = vtk.vtkPolyData()
    pts_vtk = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for p in points:
        pid = pts_vtk.InsertNextPoint(p[0], p[1], p[2])
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
    pts_pd.SetPoints(pts_vtk)
    pts_pd.SetVerts(verts)

    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetInputData(pts_pd)
    selector.SetSurfaceData(vtk_implicit.polydata)
    selector.Update()
    return np.array([selector.IsInside(i) for i in range(len(points))], dtype=bool)


def clamp_with_vtk_binary_search(vtk_implicit, vol_pos, vol_dirs, vol_steps, intended_end):
    inside = vtk_inside_mask(vtk_implicit, intended_end)
    outside_mask = ~inside

    actual_end = intended_end.copy()
    if outside_mask.any():
        lo = np.zeros(outside_mask.sum(), dtype=np.float32)
        hi = vol_steps[outside_mask].copy()
        origins = vol_pos[outside_mask]
        dirs = vol_dirs[outside_mask]

        for _ in range(6):
            mid = (lo + hi) / 2
            probe = origins + dirs * mid[:, None]
            probe_outside = ~vtk_inside_mask(vtk_implicit, probe)
            hi[probe_outside] = mid[probe_outside]
            lo[~probe_outside] = mid[~probe_outside]

        clamped_steps = (lo + hi) / 2 * 0.99
        actual_end[outside_mask] = origins + dirs * clamped_steps[:, None]

    return actual_end


def clamp_with_fcpw_rays(geometry_query, vol_pos, vol_dirs, vol_steps, intended_end):
    if not hasattr(geometry_query, "first_ray_hits"):
        return intended_end

    actual_end = intended_end.copy()
    moving = vol_steps > 1e-8
    if not moving.any():
        return actual_end

    max_dist = vol_steps[moving].astype(np.float32)
    hit, distances, _ = geometry_query.first_ray_hits(
        vol_pos[moving], vol_dirs[moving], max_dist)
    if hit.any():
        moving_indices = np.flatnonzero(moving)
        hit_indices = moving_indices[hit]
        clamped_steps = np.maximum(distances[hit] * 0.99, 0.0)
        actual_end[hit_indices] = vol_pos[hit_indices] + vol_dirs[hit_indices] * clamped_steps[:, None]
    return actual_end


def multi_step_constrained_walk_inside(geometry_query, vtk_implicit, vol_points, surf_points,
                                        steps=WALK_STEPS, min_step=MIN_STEP,
                                        max_step=MAX_STEP,
                                        init_directions=None, init_step_lengths=None,
                                        collision_backend="fcpw_ray"):
    """Random walk inside the vessel with wall collision detection.

    Uses a geometry query backend for supervision and selectable collision handling.
    """

    N = vol_points.shape[0]
    M = surf_points.shape[0]
    all_points = np.vstack([vol_points, surf_points]).astype(np.float32)
    positions = all_points.copy()

    supervise_list = []
    vol_mask = np.ones(N + M, dtype=bool)
    vol_mask[-M:] = False

    if init_directions is not None:
        directions = init_directions.copy()
    else:
        phi = np.random.uniform(0, 2 * np.pi, size=(N + M, 1))
        cos_theta = np.random.uniform(-1, 1, size=(N + M, 1))
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        directions = np.concatenate([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ], axis=1).astype(np.float32)

    if init_step_lengths is not None:
        step_lengths = init_step_lengths.copy()
    else:
        step_lengths = np.random.uniform(min_step, max_step, size=(N + M,)).astype(np.float32)
        step_lengths[-M:] = 0

    for step_idx in range(steps):
        # Supervision: displacement to nearest wall.
        _, closest, _ = geometry_query.closest_points(positions)
        supervise_list.append(positions - closest)

        if step_idx == steps - 1:
            break

        # Move volume points
        vol_pos = positions[vol_mask].copy()
        vol_dirs = directions[vol_mask]
        vol_steps = step_lengths[vol_mask]

        intended_end = vol_pos + vol_dirs * vol_steps[:, None]
        if collision_backend == "vtk":
            actual_end = clamp_with_vtk_binary_search(
                vtk_implicit, vol_pos, vol_dirs, vol_steps, intended_end)
        elif collision_backend == "fcpw_ray":
            actual_end = clamp_with_fcpw_rays(
                geometry_query, vol_pos, vol_dirs, vol_steps, intended_end)
        else:
            raise ValueError(f"Unsupported collision backend: {collision_backend}")

        positions[vol_mask] = actual_end
        positions[-M:] = surf_points

    supervise = np.concatenate(supervise_list, axis=1)
    condition = np.concatenate([directions, step_lengths[:, None]], axis=1)

    return {'supervise': supervise, 'condition': condition,
            'directions': directions, 'step_lengths': step_lengths}


def process_single_mesh(stl_path, save_root, dataset_name,
                        n_random_walks=N_RANDOM_WALKS,
                        base_walks=BASE_WALKS,
                        perturb_sigma=PERTURB_SIGMA,
                        geometry_backend="auto",
                        collision_backend="fcpw_ray",
                        save_dtype="float16",
                        force=False,
                        qc_row=None):
    """Process one STL file into pre-training data."""
    name = os.path.splitext(os.path.basename(stl_path))[0]
    save_dir = os.path.join(save_root, dataset_name, name)
    os.makedirs(save_dir, exist_ok=True)
    qc_status = qc_row.get("status") if qc_row else None
    qc_reasons = qc_row.get("reasons") if qc_row else None
    np_save_dtype = dtype_from_name(save_dtype)
    expected_meta = expected_generation_meta(
        stl_path, dataset_name, n_random_walks, base_walks, perturb_sigma,
        geometry_backend, collision_backend, save_dtype)

    if force:
        remove_expected_output_files(save_dir, n_random_walks)
    elif is_complete_output(save_dir, n_random_walks):
        meta_ok, meta_msg = complete_output_matches_meta(save_dir, expected_meta)
        if meta_ok:
            update_existing_meta_qc(save_dir, stl_path, dataset_name, qc_row)
            return True, "already exists"
        remove_expected_output_files(save_dir, n_random_walks)
        print(f"[REGEN] {dataset_name}/{name}: complete output exists but meta mismatch ({meta_msg})")

    try:
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 10:
            write_meta(save_dir,
                       dataset=dataset_name,
                       source_path=os.path.abspath(stl_path),
                       qc_status=qc_status,
                       qc_reasons=qc_reasons,
                       success=False,
                       status="invalid mesh")
            return False, "invalid mesh"

        original_vertices = int(len(mesh.vertices))
        original_faces = int(len(mesh.faces))
        original_watertight = bool(mesh.is_watertight)

        # Keep largest component (skip split for speed, just remove noise by face count)
        kept_largest_component = False
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                components.sort(key=lambda c: len(c.faces), reverse=True)
                mesh = components[0]
                kept_largest_component = True
        except Exception:
            pass  # Use mesh as-is if split fails

        # Normalize
        mesh, scale = transform_mesh(mesh)

        # Build VTK distance field
        vtk_dist = VTKDistanceField(mesh)
        geometry_query = build_geometry_query(mesh, geometry_backend)
        if collision_backend == "fcpw_ray" and not hasattr(geometry_query, "first_ray_hits"):
            raise RuntimeError("collision_backend=fcpw_ray requires FCPW geometry backend")

        # Sample surface points
        surf_pts, face_idx = mesh.sample(N_SURF, return_index=True)
        surf_normals = mesh.face_normals[face_idx].astype(np.float32)

        # Sample interior points
        vol_pts = sample_volume_inside_mesh(vtk_dist, mesh)
        if vol_pts is None or len(vol_pts) < 1000:
            write_meta(save_dir,
                       dataset=dataset_name,
                       source_path=os.path.abspath(stl_path),
                       qc_status=qc_status,
                       qc_reasons=qc_reasons,
                       original_vertices=original_vertices,
                       original_faces=original_faces,
                       original_watertight=original_watertight,
                       processed_vertices=int(len(mesh.vertices)),
                       processed_faces=int(len(mesh.faces)),
                       processed_watertight=bool(mesh.is_watertight),
                       kept_largest_component=kept_largest_component,
                       success=False,
                       status=f"insufficient interior points ({0 if vol_pts is None else len(vol_pts)})")
            return False, f"insufficient interior points ({0 if vol_pts is None else len(vol_pts)})"

        # Pad if not enough
        padded_volume_points = False
        sampled_volume_points = int(len(vol_pts))
        if len(vol_pts) < N_VOL:
            idx = np.random.choice(len(vol_pts), N_VOL - len(vol_pts))
            vol_pts = np.concatenate([vol_pts, vol_pts[idx]], axis=0)
            padded_volume_points = True
        padded_ratio = max(0, N_VOL - sampled_volume_points) / N_VOL

        vol_pts = vol_pts[:N_VOL]

        # Compute distance and direction to wall for interior points
        vol_dist, vol_dir = get_wall_distance_and_direction(geometry_query, vol_pts)

        # Assemble x.npy: (N_VOL + N_SURF, 7)
        vol_data = np.hstack([vol_pts,
                              vol_dist.reshape(-1, 1),
                              vol_dir]).astype(np_save_dtype)
        surf_data = np.hstack([surf_pts,
                               np.zeros((N_SURF, 1), dtype=np.float32),
                               surf_normals]).astype(np_save_dtype)
        x = np.vstack([vol_data, surf_data])

        np.save(os.path.join(save_dir, "x.npy"), x)

        # Generate random walks
        base_walk_results = []
        n_base_walks = min(base_walks, n_random_walks)
        for w in range(n_base_walks):
            result = multi_step_constrained_walk_inside(
                geometry_query, vtk_dist, vol_pts, surf_pts,
                collision_backend=collision_backend)
            base_walk_results.append(result)

        # Generate one perturbed training view from each base probe. Base
        # walks are seeds only and are never saved as training views.
        for j in range(n_random_walks):
            base_idx = j % n_base_walks
            base_dirs = base_walk_results[base_idx]['directions']
            base_steps = base_walk_results[base_idx]['step_lengths']

            perturbed_dirs = base_dirs + np.random.randn(*base_dirs.shape).astype(np.float32) * perturb_sigma
            norms = np.linalg.norm(perturbed_dirs, axis=1, keepdims=True)
            perturbed_dirs = perturbed_dirs / (norms + 1e-8)

            result = multi_step_constrained_walk_inside(
                geometry_query, vtk_dist, vol_pts, surf_pts,
                init_directions=perturbed_dirs,
                init_step_lengths=base_steps,
                collision_backend=collision_backend)

            np.save(os.path.join(save_dir, f"supervise_{j}.npy"),
                    result['supervise'].astype(np_save_dtype))
            np.save(os.path.join(save_dir, f"condition_{j}.npy"),
                    result['condition'].astype(np_save_dtype))

        write_meta(save_dir,
                   dataset=dataset_name,
                   source_path=os.path.abspath(stl_path),
                   qc_status=qc_status,
                   qc_reasons=qc_reasons,
                   original_vertices=original_vertices,
                   original_faces=original_faces,
                   original_watertight=original_watertight,
                   processed_vertices=int(len(mesh.vertices)),
                   processed_faces=int(len(mesh.faces)),
                   processed_watertight=bool(mesh.is_watertight),
                   kept_largest_component=kept_largest_component,
                   normalized_target_length=TARGET_LENGTH,
                   scale=float(scale),
                   n_surface_points=N_SURF,
                   n_volume_points=N_VOL,
                   sampled_volume_points=sampled_volume_points,
                   padded_volume_points=padded_volume_points,
                   padded_ratio=float(padded_ratio),
                   n_random_walks=n_random_walks,
                   base_walks=n_base_walks,
                   perturb_sigma=perturb_sigma,
                   walk_steps=WALK_STEPS,
                   geometry_backend_requested=geometry_backend,
                   geometry_backend_effective=geometry_query.backend,
                   collision_backend=collision_backend,
                   save_dtype=save_dtype,
                   success=True,
                   status="OK")
        return True, "OK"

    except Exception as e:
        write_meta(save_dir,
                   dataset=dataset_name,
                   source_path=os.path.abspath(stl_path),
                   qc_status=qc_status,
                   qc_reasons=qc_reasons,
                   success=False,
                   status=str(e)[:300])
        return False, str(e)[:100]


def _worker_fn(args_tuple):
    """Unpack tuple for Pool.starmap."""
    return process_single_mesh(*args_tuple)


def _build_work_items(files, save_root, ds, args, qc_by_key, BASE):
    """Build list of argument tuples for worker processes."""
    items = []
    for f in files:
        qc_row = qc_row_for_file(f, BASE, qc_by_key) if qc_by_key else None
        items.append((
            f, save_root, ds,
            args.n_random_walks,
            args.base_walks,
            args.perturb_sigma,
            args.geometry_backend,
            args.collision_backend,
            args.save_dtype,
            args.force,
            qc_row,
        ))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=["4TCTA_AAA", "CMHA", "IntrA", "AneuRisk", "Aneux", "Totalsegmentator", "VMR"],
                        help='Datasets to process. VMR CFD reserved cases are excluded unless --include_reserved_vmr is set.')
    parser.add_argument('--save_root', type=str,
                        default='./HemoData/Vascular_PreTrain')
    parser.add_argument('--max_files', type=int, default=0,
                        help='Max files per dataset (0=all)')
    parser.add_argument('--n_random_walks', type=int, default=N_RANDOM_WALKS,
                        help='Number of lifted dynamics samples per mesh')
    parser.add_argument('--base_walks', type=int, default=BASE_WALKS,
                        help='Number of base probes; each yields one perturbed training view')
    parser.add_argument('--perturb_sigma', type=float, default=PERTURB_SIGMA,
                        help='Gaussian perturbation std for directions after base walks')
    parser.add_argument('--geometry_backend', type=str, default='auto',
                        choices=['auto', 'fcpw', 'trimesh'],
                        help='Closest-point backend. auto uses FCPW when available, otherwise trimesh.')
    parser.add_argument('--collision_backend', type=str, default='fcpw_ray',
                        choices=['vtk', 'fcpw_ray'],
                        help='Random-walk collision backend. fcpw_ray is faster; vtk preserves old behavior.')
    parser.add_argument('--save_dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='dtype for x/condition/supervise npy outputs')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate outputs even when all expected npy files already exist')
    parser.add_argument('--vmr_reserved_manifest', type=str,
                        default='./HemoData/VMR_CFD_Splits/vmr_cfd_split.json',
                        help='Reserved VMR CFD split manifest to exclude from pre-training')
    parser.add_argument('--include_reserved_vmr', action='store_true',
                        help='Allow pre-training generation from VMR CFD reserved cases')
    parser.add_argument('--aneumo_reserved_manifest', type=str,
                        default='./HemoData/Aneumo_CFD_Splits/aneumo_cfd_split.json',
                        help='Reserved aneumo CFD split manifest to exclude from STL pre-training')
    parser.add_argument('--include_reserved_aneumo', action='store_true',
                        help='Allow pre-training generation from aneumo CFD reserved cases')
    parser.add_argument('--qc_manifest', type=str, default=DEFAULT_QC_MANIFEST,
                        help='STL QC jsonl manifest. Default keeps pass/warn and skips fail.')
    parser.add_argument('--qc_statuses', nargs='+', default=['pass', 'warn'],
                        help='QC statuses allowed for pre-training, e.g. pass warn or pass,warn')
    parser.add_argument('--ignore_qc', action='store_true',
                        help='Do not filter STL files with QC manifest')
    parser.add_argument('--require_qc_match', action='store_true',
                        help='Skip STL files that do not appear in the QC manifest')
    parser.add_argument('--allow_missing_qc_manifest', action='store_true',
                        help='Allow generation to continue when --qc_manifest is missing or empty')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers. 1=serial (default).')
    parser.add_argument('--include_list', type=str, default=None,
                        help='Optional newline-delimited absolute STL include-list. '
                             'When set, only these STL files are processed.')
    args = parser.parse_args()
    if args.n_random_walks <= 0:
        raise ValueError("--n_random_walks must be positive")
    if args.base_walks <= 0:
        raise ValueError("--base_walks must be positive")

    base_raw = "./HemoData/Vascular_STL"
    base_capped = "./HemoData/Vascular_STL_capped"
    if os.path.isdir(base_capped):
        BASE = base_capped
    else:
        BASE = base_raw
        print(f"[WARN] Capped STL directory not found, falling back to raw STL: {base_raw}")

    print("=" * 60)
    print("Vascular Pre-Training Data Generation")
    print(f"Output: {args.save_root}")
    print(f"Input: {BASE}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    total_ok = 0
    total_fail = 0
    total_qc_skip = 0
    include_paths = load_include_list(args.include_list)
    if include_paths is not None:
        print(f"Using explicit STL include-list: {args.include_list} ({len(include_paths)} paths)")
    reserved_vmr_cases = load_reserved_vmr_cases(args.vmr_reserved_manifest)
    if reserved_vmr_cases and not args.include_reserved_vmr:
        print(f"Reserved VMR CFD cases excluded from pre-training: {len(reserved_vmr_cases)}")
    reserved_aneumo_cases = load_reserved_vmr_cases(args.aneumo_reserved_manifest)
    if reserved_aneumo_cases and not args.include_reserved_aneumo:
        print(f"Reserved aneumo CFD cases excluded from STL pre-training: {len(reserved_aneumo_cases)}")

    qc_by_key = {}
    allowed_qc_statuses = _normalize_statuses(args.qc_statuses)
    if not args.ignore_qc and not allowed_qc_statuses:
        raise ValueError("--qc_statuses must contain at least one status unless --ignore_qc is set")
    if args.ignore_qc:
        print("STL QC filtering disabled by --ignore_qc")
    else:
        qc_by_key, qc_status_counts = load_qc_manifest(args.qc_manifest)
        if qc_by_key:
            print(f"Loaded STL QC manifest: {args.qc_manifest}")
            print(f"Allowed QC statuses: {sorted(allowed_qc_statuses)}")
            print(f"QC status counts: {qc_status_counts}")
        else:
            message = f"STL QC manifest not found or empty: {args.qc_manifest}"
            if args.allow_missing_qc_manifest:
                print(f"[WARN] {message}")
                print("[WARN] Proceeding without QC filtering. Run data_preprocess/vascular_stl_qc.py first.")
            else:
                raise FileNotFoundError(
                    message + " (use --allow_missing_qc_manifest only for debugging)")

    for ds in args.datasets:
        stl_dir = os.path.join(BASE, ds)
        files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))
        if include_paths is not None:
            before = len(files)
            files = [f for f in files if os.path.abspath(f) in include_paths]
            print(f"[{ds}] Include-list kept {len(files)}/{before} STL files")
        if ds == "VMR" and reserved_vmr_cases and not args.include_reserved_vmr:
            before = len(files)
            files = [f for f in files if vmr_case_id_from_stl(f, reserved_vmr_cases) not in reserved_vmr_cases]
            print(f"[VMR] Excluded {before - len(files)} STL files from reserved CFD cases")
        if ds.lower() == "aneumo" and reserved_aneumo_cases and not args.include_reserved_aneumo:
            before = len(files)
            files = [f for f in files if aneumo_case_id_from_stl(f) not in reserved_aneumo_cases]
            print(f"[{ds}] Excluded {before - len(files)} STL files from reserved CFD cases")
        if qc_by_key:
            before = len(files)
            files, qc_skipped, qc_unmatched, qc_skipped_by_status = filter_files_by_qc(
                files, BASE, qc_by_key, allowed_qc_statuses, args.require_qc_match)
            total_qc_skip += len(qc_skipped)
            if qc_skipped or qc_unmatched:
                print(f"[{ds}] QC kept {len(files)}/{before}; skipped={len(qc_skipped)} "
                      f"unmatched={qc_unmatched} skipped_by_status={qc_skipped_by_status}")
                for path, reason in qc_skipped[:3]:
                    print(f"  QC_SKIP: {os.path.basename(path)}: {reason}")
        elif not args.ignore_qc and args.allow_missing_qc_manifest:
            print(f"[{ds}] QC manifest has no rows for this run; keeping all {len(files)} files")
        if args.max_files > 0:
            files = files[:args.max_files]

        print(f"\n[{ds}] Processing {len(files)} files (workers={args.num_workers})...")
        ok = 0
        fail = 0
        t0 = time.time()

        work_items = _build_work_items(files, args.save_root, ds, args, qc_by_key, BASE)

        if args.num_workers <= 1:
            for item in tqdm(work_items, desc=ds, unit="mesh", dynamic_ncols=True):
                success, msg = _worker_fn(item)
                if success:
                    ok += 1
                else:
                    fail += 1
        else:
            with mp.Pool(processes=args.num_workers) as pool:
                results = pool.imap_unordered(_worker_fn, work_items)
                pbar = tqdm(results, total=len(work_items), desc=ds, unit="mesh",
                            dynamic_ncols=True)
                for success, msg in pbar:
                    if success:
                        ok += 1
                    else:
                        fail += 1

        total_ok += ok
        total_fail += fail
        print(f"[{ds}] Done: {ok} ok, {fail} fail")

    print("\n" + "=" * 60)
    print(f"Total: {total_ok} ok, {total_fail} fail, {total_qc_skip} QC skipped")
    print(f"Output: {args.save_root}")


if __name__ == "__main__":
    main()
