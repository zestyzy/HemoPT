#!/usr/bin/env python3
"""Cap open boundaries of vascular STL meshes to make them watertight.

For each non-watertight mesh:
1. Detect boundary edges (edges belonging to only one face)
2. Find connected boundary loops
3. Fill each loop with a planar fan triangulation
4. Save capped mesh + boundary type labels
"""
import os
import sys
import json
import glob
import traceback
import numpy as np
import trimesh

BASE = "./HemoData"
SRC = os.path.join(BASE, "Vascular_STL")
DST = os.path.join(BASE, "Vascular_STL_capped")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cap_mesh(mesh):
    """Cap all open boundaries of a mesh, return capped mesh and cap face indices.

    Returns:
        capped_mesh: trimesh.Trimesh (watertight if capping succeeds)
        cap_face_mask: (F,) boolean array, True for cap faces
        success: bool
    """
    if mesh.is_watertight:
        return mesh, np.zeros(len(mesh.faces), dtype=bool), True

    # Get boundary edges (edges with only one adjacent face)
    edges = mesh.edges_unique
    edge_faces = mesh.faces_unique_edges

    # Count how many faces reference each edge
    edge_count = np.zeros(len(edges), dtype=int)
    for face_edges in edge_faces:
        for ei in face_edges:
            edge_count[ei] += 1

    boundary_edge_indices = np.where(edge_count == 1)[0]
    boundary_edges = edges[boundary_edge_indices]

    if len(boundary_edges) == 0:
        return mesh, np.zeros(len(mesh.faces), dtype=bool), False

    # Build adjacency to find boundary loops
    # Each boundary vertex -> its neighbors via boundary edges
    n_verts = len(mesh.vertices)
    adj = {}
    for e in boundary_edges:
        v0, v1 = e
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    # Find connected loops via DFS
    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = []
        current = start
        prev = -1
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbors = adj.get(current, [])
            next_node = None
            for n in neighbors:
                if n != prev and n not in visited:
                    next_node = n
                    break
            if next_node is None:
                break
            prev = current
            current = next_node

        # Check if loop is closed
        if len(loop) >= 3:
            last = loop[-1]
            if start in adj.get(last, []):
                loops.append(loop)

    if not loops:
        return mesh, np.zeros(len(mesh.faces), dtype=bool), False

    # Create cap faces for each loop using fan triangulation
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    cap_faces_list = []

    for loop in loops:
        if len(loop) < 3:
            continue

        loop_verts = vertices[loop]
        centroid = loop_verts.mean(axis=0)

        # Add centroid as new vertex
        centroid_idx = len(vertices)
        vertices = np.vstack([vertices, centroid.reshape(1, 3)])

        # Fan triangulation from centroid
        n_loop = len(loop)
        for i in range(n_loop):
            v0 = loop[i]
            v1 = loop[(i + 1) % n_loop]
            # Orient face to point outward (away from mesh interior)
            # Use consistent winding: centroid -> v0 -> v1
            face = np.array([centroid_idx, v0, v1], dtype=np.int64)
            cap_faces_list.append(face)

    if not cap_faces_list:
        return mesh, np.zeros(len(mesh.faces), dtype=bool), False

    cap_faces = np.array(cap_faces_list, dtype=np.int64)
    n_original_faces = len(faces)
    all_faces = np.vstack([faces, cap_faces])

    capped_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)

    # Build cap face mask
    cap_face_mask = np.zeros(len(capped_mesh.faces), dtype=bool)
    # After processing, face indices may shift, so we detect cap faces by checking
    # if they reference the centroid vertex
    for i, face in enumerate(capped_mesh.faces):
        if centroid_idx in face:
            cap_face_mask[i] = True

    return capped_mesh, cap_face_mask, capped_mesh.is_watertight


def repair_mesh(mesh):
    """Basic mesh repair for segmentation-derived meshes."""
    # Remove non-finite values
    mesh.remove_infinite_values()
    # Merge close vertices
    mesh.merge_vertices()
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    # Fix normals
    mesh.fix_normals()
    return mesh


def process_dataset(dataset_name):
    """Process all STL files in a dataset directory."""
    src_dir = os.path.join(SRC, dataset_name)
    dst_dir = os.path.join(DST, dataset_name)

    if not os.path.isdir(src_dir):
        print(f"[{dataset_name}] Directory not found, skipping")
        return 0, 0, 0

    ensure_dir(dst_dir)

    stl_files = sorted(glob.glob(os.path.join(src_dir, "*.stl")))
    total = len(stl_files)
    capped = 0
    failed = 0

    for i, stl_path in enumerate(stl_files):
        name = os.path.splitext(os.path.basename(stl_path))[0]
        out_stl = os.path.join(dst_dir, f"{name}.stl")
        out_json = os.path.join(dst_dir, f"{name}_boundary.json")

        try:
            mesh = trimesh.load(stl_path, process=False)

            if not isinstance(mesh, trimesh.Trimesh):
                print(f"  [{i+1}/{total}] {name}: not a Trimesh, skipping")
                failed += 1
                continue

            # Basic repair
            mesh = repair_mesh(mesh)

            if mesh.is_watertight:
                # Already watertight, just copy
                mesh.export(out_stl)
                boundary_info = {
                    "n_faces": len(mesh.faces),
                    "cap_face_indices": [],
                    "wall_face_count": len(mesh.faces),
                    "cap_face_count": 0,
                    "was_already_watertight": True
                }
                with open(out_json, 'w') as f:
                    json.dump(boundary_info, f)
                capped += 1
            else:
                # Cap open boundaries
                capped_mesh, cap_mask, success = cap_mesh(mesh)

                if success and capped_mesh.is_watertight:
                    capped_mesh.export(out_stl)
                    cap_indices = np.where(cap_mask)[0].tolist()
                    boundary_info = {
                        "n_faces": len(capped_mesh.faces),
                        "cap_face_indices": cap_indices,
                        "wall_face_count": int((~cap_mask).sum()),
                        "cap_face_count": int(cap_mask.sum()),
                        "was_already_watertight": False
                    }
                    with open(out_json, 'w') as f:
                        json.dump(boundary_info, f)
                    capped += 1
                else:
                    print(f"  [{i+1}/{total}] {name}: capping failed (not watertight after cap)")
                    failed += 1

        except Exception as e:
            print(f"  [{i+1}/{total}] {name}: ERROR - {e}")
            failed += 1

        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"  [{dataset_name}] Progress: {i+1}/{total} (capped={capped}, failed={failed})")

    print(f"[{dataset_name}] Done: {capped} capped, {failed} failed, {total} total")
    return capped, failed, total


if __name__ == "__main__":
    datasets = ["4TCTA_AAA", "CMHA", "IntrA", "VMR",
                "AneuRisk", "Aneux", "Totalsegmentator"]

    print("=" * 60)
    print("Vascular Mesh Capping")
    print("=" * 60)

    total_capped = 0
    total_failed = 0
    total_all = 0

    for ds in datasets:
        c, f, t = process_dataset(ds)
        total_capped += c
        total_failed += f
        total_all += t

    print("=" * 60)
    print(f"Total: {total_capped} capped, {total_failed} failed, {total_all} total")
    print(f"Output: {DST}")
