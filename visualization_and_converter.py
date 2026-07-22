"""
=====================================================================
 BIM 3D VIEWER + ADAPTIVE IDF GENERATOR (v14 - EnergyPlus 26.1 build)
=====================================================================
Architecture Note: 
Implemented geometry priority: explicit IFC space-boundary geometry,
validated watertight IfcSpace mesh, then a reported low-confidence
bounding-box fallback. Unsupported physical-element cell reconstruction
is not claimed. Unsafe partial adjacencies are rejected and reported.
=====================================================================
"""
from __future__ import annotations

import io
import argparse
import os
import re
import copy
import glob
import math
import tempfile
import shutil
import subprocess
import traceback
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict, deque

import numpy as np
try:
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:  # UI dependencies are optional for batch/headless conversion.
    go = None
    st = None

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.unit
import ifcopenshell.util.placement
import ifcopenshell.util.element
from shapely.geometry import Polygon as ShapelyPolygon, box as ShapelyBox
from shapely.ops import triangulate as shapely_triangulate
from shapely.strtree import STRtree

import eppy
from eppy.modeleditor import IDF

# =====================================================================
# SECTION 1 — CONFIGURATION & CONSTANTS
# =====================================================================
if st is not None:
    st.set_page_config(page_title="BIM Viewer", page_icon="🏗️", layout="wide", initial_sidebar_state="collapsed")

IFC_TYPE_COLORS = {
    "IfcWall": "#5B8CCC", "IfcWallStandardCase": "#5B8CCC", "IfcSlab": "#8FBC8F", "IfcRoof": "#B07040",
    "IfcDoor": "#E8A030", "IfcWindow": "#88D4F5", "IfcSpace": "#00E5C0", "IfcColumn": "#AABBCC",
    "IfcCurtainWall": "#CCE8FF", "IfcFurnishingElement": "#D4A070"
}

IFC_TYPE_LABELS = {
    "IfcWall": "Walls", "IfcWallStandardCase": "Walls", "IfcSlab": "Slabs / Floors", "IfcRoof": "Roofs",
    "IfcDoor": "Doors", "IfcWindow": "Windows", "IfcSpace": "Spaces", "IfcCurtainWall": "Curtain Walls",
}

ROLE_FOR_TYPE = {
    "IfcWall": ("wall",), "IfcWallStandardCase": ("wall",), "IfcCurtainWall": ("wall",),
    "IfcRoof": ("roof",), "IfcSlab": ("roof", "floor"),
}

# OLD_FILE visualization lookup tables. These are intentionally isolated from
# the converter constants above.
VISUAL_IFC_TYPE_COLORS = {
    "IfcWall": "#5B8CCC", "IfcWallStandardCase": "#5B8CCC",
    "IfcSlab": "#8FBC8F", "IfcRoof": "#B07040",
    "IfcDoor": "#E8A030", "IfcWindow": "#88D4F5",
    "IfcStair": "#AA88CC", "IfcStairFlight": "#AA88CC",
    "IfcSpace": "#00E5C0", "IfcColumn": "#AABBCC", "IfcBeam": "#7799AA",
    "IfcCurtainWall": "#CCE8FF", "IfcRailing": "#CC8866",
    "IfcFurnishingElement": "#D4A070", "IfcPlate": "#99AABB", "IfcMember": "#668899",
}

MATERIAL_KEYWORD_COLORS = {
    "concrete": "#9BA8A0", "cement": "#9BA8A0",
    "brick": "#C4724A", "masonry": "#C4724A", "cmu": "#B09080",
    "glass": "#88D4F5", "glazing": "#88D4F5",
    "wood": "#C8A060", "timber": "#C8A060",
    "steel": "#AABBCC", "metal": "#AABBCC",
    "aluminum": "#C8D4DC", "aluminium": "#C8D4DC",
    "gypsum": "#E8E4DC", "plaster": "#E8E4DC", "drywall": "#E8E4DC",
    "insulation": "#F0D080", "membrane": "#A8C880", "roofing": "#B07040",
    "floor": "#8FBC8F", "carpet": "#B09878", "tile": "#D0C8B8",
    "stone": "#B8AFA0", "marble": "#E0D8D0", "party": "#5B8CCC",
}

IFC_TYPE_OPACITY = {
    "IfcSpace": 0.05, "IfcWindow": 0.40, "IfcCurtainWall": 0.40,
    "IfcDoor": 0.85, "IfcSlab": 0.90, "IfcRoof": 0.88,
    "IfcWall": 0.78, "IfcWallStandardCase": 0.78,
    "IfcColumn": 0.85, "IfcBeam": 0.85, "IfcStair": 0.85, "IfcStairFlight": 0.85,
    "IfcRailing": 0.80, "IfcFurnishingElement": 0.70,
}

VISUAL_IFC_TYPE_LABELS = {
    "IfcWall": "Walls", "IfcWallStandardCase": "Walls",
    "IfcSlab": "Slabs / Floors", "IfcRoof": "Roofs",
    "IfcDoor": "Doors", "IfcWindow": "Windows",
    "IfcStair": "Stairs", "IfcStairFlight": "Stairs",
    "IfcSpace": "Spaces", "IfcColumn": "Columns", "IfcBeam": "Beams",
    "IfcCurtainWall": "Curtain Walls", "IfcRailing": "Railings",
    "IfcFurnishingElement": "Furniture", "IfcPlate": "Plates", "IfcMember": "Members",
}

VISUAL_ROLE_FOR_TYPE = {
    "IfcWall": ("wall",), "IfcWallStandardCase": ("wall",), "IfcCurtainWall": ("wall",),
    "IfcRoof": ("roof",),
    "IfcSlab": ("roof", "floor"),
}

MATERIAL_THERMAL_PROPS = {
    "concrete":   ("MediumRough", 1.70, 2300, 900), "cement":     ("MediumRough", 1.70, 2300, 900),
    "brick":      ("MediumRough", 0.80, 1900, 800), "masonry":    ("MediumRough", 0.80, 1900, 800),
    "wood":       ("MediumSmooth", 0.15, 600, 1600), "timber":     ("MediumSmooth", 0.15, 600, 1600),
    "steel":      ("Smooth", 45.0, 7800, 500), "metal":      ("Smooth", 45.0, 7800, 500),
    "aluminum":   ("Smooth", 205.0, 2700, 900), "gypsum":     ("Smooth", 0.16, 800, 1090),
    "insulation": ("MediumRough", 0.04, 40, 1400), "roofing":    ("Rough", 0.16, 1100, 1460),
    "glass":      ("Smooth", 1.00, 2500, 750),
}
DEFAULT_MATERIAL_PROPS = ("MediumRough", 1.13, 1400, 1000)
DEFAULT_LAYER_THICKNESS = {"wall": 0.20, "roof": 0.15, "floor": 0.20}

GEOM_TOL, NORMAL_TOL, MIN_SURFACE_AREA, MIN_ZONE_VOLUME = 0.01, 0.02, 0.05, 0.10
OPENING_CONTAINMENT_TOL_M = 0.001
MAX_BUILDING_SPAN_M = 10000.0
MAX_BUILDING_HEIGHT_M = 1000.0
MIN_ADJACENCY_TOL_M = 0.03
MAX_ADJACENCY_TOL_M = 0.50
ADJACENCY_OVERLAP_RATIO = 0.98
MIN_FOOTPRINT_M, MIN_ROOM_HEIGHT_M, DEFAULT_ROOM_HEIGHT_M = 0.30, 0.50, 3.0

TARGET_ENERGYPLUS_VERSION = "26.1"
TARGET_ENERGYPLUS_RELEASE = "26.1.0"

_IDD_ALREADY_SET = None

# =====================================================================
# SECTION 2 — CORE IFC PARSING
# =====================================================================
def _load_ifc(file_bytes: bytes) -> ifcopenshell.file:
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        return ifcopenshell.open(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

def _length_scale_to_m(ifc) -> float:
    """Return the factor that converts native IFC length attributes to metres."""
    try:
        scale = float(ifcopenshell.util.unit.calculate_unit_scale(ifc))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid length scale {scale}")
        return scale
    except Exception as exc:
        raise ValueError("IFC LENGTHUNIT is missing or cannot be resolved safely") from exc

def _parent_storey(ifc, element) -> str:
    try:
        for rel in ifc.by_type("IfcRelContainedInSpatialStructure"):
            if element in rel.RelatedElements:
                c = rel.RelatingStructure
                if c.is_a("IfcBuildingStorey"): return c.Name or c.GlobalId
    except Exception: pass
    return "Unknown"

def _get_material_name(ifc, element) -> str:
    try:
        for rel in ifc.by_type("IfcRelAssociatesMaterial"):
            if element in rel.RelatedObjects:
                mat = rel.RelatingMaterial
                if hasattr(mat, "Name") and mat.Name: return mat.Name
                if mat.is_a("IfcMaterialLayerSetUsage") and mat.ForLayerSet and mat.ForLayerSet.MaterialLayers:
                    return mat.ForLayerSet.MaterialLayers[0].Material.Name
    except Exception: pass
    return ""

def _new_geometry_settings():
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    try:
        settings.set(settings.CONVERT_BACK_UNITS, False)
    except Exception:
        pass
    try:
        settings.set(settings.WELD_VERTICES, True)
    except Exception:
        pass
    return settings

def _tessellate(element, settings):
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        g = shape.geometry
        vertices = np.array(g.verts, dtype=np.float64).reshape(-1, 3)
        faces = np.array(g.faces, dtype=np.int32).reshape(-1, 3)
        return vertices, faces
    except Exception:
        return None

def _mesh_component_count(vertices: np.ndarray, faces: np.ndarray, tol=GEOM_TOL / 10) -> int:
    """Count face-connected shells after coordinate quantisation."""
    if len(faces) == 0:
        return 0
    q = np.round(vertices / max(tol, 1e-8)).astype(np.int64)
    edge_owners = defaultdict(list)
    for face_index, tri in enumerate(faces):
        pts = [tuple(q[int(i)]) for i in tri]
        if len(set(pts)) != 3:
            continue
        for a, b in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            edge_owners[tuple(sorted((a, b)))].append(face_index)
    neighbours = defaultdict(set)
    for owners in edge_owners.values():
        for owner in owners[1:]:
            neighbours[owners[0]].add(owner)
            neighbours[owner].add(owners[0])
    unseen = set(range(len(faces)))
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for neighbour in neighbours[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return components


def _mesh_is_watertight(vertices: np.ndarray, faces: np.ndarray, tol=GEOM_TOL / 10) -> bool:
    """Require a closed, edge-manifold, single-component shell."""
    if len(vertices) < 4 or len(faces) < 4:
        return False
    q = np.round(vertices / max(tol, 1e-8)).astype(np.int64)
    counts = Counter()
    for tri in faces:
        pts = [tuple(q[int(i)]) for i in tri]
        if len(set(pts)) != 3:
            continue
        for a, b in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            counts[tuple(sorted((a, b)))] += 1
    return (
        bool(counts)
        and all(count == 2 for count in counts.values())
        and _mesh_component_count(vertices, faces, tol) == 1
    )

def _boundary_role(boundary) -> tuple[str, str]:
    element = getattr(boundary, "RelatedBuildingElement", None)
    if element:
        if element.is_a("IfcWall") or element.is_a("IfcCurtainWall"):
            return "Wall", "wall"
        if element.is_a("IfcRoof"):
            return "Roof", "roof"
        if element.is_a("IfcSlab"):
            predefined = str(getattr(element, "PredefinedType", "") or "").upper()
            if predefined == "ROOF":
                return "Roof", "roof"
            if predefined in {"FLOOR", "BASESLAB"}:
                return "Floor", "floor"
    return "", ""


def _scaled_axis_placement(placement, native_scale_to_m: float) -> np.ndarray:
    matrix = np.asarray(
        ifcopenshell.util.placement.get_axis2placement(placement), dtype=float
    )
    matrix[:3, 3] *= native_scale_to_m
    return matrix


def _curve_points_native(curve) -> np.ndarray | None:
    """Extract common IFC 2D/3D polyline encodings without tessellation."""
    if curve is None:
        return None
    if curve.is_a("IfcPolyline"):
        coords = [list(point.Coordinates) for point in list(curve.Points or [])]
    elif curve.is_a("IfcIndexedPolyCurve"):
        point_list = getattr(curve, "Points", None)
        coords = list(getattr(point_list, "CoordList", []) or [])
    elif curve.is_a("IfcCompositeCurve"):
        pieces = []
        for segment in list(getattr(curve, "Segments", []) or []):
            points = _curve_points_native(getattr(segment, "ParentCurve", None))
            if points is None:
                return None
            if pieces and np.linalg.norm(pieces[-1][-1] - points[0]) <= GEOM_TOL:
                points = points[1:]
            pieces.append(points)
        return np.vstack(pieces) if pieces else None
    else:
        return None
    if len(coords) < 2:
        return None
    width = max(len(row) for row in coords)
    if width not in (2, 3):
        return None
    return np.asarray([list(row) + [0.0] * (3 - len(row)) for row in coords], dtype=float)


def _analytic_connection_polygons(item, native_scale_to_m: float) -> list[np.ndarray]:
    """Handle common space-boundary surfaces that OCC cannot tessellate."""
    if item.is_a("IfcSurfaceOfLinearExtrusion"):
        profile = getattr(item, "SweptCurve", None)
        curve = getattr(profile, "Curve", None) if profile else None
        if curve is None and profile:
            curve = getattr(profile, "OuterCurve", None)
        points = _curve_points_native(curve)
        if points is None:
            return []
        points *= native_scale_to_m
        profile_position = getattr(profile, "Position", None)
        if profile_position:
            matrix = _scaled_axis_placement(profile_position, native_scale_to_m)
            points = (np.column_stack((points, np.ones(len(points)))) @ matrix.T)[:, :3]
        surface_position = getattr(item, "Position", None)
        surface_matrix = (
            _scaled_axis_placement(surface_position, native_scale_to_m)
            if surface_position else np.eye(4)
        )
        points = (np.column_stack((points, np.ones(len(points)))) @ surface_matrix.T)[:, :3]
        direction = np.asarray(item.ExtrudedDirection.DirectionRatios, dtype=float)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        direction = surface_matrix[:3, :3] @ direction
        extrusion = direction * float(item.Depth) * native_scale_to_m
        polygons = []
        for index in range(len(points) - 1):
            polygon = np.asarray([
                points[index], points[index + 1],
                points[index + 1] + extrusion, points[index] + extrusion,
            ])
            if _polygon_area(polygon) >= MIN_SURFACE_AREA:
                polygons.append(polygon)
        return polygons

    if item.is_a("IfcCurveBoundedPlane"):
        points = _curve_points_native(getattr(item, "OuterBoundary", None))
        basis = getattr(item, "BasisSurface", None)
        position = getattr(basis, "Position", None) if basis else None
        if points is None or position is None:
            return []
        points *= native_scale_to_m
        matrix = _scaled_axis_placement(position, native_scale_to_m)
        polygon = (np.column_stack((points, np.ones(len(points)))) @ matrix.T)[:, :3]
        return [polygon]
    return []

def _build_space_boundary_geometry(space_element, settings, report, native_scale_to_m):
    """Use explicit IFC boundary geometry when IfcOpenShell can tessellate it.

    Unsupported connection-geometry encodings are reported and fall through to
    the space-shell strategy; they are never silently labelled as boundary data.
    """
    surfaces = []
    boundaries = list(getattr(space_element, "BoundedBy", []) or [])
    # IFC4 second-level boundaries supersede first-level boundaries where both
    # are exported. Prefer the most specific set to prevent double surfaces.
    second_level = [b for b in boundaries if b.is_a("IfcRelSpaceBoundary2ndLevel")]
    if second_level:
        boundaries = second_level
    seen_geometry = set()
    for boundary in boundaries:
        connection = getattr(boundary, "ConnectionGeometry", None)
        item = getattr(connection, "SurfaceOnRelatingElement", None) if connection else None
        if item is None:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, item)
            geometry = getattr(shape, "geometry", shape)
            vertices = np.asarray(geometry.verts, dtype=float).reshape(-1, 3)
            # Connection geometry is expressed in the relating space's object
            # coordinate system, unlike product meshes generated with world coords.
            placement = getattr(space_element, "ObjectPlacement", None)
            if placement:
                matrix = np.asarray(ifcopenshell.util.placement.get_local_placement(placement), dtype=float)
                # Connection geometry is tessellated in SI metres, but the
                # placement translation is returned in native IFC units.
                # Scale only translation; rotation is dimensionless.
                matrix[:3, 3] *= native_scale_to_m
                homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
                vertices = (homogeneous @ matrix.T)[:, :3]
            faces = np.asarray(geometry.faces, dtype=np.int32).reshape(-1, 3)
            polygons = _mesh_to_planar_polygons(
                vertices, faces, report, f"space boundary {boundary.GlobalId}"
            )
        except Exception as exc:
            polygons = _analytic_connection_polygons(item, native_scale_to_m)
            if not polygons:
                report["warnings"].append(
                    f"Space boundary {boundary.GlobalId} geometry unsupported: {exc}"
                )
                continue
            placement = getattr(space_element, "ObjectPlacement", None)
            if placement:
                matrix = np.asarray(
                    ifcopenshell.util.placement.get_local_placement(placement), dtype=float
                )
                matrix[:3, 3] *= native_scale_to_m
                polygons = [
                    (np.column_stack((polygon, np.ones(len(polygon)))) @ matrix.T)[:, :3]
                    for polygon in polygons
                ]
            report["analytic_space_boundaries_recovered"] += len(polygons)
        declared_type, declared_role = _boundary_role(boundary)
        for polygon in polygons:
            stype, role = (declared_type, declared_role) if declared_role else _classify_surface(polygon)
            q = np.round(np.asarray(polygon) / GEOM_TOL).astype(np.int64)
            signature = tuple(sorted(map(tuple, q)))
            if signature in seen_geometry:
                report["warnings"].append(
                    f"Duplicate space boundary geometry ignored: {boundary.GlobalId}"
                )
                continue
            seen_geometry.add(signature)
            surfaces.append({
                "surface_type": stype,
                "role": role,
                "vertices": polygon,
                "boundary_guid": boundary.GlobalId,
                "source_element": getattr(getattr(boundary, "RelatedBuildingElement", None), "GlobalId", None),
                "boundary_internal_external": str(getattr(boundary, "InternalOrExternalBoundary", "") or "").upper(),
                "boundary_physical_virtual": str(getattr(boundary, "PhysicalOrVirtualBoundary", "") or "").upper(),
                "corresponding_boundary": getattr(getattr(boundary, "CorrespondingBoundary", None), "GlobalId", None),
            })
    return surfaces

def _visual_parent_storey(ifc, element) -> str:
    try:
        for rel in ifc.by_type("IfcRelContainedInSpatialStructure"):
            if element in rel.RelatedElements:
                c = rel.RelatingStructure
                if c.is_a("IfcBuildingStorey"):
                    return c.Name or c.GlobalId
        for rel in ifc.by_type("IfcRelAggregates"):
            if hasattr(rel, "RelatedObjects") and element in rel.RelatedObjects:
                p = rel.RelatingObject
                if p.is_a("IfcBuildingStorey"):
                    return p.Name or p.GlobalId
    except Exception:
        pass
    return "Unknown"


def _visual_get_material_name(ifc, element) -> str:
    try:
        for rel in ifc.by_type("IfcRelAssociatesMaterial"):
            if element in rel.RelatedObjects:
                mat = rel.RelatingMaterial
                if hasattr(mat, "Name") and mat.Name:
                    return mat.Name
                if mat.is_a("IfcMaterialLayerSetUsage"):
                    ls = mat.ForLayerSet
                    if ls and ls.MaterialLayers:
                        m = ls.MaterialLayers[0].Material
                        if m and m.Name:
                            return m.Name
                if mat.is_a("IfcMaterialLayerSet"):
                    if mat.MaterialLayers:
                        m = mat.MaterialLayers[0].Material
                        if m and m.Name:
                            return m.Name
                if mat.is_a("IfcMaterialList"):
                    if mat.Materials and mat.Materials[0].Name:
                        return mat.Materials[0].Name
    except Exception:
        pass
    return ""


def _color_for_element(ifc, element, ifc_type: str) -> str:
    mat_name = _visual_get_material_name(ifc, element).lower()
    for keyword, color in MATERIAL_KEYWORD_COLORS.items():
        if keyword in mat_name:
            return color
    return VISUAL_IFC_TYPE_COLORS.get(ifc_type, "#8899AA")


def _visual_tessellate(element, settings):
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        g = shape.geometry
        vertices = np.array(g.verts, dtype=np.float32).reshape(-1, 3)
        faces = np.array(g.faces, dtype=np.int32).reshape(-1, 3)
        return vertices, faces
    except Exception:
        return None


def build_3d_traces(ifc, storey_filter=None, type_filter=None):
    """
    Walks through every wall/room/slab/etc. in the IFC file and turns
    each one into a Plotly 3D "trace" (a drawable mesh) with a color,
    hover tooltip, and so on.
    """
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    all_types = list(VISUAL_IFC_TYPE_LABELS.keys())
    if type_filter:
        all_types = [t for t in all_types if VISUAL_IFC_TYPE_LABELS.get(t, "") in type_filter]

    traces = []
    space_meta = []
    element_meta = {}
    seen_labels = set()

    for ifc_type in all_types:
        try:
            elements = ifc.by_type(ifc_type)
        except Exception:
            continue

        label = VISUAL_IFC_TYPE_LABELS.get(ifc_type, ifc_type)
        opacity = IFC_TYPE_OPACITY.get(ifc_type, 0.80)

        for el in elements:
            storey = _visual_parent_storey(ifc, el)
            if storey_filter and storey not in storey_filter:
                continue

            shape = _visual_tessellate(el, settings)
            if shape is None:
                continue
            vertices, faces = shape
            if not len(vertices) or not len(faces):
                continue

            color = _color_for_element(ifc, el, ifc_type)
            if ifc_type == "IfcSpace":
                color = "#C8FFF0"
            mat_name = _visual_get_material_name(ifc, el)
            el_name = getattr(el, "Name", None) or el.GlobalId

            hover_text = (
                f"<b>{label}</b><br>"
                f"Name: {el_name}<br>"
                f"Storey: {storey}<br>"
                + (f"Material: {mat_name}<br>" if mat_name else "")
                + f"ID: {el.GlobalId}"
            )

            trace = dict(
                type="mesh3d",
                x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(), z=vertices[:, 2].tolist(),
                i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
                color=color,
                opacity=opacity,
                name=label,
                legendgroup=label,
                showlegend=(label not in seen_labels),
                hovertemplate=hover_text + "<extra></extra>",
                flatshading=True,
                lighting=dict(ambient=0.65, diffuse=0.85, roughness=0.4,
                              specular=0.15, fresnel=0.05),
                lightposition=dict(x=1, y=2, z=3),
                meta=dict(global_id=el.GlobalId),
            )
            seen_labels.add(label)
            traces.append(trace)

            element_meta[el.GlobalId] = {
                "global_id": el.GlobalId,
                "ifc_type": ifc_type,
                "label": label,
                "name": el_name,
                "long_name": getattr(el, "LongName", "") or "",
                "storey": storey,
                "material": mat_name,
                "roles": VISUAL_ROLE_FOR_TYPE.get(ifc_type, ()),
            }

            if ifc_type == "IfcSpace":
                centroid = vertices.mean(axis=0).tolist()
                space_meta.append({
                    "global_id": el.GlobalId,
                    "name": el_name,
                    "long_name": getattr(el, "LongName", "") or "",
                    "storey": storey,
                    "centroid": centroid,
                    "material": mat_name,
                })

    return traces, space_meta, element_meta

def parse_ifc(file_bytes: bytes) -> dict:
    ifc = _load_ifc(file_bytes)
    native_scale_to_m = _length_scale_to_m(ifc)
    projects = ifc.by_type("IfcProject")
    storeys = [{"name": s.Name or s.GlobalId, "elevation": float(getattr(s, "Elevation", 0) or 0) * native_scale_to_m, "global_id": s.GlobalId} for s in ifc.by_type("IfcBuildingStorey")]
    storeys.sort(key=lambda x: x["elevation"])
    spaces = [{"global_id": sp.GlobalId, "name": sp.Name or sp.GlobalId, "long_name": getattr(sp, "LongName", "") or "", "storey": _parent_storey(ifc, sp)} for sp in ifc.by_type("IfcSpace")]
    return {"ifc": ifc, "storeys": storeys, "spaces": spaces, "project": projects[0].Name if projects else "Unnamed Project"}

# =====================================================================
# SECTION 3 — GEOMETRY RECONSTRUCTION & MATH
# =====================================================================
def _safe_name(value: str, prefix: str = "") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value or "Unnamed")).strip("_")
    return (prefix + cleaned)[:95] or (prefix + "Unnamed")

def _unique_name(existing: set, requested: str) -> str:
    name, counter = requested, 2
    while name in existing: name, counter = f"{requested}_{counter}", counter + 1
    existing.add(name)
    return name

def _polygon_normal(vertices: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=float)
    for i in range(len(vertices)): normal += np.cross(vertices[i], vertices[(i + 1) % len(vertices)])
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-12 else normal

def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3: return 0.0
    origin = vertices[0]
    return float(sum(np.linalg.norm(np.cross(vertices[i] - origin, vertices[i + 1] - origin)) / 2 for i in range(1, len(vertices) - 1)))


def _clean_polygon(vertices: np.ndarray) -> np.ndarray | None:
    """Return finite, non-repeating, planar vertices or ``None`` when unsafe."""
    polygon = np.asarray(vertices, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3:
        return None
    if not np.isfinite(polygon).all():
        return None

    cleaned = []
    for point in polygon:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > GEOM_TOL / 10:
            cleaned.append(point)
    if len(cleaned) > 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= GEOM_TOL / 10:
        cleaned.pop()
    if len(cleaned) < 3:
        return None

    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        for index in range(len(cleaned)):
            a = np.asarray(cleaned[index - 1])
            b = np.asarray(cleaned[index])
            c = np.asarray(cleaned[(index + 1) % len(cleaned)])
            chord = c - a
            distance = np.linalg.norm(np.cross(b - a, chord)) / max(np.linalg.norm(chord), 1e-12)
            if distance <= GEOM_TOL / 10:
                cleaned.pop(index)
                changed = True
                break

    polygon = np.asarray(cleaned, dtype=float)
    normal = _polygon_normal(polygon)
    if np.linalg.norm(normal) < 0.5 or _polygon_area(polygon) < MIN_SURFACE_AREA:
        return None
    plane_error = np.max(np.abs((polygon - polygon[0]) @ normal))
    if not math.isfinite(float(plane_error)) or plane_error > GEOM_TOL:
        return None
    return polygon


def _convex_parts(vertices: np.ndarray) -> list[np.ndarray]:
    """Keep convex faces intact and triangulate only genuinely concave faces."""
    normal = _polygon_normal(vertices)
    origin = vertices[0]
    points_2d = _project_to_plane_2d(vertices, origin, normal)
    polygon_2d = ShapelyPolygon(points_2d)
    if not polygon_2d.is_valid:
        polygon_2d = polygon_2d.buffer(0)
    if polygon_2d.geom_type != "Polygon" or polygon_2d.area <= MIN_SURFACE_AREA or polygon_2d.interiors:
        return []

    signs = []
    coords = np.asarray(polygon_2d.exterior.coords[:-1], dtype=float)
    for index in range(len(coords)):
        a, b, c = coords[index - 1], coords[index], coords[(index + 1) % len(coords)]
        ab, bc = b - a, c - b
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        if abs(cross) > 1e-10:
            signs.append(math.copysign(1.0, cross))
    if not signs or min(signs) == max(signs):
        return [vertices]

    u, v = _plane_basis(normal)
    parts = []
    for triangle in shapely_triangulate(polygon_2d):
        clipped = triangle.intersection(polygon_2d)
        if clipped.geom_type != "Polygon" or len(clipped.interiors):
            continue
        if clipped.area < MIN_SURFACE_AREA or clipped.area < triangle.area * 0.999999:
            continue
        tri_2d = np.asarray(clipped.exterior.coords[:-1], dtype=float)
        if len(tri_2d) != 3:
            continue
        tri_3d = origin + tri_2d[:, 0, None] * u + tri_2d[:, 1, None] * v
        if float(np.dot(_polygon_normal(tri_3d), normal)) < 0:
            tri_3d = tri_3d[::-1].copy()
        parts.append(tri_3d)
    return parts


def _orient_closed_shell(surfaces: list) -> list | None:
    """Orient a manifold polygon shell consistently, then outward by volume."""
    edge_owners = defaultdict(list)
    for face_index, surface in enumerate(surfaces):
        vertices = np.asarray(surface["vertices"], dtype=float)
        q = [tuple(row) for row in np.round(vertices / (GEOM_TOL / 5)).astype(np.int64)]
        for index, a in enumerate(q):
            b = q[(index + 1) % len(q)]
            if a == b:
                return None
            key = tuple(sorted((a, b)))
            direction = 1 if a < b else -1
            edge_owners[key].append((face_index, direction))
    if not edge_owners or any(len(owners) != 2 for owners in edge_owners.values()):
        return None

    neighbours = defaultdict(list)
    for owners in edge_owners.values():
        (a, da), (b, db) = owners
        parity = 1 if da == db else 0
        neighbours[a].append((b, parity))
        neighbours[b].append((a, parity))

    flips = {}
    for root in range(len(surfaces)):
        if root in flips:
            continue
        flips[root] = 0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour, parity in neighbours[current]:
                expected = flips[current] ^ parity
                if neighbour in flips and flips[neighbour] != expected:
                    return None
                if neighbour not in flips:
                    flips[neighbour] = expected
                    queue.append(neighbour)

    prepared = []
    for index, surface in enumerate(surfaces):
        item = dict(surface)
        polygon = np.asarray(surface["vertices"], dtype=float)
        item["vertices"] = polygon[::-1].copy() if flips[index] else polygon.copy()
        prepared.append(item)

    reference = np.vstack([item["vertices"] for item in prepared]).mean(axis=0)
    signed_volume = 0.0
    for item in prepared:
        polygon = item["vertices"] - reference
        for index in range(1, len(polygon) - 1):
            signed_volume += float(np.dot(polygon[0], np.cross(polygon[index], polygon[index + 1]))) / 6.0
    if abs(signed_volume) < MIN_ZONE_VOLUME:
        return None
    if signed_volume < 0:
        for item in prepared:
            item["vertices"] = item["vertices"][::-1].copy()
    return prepared


def _closed_shell_volume(surfaces: list) -> float:
    reference = np.vstack([np.asarray(item["vertices"], dtype=float) for item in surfaces]).mean(axis=0)
    volume = 0.0
    for item in surfaces:
        polygon = np.asarray(item["vertices"], dtype=float) - reference
        for index in range(1, len(polygon) - 1):
            volume += float(np.dot(polygon[0], np.cross(polygon[index], polygon[index + 1]))) / 6.0
    return abs(volume)


def _prepare_zone_shell(surfaces: list, report: dict, context: str) -> tuple[list | None, str | None]:
    """Clean, convexify, close, orient, and classify an EnergyPlus zone shell."""
    cleaned = []
    for surface in surfaces:
        polygon = _clean_polygon(surface.get("vertices"))
        if polygon is None:
            return None, "contains a degenerate, non-finite, repeated, or non-planar face"
        parts = _convex_parts(polygon)
        if not parts:
            return None, "contains a multiply-connected or unsupported polygon"
        if len(parts) > 1:
            report["concave_surfaces_triangulated"] += 1
        for part_index, part in enumerate(parts):
            item = dict(surface)
            item["vertices"] = part
            if len(parts) > 1:
                item["convex_part"] = part_index + 1
            cleaned.append(item)

    oriented = _orient_closed_shell(cleaned)
    if oriented is None or not _polygon_shell_is_closed(oriented):
        return None, "does not form one closed, consistently orientable manifold shell"

    counts = Counter()
    for item in oriented:
        surface_type, role = _classify_surface(item["vertices"])
        item["surface_type"] = surface_type
        item["role"] = role
        counts[role] += 1
    if counts["floor"] < 1 or counts["roof"] < 1 or counts["wall"] < 3:
        return None, "does not contain at least one floor, one upper face, and three walls"

    points = np.vstack([item["vertices"] for item in oriented])
    spans = points.max(axis=0) - points.min(axis=0)
    if spans[0] < MIN_FOOTPRINT_M or spans[1] < MIN_FOOTPRINT_M or spans[2] < MIN_ROOM_HEIGHT_M:
        return None, f"has physically implausible dimensions {spans.tolist()} m"
    if spans[0] > MAX_BUILDING_SPAN_M or spans[1] > MAX_BUILDING_SPAN_M or spans[2] > MAX_BUILDING_HEIGHT_M:
        return None, f"has physically implausible dimensions {spans.tolist()} m"
    return oriented, None

def _signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    return abs(sum(float(np.dot(vertices[tri[0]], np.cross(vertices[tri[1]], vertices[tri[2]]))) / 6.0 for tri in faces))

def _plane_key(triangle: np.ndarray):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = np.linalg.norm(normal)
    if length < 1e-10: return None
    normal = normal / length
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0: normal = -normal
    d = float(np.dot(normal, triangle[0]))
    return tuple(np.round(normal / NORMAL_TOL).astype(int)) + (int(round(d / GEOM_TOL)),)

def _component_boundary_loop(component_faces: list, vertices: np.ndarray):
    edge_counts = Counter()
    for face in component_faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[tuple(sorted((int(a), int(b))))] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if len(boundary_edges) < 3: return None

    adjacency = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b); adjacency[b].append(a)
    if any(len(n) != 2 for n in adjacency.values()): return None

    start = min(adjacency)
    loop, previous, current = [start], None, start
    for _ in range(len(boundary_edges) + 1):
        cands = adjacency[current]
        nxt = cands[0] if cands[0] != previous else cands[1]
        if nxt == start: break
        loop.append(nxt)
        previous, current = current, nxt
    
    if len(loop) < 3 or len(loop) != len(boundary_edges): return None
    polygon = vertices[np.array(loop, dtype=int)].astype(float)
    
    reduced = []
    for p in polygon:
        if not reduced or np.linalg.norm(p - reduced[-1]) > GEOM_TOL / 5: reduced.append(p)
    changed = True
    while changed and len(reduced) > 3:
        changed = False
        for i in range(len(reduced)):
            a, b, c = reduced[i - 1], reduced[i], reduced[(i + 1) % len(reduced)]
            line_length = max(np.linalg.norm(c - a), 1e-12)
            perpendicular_distance = np.linalg.norm(np.cross(b - a, c - a)) / line_length
            if perpendicular_distance <= GEOM_TOL:
                reduced.pop(i); changed = True; break
    return np.array(reduced, dtype=float) if len(reduced) >= 3 else None

def _mesh_to_planar_polygons(vertices: np.ndarray, faces: np.ndarray, report=None, context="mesh") -> list:
    grouped = defaultdict(list)
    for face in faces:
        key = _plane_key(vertices[face])
        if key is not None: grouped[key].append(tuple(int(i) for i in face))

    polygons = []
    for plane_faces in grouped.values():
        edge_to_faces = defaultdict(list)
        for index, face in enumerate(plane_faces):
            for edge in [tuple(sorted((face[0], face[1]))), tuple(sorted((face[1], face[2]))), tuple(sorted((face[2], face[0])))]:
                edge_to_faces[edge].append(index)
        
        neighbours = defaultdict(set)
        for owners in edge_to_faces.values():
            if len(owners) == 2:
                a, b = owners; neighbours[a].add(b); neighbours[b].add(a)

        unseen = set(range(len(plane_faces)))
        while unseen:
            root = unseen.pop()
            component = [root]
            queue = deque([root])
            while queue:
                item = queue.popleft()
                for n in neighbours[item]:
                    if n in unseen:
                        unseen.remove(n); component.append(n); queue.append(n)
            polygon = _component_boundary_loop([plane_faces[i] for i in component], vertices)
            if polygon is not None and _polygon_area(polygon) >= MIN_SURFACE_AREA:
                polygons.append(polygon)
            elif report is not None:
                report["dropped_polygon_components"] += 1
                report["warnings"].append(
                    f"{context}: planar component dropped because it is degenerate, multiply-connected, or has unsupported topology"
                )
    return polygons

def _polygon_shell_is_closed(surfaces: list, tol=GEOM_TOL / 5) -> bool:
    """Check closure of the actual reconstructed boundary-polygon set."""
    counts = Counter()
    for surface in surfaces:
        vertices = np.asarray(surface["vertices"], dtype=float)
        if len(vertices) < 3:
            return False
        q = np.round(vertices / max(tol, 1e-8)).astype(np.int64)
        for i in range(len(q)):
            a, b = tuple(q[i]), tuple(q[(i + 1) % len(q)])
            if a == b:
                continue
            counts[tuple(sorted((a, b)))] += 1
    return bool(counts) and all(count == 2 for count in counts.values())

def _orient_polygon_outward(polygon: np.ndarray, zone_centroid: np.ndarray) -> np.ndarray:
    normal = _polygon_normal(polygon)
    if np.dot(normal, polygon.mean(axis=0) - zone_centroid) < 0: return polygon[::-1].copy()
    return polygon

def _orient_and_classify_zone_surfaces(surfaces: list, zone_centroid: np.ndarray) -> list:
    """Return EnergyPlus-ready faces with normals pointing out of the zone.

    IFC slab semantics alone cannot distinguish a lower zone's ceiling from an
    upper zone's floor.  Classify horizontal faces from their outward normal
    after orientation instead of trusting the related element's PredefinedType.
    """
    prepared = []
    for surface in surfaces:
        item = dict(surface)
        polygon = _orient_polygon_outward(
            np.asarray(surface["vertices"], dtype=float), zone_centroid
        )
        surface_type, role = _classify_surface(polygon)
        item["vertices"] = polygon
        item["surface_type"] = surface_type
        item["role"] = role
        prepared.append(item)
    return prepared

def _classify_surface(polygon: np.ndarray) -> tuple:
    normal = _polygon_normal(polygon)
    if abs(float(normal[2])) < 0.35: return "Wall", "wall"
    if normal[2] > 0: return "Roof", "roof"
    return "Floor", "floor"

def _shoebox_polygons(bbox):
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    verts = [
        ("Floor", "floor", np.array([(min_x,min_y,min_z),(min_x,max_y,min_z),(max_x,max_y,min_z),(max_x,min_y,min_z)], float)),
        ("Roof", "roof", np.array([(min_x,min_y,max_z),(max_x,min_y,max_z),(max_x,max_y,max_z),(min_x,max_y,max_z)], float)),
        ("Wall", "wall", np.array([(min_x,min_y,min_z),(max_x,min_y,min_z),(max_x,min_y,max_z),(min_x,min_y,max_z)], float)),
        ("Wall", "wall", np.array([(max_x,max_y,min_z),(min_x,max_y,min_z),(min_x,max_y,max_z),(max_x,max_y,max_z)], float)),
        ("Wall", "wall", np.array([(max_x,min_y,min_z),(max_x,max_y,min_z),(max_x,max_y,max_z),(max_x,min_y,max_z)], float)),
        ("Wall", "wall", np.array([(min_x,max_y,min_z),(min_x,min_y,min_z),(min_x,min_y,max_z),(min_x,max_y,max_z)], float)),
    ]
    center = np.array([(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2])
    return [(t, r, _orient_polygon_outward(p, center)) for t, r, p in verts]

def _build_zone_geometry(space_element, settings, report, native_scale_to_m):
    boundary_surfaces = _build_space_boundary_geometry(
        space_element, settings, report, native_scale_to_m
    )
    shape = _tessellate(space_element, settings)
    boundary_points = None
    if boundary_surfaces:
        try:
            boundary_points = np.vstack([np.asarray(s["vertices"], dtype=float) for s in boundary_surfaces])
        except Exception:
            boundary_points = None

    mesh_available = shape is not None and len(shape[0]) and len(shape[1])
    if mesh_available:
        vertices, faces = shape[0].astype(float), shape[1]
        if not np.isfinite(vertices).all():
            mesh_available = False
            report["warnings"].append(
                f"{space_element.GlobalId}: IfcSpace mesh contains non-finite coordinates"
            )
    else:
        vertices, faces = np.empty((0, 3)), np.empty((0, 3), dtype=int)

    if mesh_available:
        bbox = tuple(vertices.min(axis=0).tolist() + vertices.max(axis=0).tolist())
    elif boundary_points is not None and len(boundary_points) and np.isfinite(boundary_points).all():
        bbox = tuple(boundary_points.min(axis=0).tolist() + boundary_points.max(axis=0).tolist())
    else:
        return None, "no_geometry", {}

    component_count = _mesh_component_count(vertices, faces) if mesh_available else 0
    watertight = _mesh_is_watertight(vertices, faces) if mesh_available else False
    mesh_volume = _signed_mesh_volume(vertices, faces) if watertight else 0.0
    if component_count > 1:
        report["warnings"].append(
            f"{space_element.GlobalId}: IfcSpace mesh has {component_count} disconnected shells; "
            "closed-mesh reconstruction was rejected"
        )

    # Tier 1: only a complete, manifold boundary shell is allowed through.
    if boundary_surfaces:
        prepared_boundaries, reason = _prepare_zone_shell(
            boundary_surfaces, report, f"IfcRelSpaceBoundary {space_element.GlobalId}"
        )
        if prepared_boundaries is not None:
            report["space_boundaries_used"] += len(prepared_boundaries)
            return prepared_boundaries, "ifc_space_boundaries", {
                "bbox": bbox, "mesh_volume": _closed_shell_volume(prepared_boundaries), "height_estimated": False,
                "watertight": watertight, "boundary_shell_closed": True,
                "mesh_component_count": component_count,
            }
        report["space_boundaries_rejected"] += len(boundary_surfaces)
        report["warnings"].append(
            f"{space_element.GlobalId}: complete-boundary test failed ({reason}); "
            "falling through to the IfcSpace mesh tier"
        )

    # Tier 2: a watertight IfcSpace mesh whose reconstructed polygon shell also
    # passes EnergyPlus-specific closure, normal, convexity, and role checks.
    if mesh_available and watertight and mesh_volume >= MIN_ZONE_VOLUME:
        polygons = _mesh_to_planar_polygons(
            vertices, faces, report, f"IfcSpace {space_element.GlobalId}"
        )
        mesh_surfaces = [{"vertices": polygon} for polygon in polygons]
        prepared_mesh, reason = _prepare_zone_shell(
            mesh_surfaces, report, f"IfcSpace {space_element.GlobalId}"
        )
        if prepared_mesh is not None:
            return prepared_mesh, "mesh_reconstruction", {
                "bbox": bbox, "mesh_volume": _closed_shell_volume(prepared_mesh),
                "height_estimated": False, "watertight": True,
                "mesh_component_count": component_count,
            }
        report["warnings"].append(
            f"{space_element.GlobalId}: watertight IfcSpace mesh rejected after planar-shell validation ({reason})"
        )

    # Tier 3: a closed bounding box. This is intentionally low confidence and
    # is never reported as preservation of the original room shape.
    height_estimated = False
    if bbox[5] - bbox[2] < MIN_ROOM_HEIGHT_M:
        bbox = (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[2] + DEFAULT_ROOM_HEIGHT_M)
        height_estimated = True
    fallback = [{"surface_type": t, "role": r, "vertices": p} for t, r, p in _shoebox_polygons(bbox)]
    prepared_fallback, reason = _prepare_zone_shell(
        fallback, report, f"bounding-box fallback {space_element.GlobalId}"
    )
    if prepared_fallback is None:
        return None, "invalid_bounding_box", {"bbox": bbox, "fallback_reason": reason}
    return prepared_fallback, "bounding_box_fallback", {
        "bbox": bbox, "mesh_volume": _closed_shell_volume(prepared_fallback),
        "height_estimated": height_estimated, "watertight": watertight,
        "mesh_component_count": component_count,
        "boundary_shell_closed": False,
        "fallback_reason": "No complete valid boundary shell or validated planar IfcSpace shell was available",
    }

# =====================================================================
# SECTION 4 — MATERIAL ASSIGNMENT (RESTORED)
# =====================================================================
def _match_material_props(material_name: str):
    name = (material_name or "").lower()
    for keyword, props in MATERIAL_THERMAL_PROPS.items():
        if keyword in name:
            return keyword, props, False
    return "generic", DEFAULT_MATERIAL_PROPS, True

def _get_or_create_layered_construction(idf, layers: list, role: str, cache: dict, report: dict) -> str:
    if not layers:
        layers = [{"name": "Generic", "thickness_m": DEFAULT_LAYER_THICKNESS[role], "inferred_thickness": True}]
    
    normalized = []
    for layer in layers:
        thickness = float(layer.get("thickness_m") or 0)
        inferred_thickness = thickness <= 0
        if inferred_thickness:
            thickness = DEFAULT_LAYER_THICKNESS.get(role, 0.20) / max(len(layers), 1)
        keyword, props, inferred_properties = _match_material_props(layer.get("name", ""))
        authored = layer.get("thermal_props") or {}
        if all(authored.get(key) is not None for key in ("conductivity", "density", "specific_heat")):
            props = (
                props[0], float(authored["conductivity"]),
                float(authored["density"]), float(authored["specific_heat"]),
            )
            inferred_properties = False
        normalized.append((layer.get("name", "Generic"), thickness, keyword, props, inferred_thickness, inferred_properties))

    cache_key = (role, tuple((row[2], round(row[1], 5), tuple(round(float(v), 6) if isinstance(v, (int, float)) else v for v in row[3])) for row in normalized))
    if cache_key in cache:
        return cache[cache_key]

    digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:10]
    construction_name = _safe_name(f"{role}_{'_'.join(row[2] for row in normalized)}_{digest}_construction")
    existing_materials = cache.setdefault("existing_materials", set())
    material_names = []
    
    for index, (original_name, thickness, keyword, props, inf_thick, inf_prop) in enumerate(normalized, 1):
        roughness, conductivity, density, specific_heat = props
        property_digest = hashlib.sha1(repr(props).encode("utf-8")).hexdigest()[:8]
        mat_name = _safe_name(f"{role}_{keyword}_{thickness:.4f}_{property_digest}_{index}_mat")
        if mat_name not in existing_materials:
            mat = idf.newidfobject("MATERIAL", Name=mat_name, Roughness=roughness, Thickness=thickness, Conductivity=conductivity, Density=density, Specific_Heat=specific_heat, Thermal_Absorptance=0.9, Solar_Absorptance=0.7, Visible_Absorptance=0.7)
            existing_materials.add(mat_name)
        material_names.append(mat_name)
        if inf_thick or inf_prop:
            report["material_assumptions"].append({
                "role": role, "ifc_material": original_name, "thickness_m": round(thickness, 5),
                "thickness_inferred": inf_thick, "thermal_properties_inferred": inf_prop,
            })

    construction = idf.newidfobject("CONSTRUCTION", Name=construction_name, Outside_Layer=material_names[0])
    for index, material_name in enumerate(material_names[1:], start=2):
        if f"Layer_{index}" in construction.fieldnames:
            setattr(construction, f"Layer_{index}", material_name)
            
    cache[cache_key] = construction_name
    return construction_name

# =====================================================================
# SECTION 5 — ADJACENCY, FENESTRATION & SPATIAL HASHING
# =====================================================================
def _plane_basis(normal: np.ndarray):
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    trial = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, trial)
    u = u / max(np.linalg.norm(u), 1e-12)
    return u, np.cross(normal, u)

def _project_to_plane_2d(vertices: np.ndarray, origin: np.ndarray, normal: np.ndarray):
    u, v = _plane_basis(normal)
    rel = vertices - origin
    return np.column_stack((rel @ u, rel @ v))


def _projected_surface_match(a: dict, b: dict, normal: np.ndarray):
    """Measure full-face correspondence without modifying either zone shell."""
    origin = np.asarray(a["vertices"][0], dtype=float)
    poly_a = ShapelyPolygon(
        _project_to_plane_2d(np.asarray(a["vertices"], dtype=float), origin, normal)
    ).buffer(0)
    poly_b = ShapelyPolygon(
        _project_to_plane_2d(np.asarray(b["vertices"], dtype=float), origin, normal)
    ).buffer(0)
    if (poly_a.geom_type != "Polygon" or poly_b.geom_type != "Polygon"
            or poly_a.area <= MIN_SURFACE_AREA or poly_b.area <= MIN_SURFACE_AREA
            or poly_a.interiors or poly_b.interiors):
        return False, 0.0, 0.0, None
    intersection = poly_a.intersection(poly_b)
    if intersection.geom_type != "Polygon" or intersection.interiors:
        return False, 0.0, 0.0, intersection
    overlap_a = intersection.area / max(poly_a.area, 1e-12)
    overlap_b = intersection.area / max(poly_b.area, 1e-12)
    hausdorff = float(poly_a.boundary.hausdorff_distance(poly_b.boundary))
    full = min(overlap_a, overlap_b) >= 0.999 and hausdorff <= 2 * GEOM_TOL
    return full, overlap_a, overlap_b, intersection

def _pair_corresponding_boundaries(all_surfaces: list, report: dict,
                                   max_thickness_m: float) -> set:
    """Pair reciprocal IFC 2nd-level boundaries before geometric inference."""
    by_guid = {s.get("boundary_guid"): s for s in all_surfaces if s.get("boundary_guid")}
    paired = set()
    for a in all_surfaces:
        target_guid = a.get("corresponding_boundary")
        if not target_guid or a["name"] in paired:
            continue
        b = by_guid.get(target_guid)
        if not b or b["zone_name"] == a["zone_name"] or b["name"] in paired:
            continue
        if b.get("corresponding_boundary") != a.get("boundary_guid"):
            report["warnings"].append(
                f"Non-reciprocal CorrespondingBoundary ignored: {a.get('boundary_guid')} -> {target_guid}"
            )
            continue
        na, nb = _polygon_normal(a["vertices"]), _polygon_normal(b["vertices"])
        if float(np.dot(na, nb)) > -0.90:
            report["warnings"].append(f"Corresponding boundaries have incompatible normals: {a['name']} / {b['name']}")
            continue
        separation = abs(float(np.dot(na, b["vertices"][0] - a["vertices"][0])))
        if separation > max_thickness_m:
            report["warnings"].append(
                f"Corresponding boundaries are {separation:.3f} m apart, beyond the "
                f"{max_thickness_m:.3f} m wall-thickness tolerance: {a['name']} / {b['name']}"
            )
            continue
        full, overlap_a, overlap_b, intersection = _projected_surface_match(a, b, na)
        if not full:
            report["partial_adjacencies_rejected"] += int(
                intersection is not None and getattr(intersection, "area", 0.0) > MIN_SURFACE_AREA
            )
            report["warnings"].append(
                f"Corresponding boundaries are not complete matching faces "
                f"({overlap_a:.1%}/{overlap_b:.1%}); subdivision required: {a['name']} / {b['name']}"
            )
            a["potential_internal"] = b["potential_internal"] = True
            continue
        a["outside_condition"] = b["outside_condition"] = "Surface"
        a["outside_object"], b["outside_object"] = b["name"], a["name"]
        if a["role"] in ("floor", "roof"):
            a["surface_type"] = "Ceiling" if a["role"] == "roof" else "Floor"
        if b["role"] in ("floor", "roof"):
            b["surface_type"] = "Ceiling" if b["role"] == "roof" else "Floor"
        paired.update((a["name"], b["name"]))
        report["shared_surfaces_matched"] += 1
        report["shared_surfaces_matched_by_correspondence"] += 1
    return paired

def _derive_adjacency_tolerance(element_cache: list) -> tuple[float, str]:
    thicknesses = []
    for item in element_cache:
        if item.get("type") not in {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}:
            continue
        total = sum(float(layer.get("thickness_m") or 0.0) for layer in item.get("layers", []))
        if 0.02 <= total <= MAX_ADJACENCY_TOL_M:
            thicknesses.append(total)
    if thicknesses:
        tolerance = float(np.percentile(thicknesses, 95)) + 2 * GEOM_TOL
        return min(max(tolerance, MIN_ADJACENCY_TOL_M), MAX_ADJACENCY_TOL_M), "IFC wall-layer thickness p95 + geometry tolerance"
    return 0.25, "conservative fallback; no reliable IFC wall-layer thicknesses"

def _match_adjacent_surfaces_overlap(all_surfaces: list, report: dict, max_thickness_m: float):
    plane_groups = defaultdict(list)
    for s in all_surfaces:
        n = _polygon_normal(s["vertices"])
        dom = int(np.argmax(np.abs(n)))
        sign = 1.0 if n[dom] > 0 else -1.0
        canonical_n = n * sign
        d = float(np.dot(n, s["vertices"][0]))
        key = tuple(np.round(canonical_n, 2))
        plane_groups[key].append({"surface": s, "n": n, "d": d, "sign": sign})

    paired_names = _pair_corresponding_boundaries(
        all_surfaces, report, max_thickness_m
    )
    for canonical_n, candidates in plane_groups.items():
        n_cand = len(candidates)
        if n_cand < 2: continue
        candidates.sort(key=lambda x: x["d"] * x["sign"])

        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                ca, cb = candidates[i], candidates[j]
                # Candidates are ordered on the canonical plane axis.
                if (cb["d"] * cb["sign"] - ca["d"] * ca["sign"]) > max_thickness_m:
                    break
                a, b = ca["surface"], cb["surface"]

                if a["zone_name"] == b["zone_name"] or a["name"] in paired_names or b["name"] in paired_names: continue
                if a["role"] != b["role"] and {a["role"], b["role"]} != {"floor", "roof"}: continue
                if float(np.dot(ca["n"], cb["n"])) > -0.95: continue 
                if abs(ca["d"] + cb["d"]) > max_thickness_m: continue
                
                try:
                    full, overlap_a, overlap_b, intersection = _projected_surface_match(
                        a, b, ca["n"]
                    )
                    intersection_area = float(getattr(intersection, "area", 0.0))
                    if intersection_area > MIN_SURFACE_AREA:
                        a["potential_internal"] = b["potential_internal"] = True
                    # Never declare a partial overlap as a whole-surface pair.
                    # Partial overlaps require surface subdivision, which this
                    # converter reports instead of producing a false adjacency.
                    if intersection_area > MIN_SURFACE_AREA and full:
                        a["outside_condition"] = b["outside_condition"] = "Surface"
                        a["outside_object"], b["outside_object"] = b["name"], a["name"]
                        if a["role"] in ("floor", "roof"): a["surface_type"] = "Ceiling" if a["role"] == "roof" else "Floor"
                        if b["role"] in ("floor", "roof"): b["surface_type"] = "Ceiling" if b["role"] == "roof" else "Floor"
                        paired_names.update((a["name"], b["name"]))
                        report["shared_surfaces_matched"] += 1
                    elif intersection_area > MIN_SURFACE_AREA:
                        report["partial_adjacencies_rejected"] += 1
                        report["warnings"].append(
                            f"Partial adjacency not paired ({overlap_a:.1%}/{overlap_b:.1%}): {a['name']} / {b['name']}"
                        )
                except Exception as e:
                    report["warnings"].append(f"Adjacency topology error between {a['name']} and {b['name']}: {str(e)}")
                    
    report["shared_surface_objects"] = len(paired_names)

def _finalize_unknown_wall_conditions(all_surfaces: list, report: dict):
    """Resolve lower-tier walls only after all adjacency evidence is evaluated."""
    for surface in all_surfaces:
        if surface.get("boundary_physical_virtual") == "VIRTUAL" and surface.get("outside_condition") != "Surface":
            surface["construction_name"] = surface.get("virtual_fallback_construction", surface["construction_name"])
            report["unpaired_virtual_boundaries"] += 1
            report["warnings"].append(
                f"Unpaired virtual boundary cannot use Construction:AirBoundary safely: {surface['name']}"
            )
        if (surface.get("boundary_internal_external") == "INTERNAL"
                and surface.get("outside_condition") != "Surface"):
            report["unresolved_declared_internal_surfaces"] += 1
            report["warnings"].append(
                f"Declared internal boundary has no safe reciprocal match and remains Adiabatic: {surface['name']}"
            )
        if surface.get("role") != "wall" or surface.get("boundary_internal_external"):
            continue
        if surface.get("outside_condition") == "Surface":
            continue
        if surface.get("potential_internal"):
            surface["outside_condition"] = "Adiabatic"
            report["unknown_walls_kept_adiabatic"] += 1
        else:
            surface["outside_condition"] = "Outdoors"
            report["unknown_walls_inferred_outdoors"] += 1


def _ensure_reverse_interzone_constructions(idf, all_surfaces: list, report: dict):
    """Assign a material-reversed construction to the opposite interzone face."""
    by_name = {surface["name"]: surface for surface in all_surfaces}
    constructions = {obj.Name: obj for obj in idf.idfobjects.get("CONSTRUCTION", [])}
    created = {}
    processed = set()

    def layers_for(construction):
        fields = ["Outside_Layer"] + [f"Layer_{index}" for index in range(2, 11)]
        return [getattr(construction, field, "") for field in fields if getattr(construction, field, "")]

    for surface in all_surfaces:
        if surface.get("outside_condition") != "Surface":
            continue
        other = by_name.get(surface.get("outside_object"))
        pair = tuple(sorted((surface["name"], other["name"]))) if other else None
        if not other or pair in processed:
            continue
        processed.add(pair)
        construction_name = surface.get("construction_name")
        if construction_name == "Project_Virtual_Air_Boundary":
            other["construction_name"] = construction_name
            continue
        construction = constructions.get(construction_name)
        if not construction:
            report["warnings"].append(
                f"Cannot build reverse interzone construction for {surface['name']}: {construction_name} is missing"
            )
            continue
        layers = layers_for(construction)
        if len(layers) <= 1:
            other["construction_name"] = construction_name
            continue
        if construction_name not in created:
            reverse_name = _safe_name(f"{construction_name}_Reverse")
            suffix = 2
            while reverse_name in constructions:
                reverse_name = _safe_name(f"{construction_name}_Reverse_{suffix}")
                suffix += 1
            reverse = idf.newidfobject(
                "CONSTRUCTION", Name=reverse_name, Outside_Layer=layers[-1]
            )
            for index, layer_name in enumerate(reversed(layers[:-1]), start=2):
                field = f"Layer_{index}"
                if field not in reverse.fieldnames:
                    raise ValueError(
                        f"EnergyPlus {TARGET_ENERGYPLUS_VERSION} Construction cannot serialize {len(layers)} layers"
                    )
                setattr(reverse, field, layer_name)
            constructions[reverse_name] = reverse
            created[construction_name] = reverse_name
        other["construction_name"] = created[construction_name]
        report["interzone_constructions_reconciled"] += 1


def _prune_unused_constructions(idf, all_surfaces: list, openings: list,
                                report: dict):
    """Remove converter-created constructions that no exported face references."""
    used = {
        item.get("construction_name")
        for item in [*all_surfaces, *openings]
        if item.get("construction_name")
    }
    removed = []
    for object_type in ("CONSTRUCTION", "CONSTRUCTION:AIRBOUNDARY"):
        for construction in list(idf.idfobjects.get(object_type, [])):
            if construction.Name not in used:
                removed.append(construction.Name)
                idf.removeidfobject(construction)
    report["unused_constructions_removed"] = len(removed)
    report["unused_construction_names_removed"] = removed

def _build_spatial_index(elements: list):
    geoms = [ShapelyBox(item["min"][0], item["min"][1], item["max"][0], item["max"][1]) for item in elements]
    return STRtree(geoms), geoms

def _nearest_element_indexed(surface: dict, elements: list, strtree: STRtree):
    center = surface["vertices"].mean(axis=0)
    role = surface["role"]
    valid_types = {"wall": {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}, "roof": {"IfcRoof", "IfcSlab"}, "floor": {"IfcSlab"}}[role]
    
    pt = ShapelyBox(center[0]-2.0, center[1]-2.0, center[0]+2.0, center[1]+2.0)
    candidates_idx = strtree.query(pt)
    
    best, best_score = None, float("inf")
    for idx in candidates_idx:
        item = elements[idx]
        if item["type"] not in valid_types: continue
        delta = np.maximum(np.maximum(item["min"] - center, center - item["max"]), 0)
        score = float(np.linalg.norm(delta)) + 0.05 * float(np.linalg.norm(item["center"] - center))
        if score < best_score:
            best_score, best = score, item
    return best if best_score <= 2.0 else None

def _map_openings_to_walls(ifc, settings, all_surfaces, surface_names, report,
                           coordinate_offset: np.ndarray):
    accepted = []
    surfaces_by_source_element = defaultdict(list)
    for surface in all_surfaces:
        if surface.get("source_element"): surfaces_by_source_element[surface["source_element"]].append(surface)

    for ifc_type in ("IfcWindow", "IfcDoor"):
        for element in ifc.by_type(ifc_type):
            counter_stem = ifc_type[3:].lower() + "s"

            def skip(reason):
                report[f"{counter_stem}_skipped"] += 1
                report["warnings"].append(f"{element.GlobalId}: {ifc_type} skipped: {reason}")

            parent_wall_guid = None
            opening_element = None
            try:
                for rel_fills in getattr(element, "FillsVoids", []):
                    opening = rel_fills.RelatingOpeningElement
                    opening_element = opening
                    for void_rel in getattr(opening, "VoidsElements", []):
                        parent_wall_guid = void_rel.RelatingBuildingElement.GlobalId
                        break
                    if parent_wall_guid: break
            except Exception as e:
                report["warnings"].append(f"Opening relational error {element.GlobalId}: {str(e)}")

            if not parent_wall_guid:
                skip("IfcRelFillsElement/IfcRelVoidsElement parent relation is missing")
                continue

            # The filling product mesh contains frames/panels and is not the
            # thermal aperture. Use the voiding IfcOpeningElement geometry.
            shape = _tessellate(opening_element, settings) if opening_element else None
            if not shape:
                skip("IfcOpeningElement aperture geometry could not be tessellated")
                continue
            opening_vertices = np.asarray(shape[0], dtype=float) - coordinate_offset
            polygons = _mesh_to_planar_polygons(
                opening_vertices, shape[1], report, f"IfcOpeningElement {opening_element.GlobalId}"
            )
            if not polygons:
                skip("no valid planar aperture face was reconstructed")
                continue
            candidate_surfaces = surfaces_by_source_element.get(parent_wall_guid, [])
            candidate_surfaces = [surface for surface in candidate_surfaces if surface.get("role") == "wall"]
            if not candidate_surfaces:
                skip("no exported wall face maps to the voiding parent element")
                continue
            best, best_polygon, best_score = None, None, None

            for wall in candidate_surfaces:
                wverts = wall["vertices"]
                wnormal = _polygon_normal(wverts)
                wall2d = _project_to_plane_2d(wverts, wverts[0], wnormal)
                for face_index, polygon in enumerate(polygons):
                    polygon = _clean_polygon(polygon)
                    if polygon is None:
                        continue
                    area = _polygon_area(polygon)
                    if area < 0.01:
                        continue
                    opening_normal = _polygon_normal(polygon)
                    if abs(float(np.dot(opening_normal, wnormal))) < 0.95:
                        continue
                    signed_distances = (polygon - wverts[0]) @ wnormal
                    plane_dist = float(np.max(np.abs(signed_distances)))
                    # Generous only for choosing between the two reveal faces.
                    if plane_dist > 0.50:
                        continue
                    projected = polygon - signed_distances[:, None] * wnormal
                    open2d = _project_to_plane_2d(projected, wverts[0], wnormal)
                    try:
                        wp = ShapelyPolygon(wall2d).buffer(0)
                        op = ShapelyPolygon(open2d).buffer(0)
                        if (wp.geom_type != "Polygon" or op.geom_type != "Polygon"
                                or wp.area <= 0 or op.area <= 0 or wp.interiors or op.interiors):
                            continue
                        # EnergyPlus requires true containment, not a high
                        # overlap percentage. A 95% test produced the CHKSBS
                        # Partial-Overlap/No-Overlap failures in the supplied log.
                        if not wp.covers(op) or op.difference(wp).area > 1e-10:
                            continue
                        # Deterministic ordering: containment, plane distance,
                        # larger aperture, then stable polygon iteration index.
                        score = (plane_dist, -area, face_index)
                        if best_score is None or score < best_score:
                            best, best_polygon, best_score = wall, projected, score
                    except Exception as exc:
                        report["warnings"].append(
                            f"Opening topology check failed {element.GlobalId}: {exc}"
                        )

            if best is None:
                skip("aperture is not fully contained in a coplanar exported wall face")
                continue

            # EnergyPlus forbids fenestration on adiabatic/ground surfaces.
            # Interzone openings are allowed only when the parent surface has
            # a valid reciprocal Surface boundary; those are paired below.
            if best.get("outside_condition") not in {"Outdoors", "Surface"}:
                report["invalid_openings_rejected"] += 1
                skip(f"parent {best['name']} is {best.get('outside_condition') or 'unresolved'}")
                continue
            if best.get("outside_condition") == "Surface" and not best.get("outside_object"):
                report["invalid_openings_rejected"] += 1
                skip(f"interzone parent {best['name']} has no reciprocal surface")
                continue

            polygon = best_polygon
            if float(np.dot(_polygon_normal(polygon), _polygon_normal(best["vertices"]))) < 0:
                polygon = polygon[::-1].copy()
                
            stype = "Door" if element.is_a("IfcDoor") else "Window"
            construction = "Project_Door_Generic" if stype == "Door" else "Project_Window_Generic"
            report["material_assumptions"].append({
                "role": stype.lower(), "ifc_material": "",
                "thermal_properties_inferred": True,
                "reason": "No validated IFC fenestration thermal-property mapping implemented",
                "element_guid": element.GlobalId,
            })
            name = _unique_name(surface_names, _safe_name(f"{best['zone_name']}_{stype}_{element.GlobalId[-8:]}"))
            accepted.append({
                "name": name, "surface_type": stype,
                "construction_name": construction,
                "parent_name": best["name"], "vertices": polygon,
                "element_guid": element.GlobalId,
            })
            report[f"{counter_stem}_converted"] += 1
    return accepted


def _opening_report_keys(opening: dict) -> tuple[str, str]:
    stem = "doors" if opening.get("surface_type") == "Door" else "windows"
    return f"{stem}_converted", f"{stem}_skipped"


def _reject_mapped_opening(opening: dict, report: dict, reason: str):
    converted_key, skipped_key = _opening_report_keys(opening)
    report[converted_key] = max(int(report.get(converted_key, 0)) - 1, 0)
    report[skipped_key] = int(report.get(skipped_key, 0)) + 1
    report["invalid_openings_rejected"] += 1
    element_guid = opening.get("element_guid")
    if element_guid:
        report["material_assumptions"] = [
            item for item in report.get("material_assumptions", [])
            if item.get("element_guid") != element_guid
        ]
    report["warnings"].append(f"{opening['name']}: {reason}")


def _opening_geometry_on_parent(vertices: np.ndarray, parent: dict):
    """Validate and orient a subsurface exactly on and inside its base face."""
    polygon = _clean_polygon(vertices)
    if polygon is None:
        return None, "opening polygon is degenerate, non-finite, repeated, or non-planar"
    # FenestrationSurface:Detailed in EnergyPlus 26.1 is limited to triangular
    # or quadrilateral subsurfaces.  Reject a more complex aperture instead of
    # serializing a subsurface that EnergyPlus cannot accept safely.
    if len(polygon) not in {3, 4}:
        return None, (
            f"opening has {len(polygon)} vertices; EnergyPlus 26.1 requires "
            "a 3- or 4-vertex FenestrationSurface:Detailed"
        )
    parent_vertices = np.asarray(parent["vertices"], dtype=float)
    parent_normal = _polygon_normal(parent_vertices)
    opening_normal = _polygon_normal(polygon)
    if abs(float(np.dot(parent_normal, opening_normal))) < 0.999:
        return None, "opening is not parallel to its parent"
    distances = (polygon - parent_vertices[0]) @ parent_normal
    if float(np.max(np.abs(distances))) > OPENING_CONTAINMENT_TOL_M:
        return None, "opening is not coplanar with its parent"
    polygon = polygon - distances[:, None] * parent_normal
    parent_2d = ShapelyPolygon(
        _project_to_plane_2d(parent_vertices, parent_vertices[0], parent_normal)
    ).buffer(0)
    opening_2d = ShapelyPolygon(
        _project_to_plane_2d(polygon, parent_vertices[0], parent_normal)
    ).buffer(0)
    if (parent_2d.geom_type != "Polygon" or opening_2d.geom_type != "Polygon"
            or parent_2d.interiors or opening_2d.interiors
            or opening_2d.area <= 0):
        return None, "opening or parent has unsupported polygon topology"
    if opening_2d.convex_hull.area - opening_2d.area > 1e-10:
        return None, "opening polygon is concave and cannot be exported safely"
    if not parent_2d.covers(opening_2d) or opening_2d.difference(parent_2d).area > 1e-10:
        return None, "opening is not fully contained in its parent"
    if float(np.dot(_polygon_normal(polygon), parent_normal)) < 0:
        polygon = polygon[::-1].copy()
    return polygon, None


def _prepare_openings_for_energyplus(openings: list, all_surfaces: list,
                                     surface_names: set, report: dict) -> list:
    """Keep exterior openings and make valid reciprocal interzone pairs.

    EnergyPlus requires a subsurface on each side of an interzone base surface,
    with each subsurface naming the other. IFC commonly stores only one filling
    element, so a reciprocal EnergyPlus subsurface is created when necessary.
    """
    parents = {surface["name"]: surface for surface in all_surfaces}
    result = []
    interzone = []

    for opening in openings:
        parent = parents.get(opening.get("parent_name"))
        if not parent:
            _reject_mapped_opening(opening, report, "parent surface is missing")
            continue
        polygon, reason = _opening_geometry_on_parent(opening["vertices"], parent)
        if polygon is None:
            _reject_mapped_opening(opening, report, reason)
            continue
        opening["vertices"] = polygon
        condition = parent.get("outside_condition")
        if condition == "Outdoors":
            opening["outside_object"] = ""
            result.append(opening)
        elif (condition == "Surface" and parent.get("outside_object") in parents
              and parents[parent["outside_object"]].get("outside_condition") == "Surface"
              and parents[parent["outside_object"]].get("outside_object") == parent["name"]):
            interzone.append(opening)
        else:
            _reject_mapped_opening(
                opening, report,
                f"parent surface condition {condition or 'unresolved'} cannot host fenestration",
            )

    used = set()
    for index, opening in enumerate(interzone):
        if index in used:
            continue
        parent = parents[opening["parent_name"]]
        reciprocal_parent_name = parent["outside_object"]
        reciprocal_parent = parents[reciprocal_parent_name]
        normal = _polygon_normal(parent["vertices"])
        origin = np.asarray(opening["vertices"][0], dtype=float)
        polygon_a = ShapelyPolygon(
            _project_to_plane_2d(opening["vertices"], origin, normal)
        ).buffer(0)

        best = None
        best_overlap = 0.0
        best_hausdorff = float("inf")
        best_intersection = None
        for other_index, other in enumerate(interzone):
            if other_index == index or other_index in used:
                continue
            if other.get("parent_name") != reciprocal_parent_name:
                continue
            if other.get("surface_type") != opening.get("surface_type"):
                continue
            polygon_b = ShapelyPolygon(
                _project_to_plane_2d(np.asarray(other["vertices"], dtype=float), origin, normal)
            ).buffer(0)
            if polygon_a.area <= 0 or polygon_b.area <= 0:
                continue
            intersection = polygon_a.intersection(polygon_b)
            overlap = min(
                intersection.area / polygon_a.area,
                intersection.area / polygon_b.area,
            )
            hausdorff = float(polygon_a.boundary.hausdorff_distance(polygon_b.boundary))
            if overlap > best_overlap or (overlap == best_overlap and hausdorff < best_hausdorff):
                best = other_index
                best_overlap = overlap
                best_hausdorff = hausdorff
                best_intersection = intersection

        if (best is not None and best_overlap >= 0.999
                and best_hausdorff <= OPENING_CONTAINMENT_TOL_M
                and best_intersection.geom_type == "Polygon"
                and not len(best_intersection.interiors)):
            other = interzone[best]
            other_polygon, reason = _opening_geometry_on_parent(
                other["vertices"], reciprocal_parent
            )
            if other_polygon is None:
                _reject_mapped_opening(other, report, reason)
                _reject_mapped_opening(opening, report, "reciprocal opening is invalid")
                used.update((index, best))
                continue
            other["vertices"] = other_polygon
            opening["outside_object"] = other["name"]
            other["outside_object"] = opening["name"]
            used.update((index, best))
            result.extend((opening, other))
            continue

        if best is not None and best_overlap > 0.05:
            other = interzone[best]
            _reject_mapped_opening(
                opening, report,
                f"overlapping reciprocal opening {other['name']} has incompatible geometry",
            )
            _reject_mapped_opening(
                other, report,
                f"overlapping reciprocal opening {opening['name']} has incompatible geometry",
            )
            used.update((index, best))
            continue

        if float(np.dot(_polygon_normal(parent["vertices"]), _polygon_normal(reciprocal_parent["vertices"]))) > -0.999:
            _reject_mapped_opening(
                opening, report, "reciprocal base surfaces do not have opposite normals"
            )
            used.add(index)
            continue
        separation = float(
            np.dot(normal, reciprocal_parent["vertices"][0] - opening["vertices"][0])
        )
        mirror_vertices = np.asarray(opening["vertices"], dtype=float) + separation * normal
        mirror_vertices, reason = _opening_geometry_on_parent(
            mirror_vertices, reciprocal_parent
        )
        if mirror_vertices is None:
            _reject_mapped_opening(
                opening, report, f"cannot create safe reciprocal opening: {reason}"
            )
            used.add(index)
            continue

        mirror_name = _unique_name(surface_names, _safe_name(f"{opening['name']}_Adjacent"))
        mirror = {
            **opening,
            "name": mirror_name,
            "parent_name": reciprocal_parent_name,
            "vertices": mirror_vertices,
            "outside_object": opening["name"],
            "mirrored_from": opening["name"],
        }
        opening["outside_object"] = mirror_name
        used.add(index)
        result.extend((opening, mirror))
        report["interzone_opening_mirrors_created"] += 1

    return result


def _validate_conversion_graph(all_surfaces: list, openings: list) -> dict:
    """Validate geometry and references before any EnergyPlus object is written."""
    errors = []
    checks = {
        "surface_reference_reciprocity": False,
        "subsurface_reference_reciprocity": False,
        "opening_parent_conditions": False,
        "opening_parent_containment": False,
        "zone_shell_closure": False,
        "floor_and_roof_normals": False,
        "finite_complete_vertices": False,
    }
    surface_names = [surface.get("name") for surface in all_surfaces]
    opening_names = [opening.get("name") for opening in openings]
    if len(set(surface_names)) != len(surface_names):
        errors.append("duplicate BuildingSurface:Detailed names exist")
    if len(set(opening_names)) != len(opening_names):
        errors.append("duplicate FenestrationSurface:Detailed names exist")

    surfaces = {surface["name"]: surface for surface in all_surfaces}
    subsurfaces = {opening["name"]: opening for opening in openings}
    by_zone = defaultdict(list)
    finite_ok = True
    normals_ok = True

    for surface in all_surfaces:
        by_zone[surface["zone_name"]].append(surface)
        vertices = np.asarray(surface.get("vertices"), dtype=float)
        if _clean_polygon(vertices) is None:
            errors.append(f"{surface['name']}: invalid or incomplete vertices")
            finite_ok = False
        normal = _polygon_normal(vertices)
        if surface.get("role") == "floor" and normal[2] >= -0.35:
            errors.append(f"{surface['name']}: floor normal is not downward")
            normals_ok = False
        if surface.get("role") == "roof" and normal[2] <= 0.35:
            errors.append(f"{surface['name']}: upper-surface normal is not upward")
            normals_ok = False
        condition = surface.get("outside_condition")
        target_name = surface.get("outside_object")
        if condition == "Surface":
            target = surfaces.get(target_name)
            if (not target or target.get("outside_condition") != "Surface"
                    or target.get("outside_object") != surface["name"]):
                errors.append(f"{surface['name']}: broken reciprocal interzone base-surface reference")
        elif target_name:
            errors.append(f"{surface['name']}: non-Surface boundary has an Outside Boundary Condition Object")

    shell_ok = True
    for zone_name, zone_surfaces in by_zone.items():
        roles = Counter(surface.get("role") for surface in zone_surfaces)
        if (not _polygon_shell_is_closed(zone_surfaces) or roles["floor"] < 1
                or roles["roof"] < 1 or roles["wall"] < 3):
            errors.append(f"{zone_name}: exported surfaces do not form a complete closed thermal-zone shell")
            shell_ok = False

    parent_rules_ok = True
    containment_ok = True
    for opening in openings:
        parent = surfaces.get(opening.get("parent_name"))
        if not parent or parent.get("outside_condition") not in {"Outdoors", "Surface"}:
            errors.append(f"{opening['name']}: missing parent or forbidden parent boundary condition")
            parent_rules_ok = False
            continue
        polygon, reason = _opening_geometry_on_parent(opening.get("vertices"), parent)
        if polygon is None:
            errors.append(f"{opening['name']}: {reason}")
            containment_ok = False
        target_name = opening.get("outside_object")
        if parent.get("outside_condition") == "Outdoors":
            if target_name:
                errors.append(f"{opening['name']}: exterior opening has an interzone reference")
                parent_rules_ok = False
            continue
        target = subsurfaces.get(target_name)
        reciprocal_parent = surfaces.get(parent.get("outside_object"))
        if (not target_name or not target or target.get("outside_object") != opening["name"]
                or not reciprocal_parent or target.get("parent_name") != reciprocal_parent["name"]):
            errors.append(f"{opening['name']}: broken reciprocal interzone subsurface reference")
            continue
        normal = _polygon_normal(parent["vertices"])
        origin = np.asarray(opening["vertices"][0], dtype=float)
        poly_a = ShapelyPolygon(_project_to_plane_2d(opening["vertices"], origin, normal)).buffer(0)
        poly_b = ShapelyPolygon(_project_to_plane_2d(target["vertices"], origin, normal)).buffer(0)
        intersection = poly_a.intersection(poly_b)
        overlap = min(
            intersection.area / max(poly_a.area, 1e-12),
            intersection.area / max(poly_b.area, 1e-12),
        )
        hausdorff = float(poly_a.boundary.hausdorff_distance(poly_b.boundary))
        if (overlap < 0.999 or hausdorff > OPENING_CONTAINMENT_TOL_M
                or float(np.dot(_polygon_normal(opening["vertices"]),
                                 _polygon_normal(target["vertices"]))) > -0.999):
            errors.append(f"{opening['name']}: reciprocal subsurface geometry does not match oppositely")

    checks["surface_reference_reciprocity"] = not any("base-surface reference" in error for error in errors)
    checks["subsurface_reference_reciprocity"] = not any("subsurface reference" in error for error in errors)
    checks["opening_parent_conditions"] = parent_rules_ok
    checks["opening_parent_containment"] = containment_ok
    checks["zone_shell_closure"] = shell_ok
    checks["floor_and_roof_normals"] = normals_ok
    checks["finite_complete_vertices"] = finite_ok
    return {"passed": not errors, "checks": checks, "errors": errors}

# =====================================================================
# SECTION 6 — ENERGYPLUS ENVIRONMENT & LOADS
# =====================================================================
def _idd_version(path: str) -> str | None:
    try:
        header = Path(path).read_text(encoding="utf-8", errors="ignore")[:4096]
        match = re.search(r"IDD[_\s-]*Version\s+([0-9]+\.[0-9]+)", header, re.I)
        if match:
            return match.group(1)
    except OSError:
        pass
    match = re.search(r"EnergyPlusV?([0-9]+)[-_.]([0-9]+)", str(path), re.I)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def _setup_idd(target_version: str = TARGET_ENERGYPLUS_VERSION) -> str:
    global _IDD_ALREADY_SET
    if _IDD_ALREADY_SET:
        loaded = _idd_version(_IDD_ALREADY_SET)
        if loaded != target_version:
            raise RuntimeError(
                f"Loaded EnergyPlus IDD {loaded or 'unknown'}, but this build requires {target_version}. "
                "Close Streamlit completely and restart it."
            )
        return loaded

    already_loaded = getattr(IDF, "iddname", None)
    if already_loaded:
        already_loaded = str(already_loaded)
        loaded = _idd_version(already_loaded)
        if loaded != target_version:
            raise RuntimeError(
                f"eppy already loaded IDD {loaded or 'unknown'} at {already_loaded}; "
                f"EnergyPlus {target_version} is required. Restart the Python process."
            )
        _IDD_ALREADY_SET = already_loaded
        return target_version

    candidates = []
    configured = os.environ.get("ENERGYPLUS_IDD")
    if configured:
        configured_path = Path(configured)
        candidates.append(str(configured_path / "Energy+.idd") if configured_path.is_dir() else configured)
    for variable in ("ENERGYPLUS_HOME", "ENERGYPLUS_INSTALL_DIR"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable]) / "Energy+.idd"))
    candidates.append(str(Path(__file__).resolve().parent / "Energy+.idd"))
    for pattern in [
        "C:/EnergyPlus*/Energy+.idd", "/usr/local/EnergyPlus*/Energy+.idd",
        "/opt/EnergyPlus*/Energy+.idd", "/Applications/EnergyPlus*/Energy+.idd",
    ]:
        candidates.extend(glob.glob(pattern))
    eppy_folder = Path(eppy.__file__).parent
    candidates.extend(str(p) for p in eppy_folder.glob("resources/iddfiles/Energy+V*.idd"))

    exact = sorted({str(Path(p)) for p in candidates if _idd_version(p) == target_version})
    if not exact:
        found = "; ".join(
            sorted({f"{p} ({_idd_version(p) or 'unknown'})" for p in candidates})
        ) or "none"
        raise FileNotFoundError(
            f"EnergyPlus {target_version} Energy+.idd was not found. Discovered: {found}. "
            f"Set ENERGYPLUS_IDD to your EnergyPlus {target_version} Energy+.idd path."
        )
    IDF.setiddname(exact[0])
    _IDD_ALREADY_SET = exact[0]
    return target_version


def _set_first_supported_field(obj, value, *names):
    for name in names:
        if name in obj.fieldnames:
            setattr(obj, name, value)
            return
    raise ValueError(f"{obj.key} does not contain any expected field: {', '.join(names)}")


def _find_energyplus_executable(target_version: str = TARGET_ENERGYPLUS_VERSION):
    candidates = []
    configured = os.environ.get("ENERGYPLUS_EXE")
    if configured:
        candidates.append(configured)
    for variable in ("ENERGYPLUS_HOME", "ENERGYPLUS_INSTALL_DIR"):
        if os.environ.get(variable):
            root = Path(os.environ[variable])
            candidates.extend((str(root / "energyplus"), str(root / "energyplus.exe")))
    if _IDD_ALREADY_SET:
        root = Path(_IDD_ALREADY_SET).resolve().parent
        candidates.extend((str(root / "energyplus"), str(root / "energyplus.exe")))
    for command in ("energyplus", "energyplus.exe"):
        path = shutil.which(command)
        if path:
            candidates.append(path)
    candidates.extend(glob.glob("C:/EnergyPlus*/energyplus.exe"))
    candidates.extend(glob.glob("/usr/local/EnergyPlus*/energyplus"))
    candidates.extend(glob.glob("/opt/EnergyPlus*/energyplus"))
    candidates.extend(glob.glob("/Applications/EnergyPlus*/energyplus"))
    for executable in sorted({str(Path(path)) for path in candidates if Path(path).is_file()}):
        try:
            result = subprocess.run(
                [executable, "--version"], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=15,
            )
            if (result.returncode == 0
                    and re.search(rf"(?<!\d){re.escape(target_version)}(?:\.0)?(?!\d)", result.stdout)):
                return executable, result.stdout.strip()
        except Exception:
            pass
    return None, ""


def _find_expandobjects_executable(energyplus_executable: str):
    """Find the ExpandObjects shipped with the validated EnergyPlus install."""
    energyplus_root = Path(energyplus_executable).resolve().parent
    candidates = []
    configured = os.environ.get("ENERGYPLUS_EXPANDOBJECTS")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        energyplus_root / "ExpandObjects.exe",
        energyplus_root / "ExpandObjects",
        energyplus_root / "PreProcess" / "ExpandObjects.exe",
        energyplus_root / "PreProcess" / "ExpandObjects",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _assert_serialized_schema(idf_path: str):
    text = Path(idf_path).read_text(encoding="utf-8", errors="replace")
    declared = re.findall(r"(?is)\bVERSION\s*,\s*([0-9]+(?:\.[0-9]+){1,2})\s*;", text)
    if declared != [TARGET_ENERGYPLUS_VERSION]:
        raise RuntimeError(
            f"Generated IDF must contain exactly one Version,{TARGET_ENERGYPLUS_VERSION}; "
            f"object; found {declared or 'none'}."
        )

    # Reparse the serialized file with the already-loaded target IDD.  This is
    # stronger and more maintainable than maintaining a list of fields that
    # happened to be absent from an older EnergyPlus release.
    try:
        reparsed = IDF(idf_path)
        version_objects = reparsed.idfobjects.get("VERSION", [])
        reparsed_version = (
            str(version_objects[0].Version_Identifier).strip()
            if len(version_objects) == 1 else None
        )
    except Exception as exc:
        raise RuntimeError(
            f"Generated IDF cannot be parsed by the EnergyPlus "
            f"{TARGET_ENERGYPLUS_VERSION} IDD: {exc}"
        ) from exc
    if reparsed_version != TARGET_ENERGYPLUS_VERSION:
        raise RuntimeError(
            f"Reparsed IDF declares {reparsed_version or 'no version'}, expected "
            f"{TARGET_ENERGYPLUS_VERSION}."
        )


def _energyplus_error_counts(err_text: str) -> dict:
    severe_lines = re.findall(r"(?im)^\s*\*\* Severe\s+\*\*.*$", err_text)
    fatal_lines = re.findall(r"(?im)^\s*\*\* Fatal\s+\*\*.*$", err_text)
    warning_lines = re.findall(r"(?im)^\s*\*\* Warning\s+\*\*.*$", err_text)
    return {
        "severe_errors": len(severe_lines),
        "fatal_errors": len(fatal_lines),
        "warnings": len(warning_lines),
        "severe_messages": severe_lines[:20],
        "fatal_messages": fatal_lines[:20],
    }


def _parse_epw_location(epw_path: str) -> dict:
    header = Path(epw_path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    fields = [field.strip() for field in header.split(",")]
    if len(fields) < 10 or fields[0].upper() != "LOCATION":
        raise ValueError("EPW LOCATION header is missing or malformed")
    return {
        "city": fields[1], "state": fields[2], "country": fields[3],
        "source": fields[4], "wmo": fields[5],
        "latitude_deg": float(fields[6]), "longitude_deg": float(fields[7]),
        "timezone_utc_offset_hours": float(fields[8]), "elevation_m": float(fields[9]),
    }


def _compare_epw_to_ifc(report: dict, epw_path: str) -> dict:
    epw = _parse_epw_location(epw_path)
    site = report.get("site_location", {})
    sources_present = all(
        not str(site.get(key, "")).startswith("missing")
        for key in ("latitude_source", "longitude_source", "elevation_source")
    )
    differences = {
        "latitude_deg": abs(epw["latitude_deg"] - float(site.get("ifc_latitude_deg", 0.0))),
        "longitude_deg": abs(epw["longitude_deg"] - float(site.get("ifc_longitude_deg", 0.0))),
        "timezone_hours": abs(epw["timezone_utc_offset_hours"] - float(site.get("expected_timezone_utc_offset_hours", 0.0))),
        "elevation_m": abs(epw["elevation_m"] - float(site.get("ifc_elevation_m", 0.0))),
    }
    mismatch_reasons = []
    if not sources_present:
        mismatch_reasons.append("IFC site coordinates/elevation are incomplete, so the weather match cannot be established")
    if differences["latitude_deg"] > 1.0:
        mismatch_reasons.append("latitude differs by more than 1 degree")
    if differences["longitude_deg"] > 1.0:
        mismatch_reasons.append("longitude differs by more than 1 degree")
    if differences["timezone_hours"] > 1.0:
        mismatch_reasons.append("timezone differs by more than 1 hour")
    if differences["elevation_m"] > 250.0:
        mismatch_reasons.append("elevation differs by more than 250 m")
    return {
        "epw_path": str(Path(epw_path)), "epw_location": epw,
        "differences": differences, "match": not mismatch_reasons,
        "mismatch_reasons": mismatch_reasons,
        "note": "Threshold match is a screening check, not proof that the EPW is climatologically representative.",
    }


def _run_energyplus_case(executable: str, idf_path: str, label: str,
                         extra_args: list[str]) -> dict:
    workdir = tempfile.mkdtemp(prefix=f"ifc_idf_{label}_")
    try:
        workpath = Path(workdir)
        in_idf = workpath / "in.idf"
        shutil.copy2(idf_path, in_idf)

        idd_candidates = [
            Path(_IDD_ALREADY_SET) if _IDD_ALREADY_SET else None,
            Path(executable).resolve().parent / "Energy+.idd",
        ]
        source_idd = next(
            (candidate for candidate in idd_candidates
             if candidate is not None and candidate.is_file()),
            None,
        )
        if source_idd is None:
            return {
                "attempted": True,
                "passed": False,
                "reason": (
                    f"ExpandObjects requires the exact EnergyPlus "
                    f"{TARGET_ENERGYPLUS_VERSION} Energy+.idd, but it was not found"
                ),
                "preprocessing": {"attempted": False, "passed": False},
            }
        shutil.copy2(source_idd, workpath / "Energy+.idd")

        expandobjects = _find_expandobjects_executable(executable)
        if expandobjects is None:
            return {
                "attempted": True,
                "passed": False,
                "reason": (
                    f"EnergyPlus {TARGET_ENERGYPLUS_VERSION} ExpandObjects executable was not found beside the validated "
                    "EnergyPlus executable; set ENERGYPLUS_EXPANDOBJECTS explicitly if needed"
                ),
                "preprocessing": {"attempted": False, "passed": False},
            }

        expand_proc = subprocess.run(
            [expandobjects], cwd=workdir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=120,
        )
        expanded_idf = workpath / "expanded.idf"
        preprocessing = {
            "attempted": True,
            "passed": expand_proc.returncode == 0 and expanded_idf.is_file(),
            "return_code": expand_proc.returncode,
            "executable": expandobjects,
            "idd_path": str(source_idd),
        }
        if not preprocessing["passed"]:
            preprocessor_error = workpath / "expandedidf.err"
            details = (
                preprocessor_error.read_text(encoding="utf-8", errors="replace")
                if preprocessor_error.is_file() else expand_proc.stdout
            )
            return {
                "attempted": True,
                "passed": False,
                "reason": f"EnergyPlus {TARGET_ENERGYPLUS_VERSION} ExpandObjects preprocessing failed",
                "preprocessing": preprocessing,
                "preprocessor_output": details[-4000:],
            }

        output_dir = workpath / "simulation"
        output_dir.mkdir()
        command = [
            executable, *extra_args, "--output-directory", str(output_dir),
            str(expanded_idf),
        ]
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300, cwd=workdir,
        )
        err_files = list(output_dir.glob("*.err"))
        err_text = (
            err_files[0].read_text(encoding="utf-8", errors="replace")
            if err_files else proc.stdout
        )
        counts = _energyplus_error_counts(err_text)
        return {
            "attempted": True,
            "passed": proc.returncode == 0 and counts["severe_errors"] == 0 and counts["fatal_errors"] == 0,
            "return_code": proc.returncode,
            "preprocessing": preprocessing,
            **counts,
        }
    except Exception as exc:
        return {"attempted": True, "passed": False, "reason": str(exc)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def _parse_ifc_angle(angle_tuple):
    if not angle_tuple:
        return 0.0
    values = [float(value) for value in angle_tuple]
    first_nonzero = next((value for value in values if value != 0), 0.0)
    sign = -1.0 if first_nonzero < 0 else 1.0
    degrees = abs(values[0])
    if len(values) > 1:
        degrees += abs(values[1]) / 60.0
    if len(values) > 2:
        degrees += abs(values[2]) / 3600.0
    if len(values) > 3:
        degrees += abs(values[3]) / 3_600_000_000.0
    return sign * degrees

def _get_site_location(ifc, report, native_scale_to_m):
    lat, lon, elev, tz = 0.0, 0.0, 0.0, 0.0
    latitude_present = longitude_present = elevation_present = False
    try:
        sites = ifc.by_type("IfcSite")
        if sites:
            site = sites[0]
            if getattr(site, "RefLatitude", None) is not None:
                lat = _parse_ifc_angle(site.RefLatitude)
                latitude_present = True
            if getattr(site, "RefLongitude", None) is not None:
                lon = _parse_ifc_angle(site.RefLongitude)
                longitude_present = True
            if getattr(site, "RefElevation", None) is not None:
                elev = float(site.RefElevation) * native_scale_to_m
                elevation_present = True
    except Exception as e:
        report["warnings"].append(f"Failed extracting IfcSite Location: {str(e)}")

    if longitude_present:
        tz = round((lon / 15.0) * 4.0) / 4.0
        report["warnings"].append(
            f"Expected Time_Zone estimated from IFC longitude as UTC{tz:+g}; "
            "longitude alone cannot determine the legal timezone or daylight-saving rules."
        )
    else:
        report["warnings"].append(
            "IFC longitude is missing; Site:Location longitude and timezone use explicit 0.0 placeholders."
        )
    if not latitude_present:
        report["warnings"].append(
            "IFC latitude is missing; Site:Location latitude uses an explicit 0.0 placeholder."
        )
    if not elevation_present:
        report["warnings"].append(
            "IFC reference elevation is missing; Site:Location elevation uses an explicit 0.0 m placeholder."
        )
    report["site_location"] = {
        "ifc_latitude_deg": lat,
        "ifc_longitude_deg": lon,
        "ifc_elevation_m": elev,
        "latitude_source": "IfcSite.RefLatitude" if latitude_present else "missing; placeholder 0.0",
        "longitude_source": "IfcSite.RefLongitude" if longitude_present else "missing; placeholder 0.0",
        "elevation_source": "IfcSite.RefElevation converted to metres" if elevation_present else "missing; placeholder 0.0",
        "expected_timezone_utc_offset_hours": tz,
        "timezone_source": "longitude-only estimate; verify manually" if longitude_present else "missing; placeholder 0.0",
    }
    return lat, lon, elev, tz

def _true_north_degrees(ifc) -> float:
    try:
        contexts = list(ifc.by_type("IfcGeometricRepresentationContext"))
        contexts.sort(key=lambda c: not (
            str(getattr(c, "ContextType", "") or "").lower() == "model"
            and int(getattr(c, "CoordinateSpaceDimension", 0) or 0) == 3
        ))
        for context in contexts:
            north = getattr(context, "TrueNorth", None)
            if north and getattr(north, "DirectionRatios", None):
                return float(math.degrees(math.atan2(north.DirectionRatios[0], north.DirectionRatios[1])))
    except Exception: pass
    return 0.0

def _material_thermal_properties(ifc, material) -> dict:
    """Read authored IFC material thermal properties when explicitly present."""
    if material is None:
        return {}
    aliases = {
        "thermalconductivity": "conductivity",
        "conductivity": "conductivity",
        "massdensity": "density",
        "density": "density",
        "specificheatcapacity": "specific_heat",
        "specificheat": "specific_heat",
    }
    result = {}
    try:
        for propset in ifc.by_type("IfcMaterialProperties"):
            if getattr(propset, "Material", None) != material:
                continue
            for prop in list(getattr(propset, "Properties", []) or []):
                key = aliases.get(re.sub(r"[^a-z]", "", str(getattr(prop, "Name", "")).lower()))
                nominal = getattr(prop, "NominalValue", None)
                value = getattr(nominal, "wrappedValue", nominal)
                if key and value is not None:
                    result[key] = float(value)
            for attr, key in (
                ("ThermalConductivity", "conductivity"),
                ("MassDensity", "density"),
                ("SpecificHeatCapacity", "specific_heat"),
            ):
                value = getattr(propset, attr, None)
                if value is not None:
                    result[key] = float(value)
    except Exception:
        return result
    return result

def _extract_material_layers(ifc, element, native_scale_to_m, report):
    layers = []
    try:
        materials = []
        resolved = ifcopenshell.util.element.get_material(
            element, should_skip_usage=False, should_inherit=True
        )
        if resolved:
            materials.append(resolved)
        for rel in getattr(element, "HasAssociations", []):
            if rel.is_a("IfcRelAssociatesMaterial") and rel.RelatingMaterial not in materials:
                materials.append(rel.RelatingMaterial)
        for mat in materials:
            layer_set = mat.ForLayerSet if mat.is_a("IfcMaterialLayerSetUsage") else (mat if mat.is_a("IfcMaterialLayerSet") else None)
            
            if layer_set:
                raw_layers = layer_set.MaterialLayers or []
                if mat.is_a("IfcMaterialLayerSetUsage") and getattr(mat, "DirectionSense", None) == "NEGATIVE":
                    raw_layers = reversed(raw_layers)

                for layer in raw_layers:
                    m = getattr(layer, "Material", None)
                    thickness = float(getattr(layer, "LayerThickness", 0) or 0) * native_scale_to_m
                    layers.append({
                        "name": getattr(m, "Name", "Generic"),
                        "thickness_m": thickness,
                        "inferred_thickness": thickness <= 0,
                        "thermal_props": _material_thermal_properties(ifc, m),
                    })
            if layers:
                break
            if mat.is_a("IfcMaterialConstituentSet"):
                constituents = list(getattr(mat, "MaterialConstituents", []) or [])
                for constituent in constituents:
                    material = getattr(constituent, "Material", None)
                    # Constituents have no ordered layer thickness. Preserve
                    # material identity but explicitly mark thickness inferred.
                    layers.append({
                        "name": getattr(material, "Name", None) or getattr(constituent, "Name", None) or "Generic",
                        "thickness_m": 0.0,
                        "inferred_thickness": True,
                        "source": "IfcMaterialConstituentSet",
                        "thermal_props": _material_thermal_properties(ifc, material),
                    })
                if layers:
                    report["warnings"].append(
                        f"{element.GlobalId}: IFC material constituents mapped, but ordered layer thicknesses are unavailable and were inferred"
                    )
                    break
    except Exception as e:
        report["warnings"].append(f"Material layer extraction failed for {element.GlobalId}: {str(e)}")
    return layers

def _initialise_idf(ifc, ifc_data, report, native_scale_to_m):
    idd_version = _setup_idd()
    report["energyplus_target_version"] = idd_version
    report["energyplus_idd_path"] = _IDD_ALREADY_SET
    idf = IDF(io.StringIO(""))
    idf.newidfobject("VERSION", Version_Identifier=idd_version)
    
    lat, lon, elev, tz = _get_site_location(ifc, report, native_scale_to_m)
    idf.newidfobject("SITE:LOCATION", Name="Project_Location", Latitude=lat, Longitude=lon, Time_Zone=tz, Elevation=elev)
    idf.newidfobject(
        "SIMULATIONCONTROL",
        Do_Zone_Sizing_Calculation="No",
        Do_System_Sizing_Calculation="No",
        Do_Plant_Sizing_Calculation="No",
        Run_Simulation_for_Sizing_Periods="Yes",
        Run_Simulation_for_Weather_File_Run_Periods="Yes",
    )
    idf.newidfobject("TIMESTEP", Number_of_Timesteps_per_Hour=4)
    idf.newidfobject(
        "SIZINGPERIOD:DESIGNDAY",
        Name="SummerDesignDay",
        Maximum_DryBulb_Temperature=30.0,
        Daily_DryBulb_Temperature_Range=10.0,
        Day_of_Month=21,
        Month=7,
        Day_Type="SummerDesignDay",
        Humidity_Condition_Type="WetBulb",
        Wetbulb_or_DewPoint_at_Maximum_DryBulb=23.0,
        Wind_Speed=3.0,
        Wind_Direction=0.0,
    )
    report["design_day"] = {
        "source": "generic completeness fallback; not derived from IFC or EPW",
        "maximum_dry_bulb_c": 30.0, "daily_range_c": 10.0,
        "wet_bulb_at_maximum_dry_bulb_c": 23.0,
        "wind_speed_m_s": 3.0, "wind_direction_deg": 0.0,
    }
    report["warnings"].append(
        "SizingPeriod:DesignDay is schema-complete but generic; replace it with site-specific ASHRAE/EPW design conditions before research simulation."
    )
    
    run = idf.newidfobject("RUNPERIOD", Name="Annual")
    run.Begin_Month, run.Begin_Day_of_Month, run.End_Month, run.End_Day_of_Month = 1, 1, 12, 31
    
    ground = idf.newidfobject("SITE:GROUNDTEMPERATURE:BUILDINGSURFACE")
    for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
        setattr(ground, f"{m}_Ground_Temperature", 18.0)
    report["warnings"].append("Assuming flat 18.0C ground temperature. Replace with site-appropriate GroundTemperature object prior to simulation.")

    idf.newidfobject("BUILDING", Name=ifc_data.get("project", "Baseline"), North_Axis=_true_north_degrees(ifc), Terrain="Suburbs")
    idf.newidfobject("GLOBALGEOMETRYRULES", Starting_Vertex_Position="UpperLeftCorner", Vertex_Entry_Direction="CounterClockWise", Coordinate_System="World")
    
    idf.newidfobject("SCHEDULETYPELIMITS", Name="Any Number")
    idf.newidfobject("SCHEDULETYPELIMITS", Name="Fraction", Lower_Limit_Value=0.0, Upper_Limit_Value=1.0, Numeric_Type="Continuous")
    
    for name, limit, val in [("OccupancySchedule", "Fraction", 0.5), ("EquipmentSchedule", "Fraction", 0.5), ("LightingSchedule", "Fraction", 0.6), ("HeatingSetpointSchedule", "Any Number", 21.0), ("CoolingSetpointSchedule", "Any Number", 24.0), ("ActivityLevelSchedule", "Any Number", 120.0)]:
        idf.newidfobject("SCHEDULE:CONSTANT", Name=name, Schedule_Type_Limits_Name=limit, Hourly_Value=val)
        
    idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name="BaselineThermostat", Heating_Setpoint_Schedule_Name="HeatingSetpointSchedule", Cooling_Setpoint_Schedule_Name="CoolingSetpointSchedule")
    idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", Name="Project_Simple_Glazing", UFactor=2.5, Solar_Heat_Gain_Coefficient=0.40, Visible_Transmittance=0.70)
    idf.newidfobject("CONSTRUCTION", Name="Project_Window_Generic", Outside_Layer="Project_Simple_Glazing")
    door_mat = idf.newidfobject("MATERIAL:NOMASS", Name="Project_Generic_Door_RValue", Thermal_Resistance=0.50, Roughness="MediumSmooth")
    idf.newidfobject("CONSTRUCTION", Name="Project_Door_Generic", Outside_Layer="Project_Generic_Door_RValue")
    try:
        idf.newidfobject("CONSTRUCTION:AIRBOUNDARY", Name="Project_Virtual_Air_Boundary")
        report["virtual_boundary_construction"] = "Construction:AirBoundary"
    except Exception as exc:
        report["virtual_boundary_construction"] = "unsupported by selected IDD"
        report["warnings"].append(f"Selected IDD has no usable Construction:AirBoundary object: {exc}")
    
    for var_name in ["Zone Ideal Loads Supply Air Total Heating Energy", "Zone Ideal Loads Supply Air Total Cooling Energy"]:
        idf.newidfobject("OUTPUT:VARIABLE", Key_Value="*", Variable_Name=var_name, Reporting_Frequency="Hourly")
    # EnergyPlus does not create eplustbl.* unless tabular reports are
    # explicitly requested. These fields are supported by the 26.1 IDD.
    idf.newidfobject(
        "OUTPUT:TABLE:SUMMARYREPORTS",
        Report_1_Name="AllSummary",
    )
    idf.newidfobject(
        "OUTPUTCONTROL:TABLE:STYLE",
        Column_Separator="HTML",
        Unit_Conversion="JtoKWH",
    )
    # Request at least one valid facility meter so EP-Launch also exposes the
    # meter results rather than leaving its Meters button disabled.
    idf.newidfobject(
        "OUTPUT:METER",
        Key_Name="Electricity:Facility",
        Reporting_Frequency="Hourly",
    )
    return idf

def _add_baseline_loads(idf, zone_name: str, floor_area_m2: float, space_name: str):
    name_lower = str(space_name).lower()
    if any(k in name_lower for k in ["stair", "corridor", "hall", "lobb"]): p_dens, l_dens, e_dens = 0.0, 5.0, 0.0
    elif any(k in name_lower for k in ["mech", "elec", "plant", "shaft"]): p_dens, l_dens, e_dens = 0.0, 5.0, 10.0
    elif any(k in name_lower for k in ["wc", "restroom", "bath"]): p_dens, l_dens, e_dens = 0.0, 5.0, 0.0
    else: p_dens, l_dens, e_dens = 20.0, 8.0, 5.0

    if p_dens > 0:
        people = idf.newidfobject("PEOPLE", Name=f"{zone_name}_People", Number_of_People_Calculation_Method="People", Number_of_People=round(max(floor_area_m2, 1.0) / p_dens, 2), Number_of_People_Schedule_Name="OccupancySchedule", Activity_Level_Schedule_Name="ActivityLevelSchedule")
        _set_first_supported_field(
            people, zone_name,
            "Zone_or_ZoneList_or_Space_or_SpaceList_Name",
            "Zone_or_ZoneList_Name",
        )
    else: people = None

    lights = idf.newidfobject("LIGHTS", Name=f"{zone_name}_Lights", Design_Level_Calculation_Method="LightingLevel", Lighting_Level=round(max(floor_area_m2, 1.0) * l_dens, 1), Schedule_Name="LightingSchedule")
    _set_first_supported_field(
        lights, zone_name,
        "Zone_or_ZoneList_or_Space_or_SpaceList_Name",
        "Zone_or_ZoneList_Name",
    )
    equipment = idf.newidfobject("ELECTRICEQUIPMENT", Name=f"{zone_name}_Equipment", Design_Level_Calculation_Method="EquipmentLevel", Design_Level=round(max(floor_area_m2, 1.0) * e_dens, 1), Schedule_Name="EquipmentSchedule")
    _set_first_supported_field(
        equipment, zone_name,
        "Zone_or_ZoneList_or_Space_or_SpaceList_Name",
        "Zone_or_ZoneList_Name",
    )
    # Do not automatically condition circulation/service zones that the same
    # use-condition heuristic classified as unoccupied.
    if p_dens > 0:
        idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", Zone_Name=zone_name, Template_Thermostat_Name="BaselineThermostat")
    return people, lights, equipment

# =====================================================================
# SECTION 7 — MAIN CONVERTER
# =====================================================================
def generate_baseline_idf_from_ifc(ifc, ifc_data: dict):
    settings = _new_geometry_settings()

    native_scale_to_m = _length_scale_to_m(ifc)
    report = {
        "warnings": [], "zones_created": 0, "zones_skipped": 0, "windows_converted": 0, 
        "windows_skipped": 0, "doors_converted": 0, "doors_skipped": 0,
        "shared_surfaces_matched": 0, "space_boundaries_used": 0,
        "space_boundaries_rejected": 0,
        "analytic_space_boundaries_recovered": 0,
        "shared_surfaces_matched_by_correspondence": 0,
        "partial_adjacencies_rejected": 0, "clipped_residual_area_m2": 0.0,
        "dropped_polygon_components": 0, "unknown_walls_kept_adiabatic": 0,
        "unknown_walls_inferred_outdoors": 0,
        "unpaired_virtual_boundaries": 0,
        "unresolved_declared_internal_surfaces": 0,
        "concave_surfaces_triangulated": 0,
        "interzone_constructions_reconciled": 0,
        "invalid_openings_rejected": 0,
        "interzone_opening_mirrors_created": 0,
        "high_confidence_zones": 0, "medium_confidence_zones": 0, "low_confidence_zones": 0, 
        "material_assumptions": [], "validation": {}, "zone_results": [],
        "ifc_length_scale_to_m": native_scale_to_m,
        "weld_vertices_requested": True,
        "geometry_hierarchy": [
            "IfcRelSpaceBoundary",
            "validated IfcSpace mesh",
            "reported bounding-box fallback",
        ],
        "unsupported_capabilities": [
            "physical-element closed-cell reconstruction",
            "partial-adjacency surface subdivision",
            "multiply-connected EnergyPlus surface decomposition",
        ],
    }
    idf = _initialise_idf(ifc, ifc_data, report, native_scale_to_m)

    # Element Cache & Spatial Index
    element_cache = []
    cached_element_guids = set()
    for el_type in ("IfcWall", "IfcWallStandardCase", "IfcCurtainWall", "IfcSlab", "IfcRoof"):
        for el in ifc.by_type(el_type):
            if el.GlobalId in cached_element_guids:
                continue
            cached_element_guids.add(el.GlobalId)
            try:
                shape = _tessellate(el, settings)
                if shape and len(shape[0]):
                    element_cache.append({
                        "element": el, "type": el_type, "min": shape[0].min(axis=0), 
                        "max": shape[0].max(axis=0), "center": shape[0].mean(axis=0), 
                        "layers": _extract_material_layers(ifc, el, native_scale_to_m, report),
                        "material_name": _get_material_name(ifc, el)
                    })
            except Exception as e:
                report["warnings"].append(f"Cache failed for {el.GlobalId}: {str(e)}")
                
    all_surfaces, surface_names, zone_names, zone_info, construction_cache = [], set(), set(), {}, {}
    prepared_zones = []

    for space in ifc_data.get("spaces", []):
        try:
            element = ifc.by_guid(space["global_id"])
            surfaces, source, metadata = _build_zone_geometry(
                element, settings, report, native_scale_to_m
            )
            if not surfaces:
                report["zones_skipped"] += 1
                continue
            prepared_zones.append((space, surfaces, source, metadata))
        except Exception as e:
            report["zones_skipped"] += 1
            report["warnings"].append(f"Space geometry failed {space['global_id']}: {str(e)}")

    if not prepared_zones: raise ValueError("No usable IfcSpace geometry converted.")

    original_min = np.array([
        min(meta["bbox"][0] for _, _, _, meta in prepared_zones),
        min(meta["bbox"][1] for _, _, _, meta in prepared_zones),
        min(meta["bbox"][2] for _, _, _, meta in prepared_zones),
    ], dtype=float)
    original_max = np.array([
        max(meta["bbox"][3] for _, _, _, meta in prepared_zones),
        max(meta["bbox"][4] for _, _, _, meta in prepared_zones),
        max(meta["bbox"][5] for _, _, _, meta in prepared_zones),
    ], dtype=float)
    original_spans = original_max - original_min
    if (original_spans[0] > MAX_BUILDING_SPAN_M
            or original_spans[1] > MAX_BUILDING_SPAN_M
            or original_spans[2] > MAX_BUILDING_HEIGHT_M):
        raise ValueError(
            f"Converted building extents are physically implausible ({original_spans.tolist()} m); "
            "IFC length units or placement translations are inconsistent."
        )

    # Rebase backend EnergyPlus geometry near the origin. This preserves all
    # relative dimensions while avoiding precision failures from georeferenced
    # eastings/northings. The Plotly UI continues using the original IFC mesh.
    coordinate_offset = original_min.copy()
    rebased_zones = []
    for space, surfaces, source, metadata in prepared_zones:
        for surface in surfaces:
            surface["vertices"] = np.asarray(surface["vertices"], dtype=float) - coordinate_offset
        bbox = metadata["bbox"]
        metadata = dict(metadata)
        metadata["bbox"] = (
            bbox[0] - coordinate_offset[0], bbox[1] - coordinate_offset[1], bbox[2] - coordinate_offset[2],
            bbox[3] - coordinate_offset[0], bbox[4] - coordinate_offset[1], bbox[5] - coordinate_offset[2],
        )
        rebased_zones.append((space, surfaces, source, metadata))
    prepared_zones = rebased_zones
    for item in element_cache:
        item["min"] = np.asarray(item["min"], dtype=float) - coordinate_offset
        item["max"] = np.asarray(item["max"], dtype=float) - coordinate_offset
        item["center"] = np.asarray(item["center"], dtype=float) - coordinate_offset
    strtree, _ = _build_spatial_index(element_cache) if element_cache else (None, None)

    building_min_x, building_min_y, building_min_z = 0.0, 0.0, 0.0
    building_max_x, building_max_y, building_max_z = original_spans.tolist()
    spans = {
        "x_m": building_max_x - building_min_x,
        "y_m": building_max_y - building_min_y,
        "z_m": building_max_z - building_min_z,
    }
    report["building_geometry_bounds_m"] = {
        "min": [building_min_x, building_min_y, building_min_z],
        "max": [building_max_x, building_max_y, building_max_z],
        "span": spans,
    }
    report["ifc_world_geometry_bounds_m"] = {
        "min": original_min.tolist(), "max": original_max.tolist(),
        "span": original_spans.tolist(),
    }
    report["energyplus_coordinate_offset_m"] = coordinate_offset.tolist()
    elevation_tol = max(0.03, min(0.15, max(building_max_z - building_min_z, 1.0) * 1e-4))

    for space, surfaces, geometry_source, metadata in prepared_zones:
        zone_name = _unique_name(zone_names, _safe_name(space.get("name"), "Zone_"))
        zone = idf.newidfobject("ZONE", Name=zone_name, Multiplier=1)
        bbox = metadata["bbox"]
        
        floor_area = sum(_polygon_area(s["vertices"]) for s in surfaces if s["role"] == "floor")
        if floor_area <= 0: floor_area = max((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]), 1.0)
        volume = float(metadata.get("mesh_volume") or 0.0)
        if volume < MIN_ZONE_VOLUME:
            volume = floor_area * max(bbox[5] - bbox[2], DEFAULT_ROOM_HEIGHT_M)
            report["warnings"].append(f"{zone_name}: volume inferred because no validated closed shell was available")
        bbox_volume = max((bbox[3]-bbox[0])*(bbox[4]-bbox[1])*(bbox[5]-bbox[2]), 1e-9)
        
        # Leave Zone volume, floor area, and ceiling height blank so EnergyPlus
        # calculates them from the validated closed shell. Serializing estimates
        # caused the supplied area/volume mismatch warnings.

        zone_surface_records = []
        mapped_material_count = 0
        
        for index, surface in enumerate(surfaces, 1):
            is_virtual = surface.get("boundary_physical_virtual") == "VIRTUAL"
            nearest = None if is_virtual else (_nearest_element_indexed(surface, element_cache, strtree) if strtree else None)
            layers = nearest["layers"] if nearest and nearest["layers"] else []
            if not layers and nearest and nearest.get("material_name"):
                layers = [{"name": nearest["material_name"], "thickness_m": 0, "inferred_thickness": True}]
            if nearest: mapped_material_count += 1
            
            virtual_fallback_construction = None
            if is_virtual:
                virtual_fallback_construction = _get_or_create_layered_construction(
                    idf, [], surface["role"], construction_cache, report
                )
            if is_virtual and report.get("virtual_boundary_construction") == "Construction:AirBoundary":
                construction_name = "Project_Virtual_Air_Boundary"
            else:
                construction_name = _get_or_create_layered_construction(idf, layers, surface["role"], construction_cache, report)
                if is_virtual:
                    report["warnings"].append(
                        f"{space['global_id']}: virtual boundary fell back to a generic construction because the selected IDD lacks Construction:AirBoundary"
                    )
            
            name = _unique_name(surface_names, _safe_name(f"{zone_name}_{surface['surface_type']}_{index}"))
            mean_z = float(surface["vertices"][:,2].mean())
            
            declared_boundary = surface.get("boundary_internal_external", "")
            if declared_boundary == "EXTERNAL":
                outside = "Outdoors"
            elif declared_boundary == "INTERNAL":
                outside = "Adiabatic"  # replaced with Surface only after safe reciprocal matching
            elif surface["role"] == "floor":
                outside = "Ground" if abs(mean_z-building_min_z) <= elevation_tol else "Adiabatic"
            elif surface["role"] == "roof":
                outside = "Outdoors" if abs(mean_z-building_max_z) <= elevation_tol else "Adiabatic"
            else:
                outside = "Adiabatic"  # lower-tier wall resolved only after adjacency evidence

            record = {**surface, "name": name, "zone_name": zone_name, "construction_name": construction_name, "outside_condition": outside, "outside_object": ""}
            if virtual_fallback_construction:
                record["virtual_fallback_construction"] = virtual_fallback_construction
            if not record.get("source_element"):
                record["source_element"] = nearest["element"].GlobalId if nearest else None
            all_surfaces.append(record)
            zone_surface_records.append(record)

        _add_baseline_loads(idf, zone_name, floor_area, space.get("long_name", space.get("name", "")))
        
        roles = Counter(s["role"] for s in zone_surface_records)
        mesh_ratio = min(max(volume / bbox_volume, 0.0), 1.0) if bbox_volume else 0.0
        material_ratio = mapped_material_count / max(len(zone_surface_records), 1)
        
        if geometry_source == "bounding_box_fallback": confidence = "low"
        elif geometry_source == "ifc_space_boundaries" and metadata.get("boundary_shell_closed") and material_ratio >= 0.5: confidence = "high"
        elif geometry_source == "mesh_reconstruction" and roles["wall"] >= 3 and roles["floor"] >= 1 and roles["roof"] >= 1 and mesh_ratio >= 0.35 and material_ratio >= 0.5: confidence = "high"
        else: confidence = "medium"
            
        report[f"{confidence}_confidence_zones"] += 1
        report["zones_created"] += 1
        
        zone_info[space["global_id"]] = {"zone_name": zone_name, "floor_area_m2": round(floor_area, 2), "volume_m3": round(volume, 2), "surface_count": len(surfaces), "confidence": confidence, "geometry_source": geometry_source}
        report["zone_results"].append({
            "ifc_space_guid": space["global_id"], "zone_name": zone_name,
            "geometry_source": geometry_source, "confidence": confidence,
            "watertight_source_mesh": bool(metadata.get("watertight", False)),
            "boundary_shell_closed": metadata.get("boundary_shell_closed"),
            "mesh_component_count": metadata.get("mesh_component_count"),
            "floor_area_m2": round(floor_area, 3), "volume_m3": round(volume, 3),
            "surface_count": len(surfaces), "material_mapping_ratio": round(material_ratio, 3),
            "fallback_reason": metadata.get("fallback_reason"),
        })

    adjacency_tolerance, tolerance_source = _derive_adjacency_tolerance(element_cache)
    report["adjacency_plane_tolerance_m"] = round(adjacency_tolerance, 4)
    report["adjacency_tolerance_source"] = tolerance_source
    if tolerance_source.startswith("conservative fallback"):
        report["warnings"].append(
            "Adjacency tolerance used a 0.25 m fallback because reliable IFC wall-layer thicknesses were unavailable"
        )
    _match_adjacent_surfaces_overlap(all_surfaces, report, max_thickness_m=adjacency_tolerance)
    _finalize_unknown_wall_conditions(all_surfaces, report)
    _ensure_reverse_interzone_constructions(idf, all_surfaces, report)
    openings = _map_openings_to_walls(
        ifc, settings, all_surfaces, surface_names, report, coordinate_offset
    )
    openings = _prepare_openings_for_energyplus(
        openings, all_surfaces, surface_names, report
    )
    _prune_unused_constructions(idf, all_surfaces, openings, report)

    structural_validation = _validate_conversion_graph(all_surfaces, openings)
    report["validation"]["pre_serialization_structure"] = structural_validation
    if not structural_validation["passed"]:
        summary = "; ".join(structural_validation["errors"][:8])
        raise ValueError(
            "Pre-serialization structural validation failed; invalid IDF was not offered: " + summary
        )

    for s in all_surfaces:
        obj = idf.newidfobject("BUILDINGSURFACE:DETAILED", Name=s["name"], Surface_Type=s["surface_type"], Construction_Name=s["construction_name"], Zone_Name=s["zone_name"], Outside_Boundary_Condition=s["outside_condition"], Sun_Exposure="SunExposed" if s["outside_condition"] == "Outdoors" else "NoSun", Wind_Exposure="WindExposed" if s["outside_condition"] == "Outdoors" else "NoWind")
        if s.get("outside_object"): obj.Outside_Boundary_Condition_Object = s["outside_object"]
        obj.Number_of_Vertices = len(s["vertices"])
        for i, (x, y, z) in enumerate(s["vertices"], 1):
            setattr(obj, f"Vertex_{i}_Xcoordinate", x); setattr(obj, f"Vertex_{i}_Ycoordinate", y); setattr(obj, f"Vertex_{i}_Zcoordinate", z)

    for o in openings:
        obj = idf.newidfobject("FENESTRATIONSURFACE:DETAILED", Name=o["name"], Surface_Type=o["surface_type"], Construction_Name=o["construction_name"], Building_Surface_Name=o["parent_name"], Number_of_Vertices=len(o["vertices"]))
        if o.get("outside_object"):
            obj.Outside_Boundary_Condition_Object = o["outside_object"]
        for i, (x, y, z) in enumerate(o["vertices"], 1):
            setattr(obj, f"Vertex_{i}_Xcoordinate", x); setattr(obj, f"Vertex_{i}_Ycoordinate", y); setattr(obj, f"Vertex_{i}_Zcoordinate", z)

    idf_path = tempfile.NamedTemporaryFile(suffix="_energyplus_26_1.idf", delete=False).name
    report_path = tempfile.NamedTemporaryFile(suffix="_conversion_report.json", delete=False).name
    idf.saveas(idf_path)
    _assert_serialized_schema(idf_path)
    report["validation"]["serialization_schema_26_1"] = {
        "passed": True,
        "idd_path": _IDD_ALREADY_SET,
        "version_object": TARGET_ENERGYPLUS_VERSION,
        "note": (
            f"Serialized by eppy using the exact EnergyPlus "
            f"{TARGET_ENERGYPLUS_VERSION} IDD and reparsed with the same schema."
        ),
    }
    
    executable, executable_version = _find_energyplus_executable()
    if executable:
        report["validation"]["energyplus_version"] = executable_version
        report["validation"]["energyplus_executable"] = executable
        design_day = _run_energyplus_case(
            executable, idf_path, "design_day", ["--design-day"]
        )
        report["validation"]["design_day_run"] = design_day
        report["validation"]["energyplus_run"] = design_day

        epw_path = os.environ.get("ENERGYPLUS_EPW")
        if epw_path:
            if Path(epw_path).is_file():
                try:
                    comparison = _compare_epw_to_ifc(report, epw_path)
                    report["validation"]["weather_file_comparison"] = comparison
                    report["validation"]["weather_run"] = _run_energyplus_case(
                        executable, idf_path, "weather", ["--weather", epw_path]
                    )
                except Exception as exc:
                    report["validation"]["weather_file_comparison"] = {
                        "match": False, "reason": str(exc), "epw_path": epw_path,
                    }
                    report["validation"]["weather_run"] = {
                        "attempted": False, "passed": False,
                        "reason": "EPW header could not be validated before simulation",
                    }
            else:
                report["validation"]["weather_run"] = {
                    "attempted": False, "passed": False,
                    "reason": f"ENERGYPLUS_EPW does not exist: {epw_path}",
                }
        else:
            report["validation"]["weather_run"] = {
                "attempted": False, "passed": False,
                "reason": "No EPW supplied through ENERGYPLUS_EPW; weather simulation was not attempted",
            }
    else:
        report["validation"]["energyplus_run"] = {
            "attempted": False, "passed": False,
            "reason": (
                f"Exact EnergyPlus {TARGET_ENERGYPLUS_RELEASE} executable not found; "
                "generated IDF is unvalidated"
            ),
        }
        report["validation"]["design_day_run"] = report["validation"]["energyplus_run"]
        report["validation"]["weather_run"] = {
            "attempted": False, "passed": False,
            "reason": f"Exact EnergyPlus {TARGET_ENERGYPLUS_RELEASE} executable not found",
        }

    run_passed = bool(report["validation"].get("design_day_run", {}).get("passed"))
    quality_failures = []
    if report.get("design_day", {}).get("source", "").startswith("generic"):
        quality_failures.append("design-day conditions are generic rather than site-specific")
    if report["low_confidence_zones"]:
        quality_failures.append("one or more zones use bounding-box fallback")
    if report["medium_confidence_zones"]:
        quality_failures.append("one or more zones did not satisfy every high-confidence criterion")
    if report["partial_adjacencies_rejected"]:
        quality_failures.append("one or more partial adjacencies require surface subdivision")
    if report["dropped_polygon_components"]:
        quality_failures.append("one or more planar components had unsupported topology and were dropped")
    if report["clipped_residual_area_m2"] > 0.01:
        quality_failures.append("adjacency clipping discarded more than 0.01 m2 of residual surface area")
    if any(z.get("geometry_source") == "ifc_space_boundaries" and not z.get("boundary_shell_closed") for z in report["zone_results"]):
        quality_failures.append("one or more IFC space-boundary polygon sets are not closed")
    if report["unknown_walls_kept_adiabatic"]:
        quality_failures.append("one or more unresolved lower-tier walls were conservatively kept adiabatic")
    if report["unknown_walls_inferred_outdoors"]:
        quality_failures.append("one or more wall boundary conditions were inferred as Outdoors")
    if report["unresolved_declared_internal_surfaces"]:
        quality_failures.append("one or more IFC-declared internal surfaces lack a reciprocal match")
    if report["unpaired_virtual_boundaries"]:
        quality_failures.append("one or more virtual boundaries lack a valid reciprocal pair")
    if report["invalid_openings_rejected"]:
        quality_failures.append("one or more openings could not be represented safely")
    if report["windows_skipped"] or report["doors_skipped"]:
        quality_failures.append("one or more IFC windows or doors were skipped")
    if report["material_assumptions"]:
        quality_failures.append("one or more material thicknesses or thermal properties were inferred")
    if report["zones_skipped"]:
        quality_failures.append("one or more IfcSpace objects were skipped")
    if not run_passed:
        quality_failures.append("EnergyPlus validation did not pass")
    weather_comparison = report["validation"].get("weather_file_comparison")
    if weather_comparison and not weather_comparison.get("match"):
        quality_failures.append("the validation EPW does not match the IFC site within screening thresholds")
    report["validation"]["research_quality_gate"] = {
        "passed": not quality_failures,
        "failures": quality_failures,
        "meaning": "A pass indicates structural conversion checks only; it is not empirical energy validation.",
    }

    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return idf_path, report_path, zone_info, {}, report

# =====================================================================
# SECTION 8 — STREAMLIT UI
# =====================================================================
def _scene_layout(height=650):
    """All the 3D-camera / background styling for the Plotly figure."""
    axis = dict(
        backgroundcolor="#0F172A", gridcolor="#334155", showbackground=True,
        zerolinecolor="#475569", tickfont=dict(color="#94A3B8", size=10), showspikes=False,
    )
    return dict(
        paper_bgcolor="#020617", plot_bgcolor="#020617", height=height,
        scene=dict(
            bgcolor="#0F172A",
            xaxis={**axis, "title": dict(text="", font=dict(color="#94A3B8", size=10))},
            yaxis={**axis, "title": dict(text="", font=dict(color="#94A3B8", size=10))},
            zaxis={**axis, "title": dict(text="Z (m)", font=dict(color="#94A3B8", size=10))},
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.7), up=dict(x=0, y=0, z=1),
                        projection=dict(type="perspective")),
            aspectmode="data",
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="bim-viewer",
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#38BDF8",
                        font=dict(color="#F8FAFC", size=12, family="Arial, sans-serif"),
                        namelength=-1),
        modebar=dict(bgcolor="#0F172A", color="#94A3B8", activecolor="#38BDF8", orientation="v"),
    )


def _render_dashboard_css():
    st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"] { background: #020617; color: #E2E8F0; }
  [data-testid="stSidebar"] { display: none !important; }
  html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Arial, sans-serif; color:#E2E8F0; }
  .app-header { text-align:center; padding:18px 14px 14px; border-bottom:1px solid #1E293B; }
  .app-title { font-size:1.4rem; font-weight:700; color:#F8FAFC; margin:0; }
  .app-title span { color:#38BDF8; }
  .app-subtitle { font-size:0.85rem; color:#94A3B8; margin-top:2px; }
  .file-badge { display:inline-flex; align-items:center; gap:6px; background:#0F172A;
    border:1px solid #334155; border-radius:6px; padding:5px 12px; font-size:0.82rem; color:#94A3B8; }
  .file-badge.on { border-color:#0EA5E9; background:#082F49; color:#7DD3FC; }
  .file-badge .led { width:6px; height:6px; border-radius:50%; background:#475569; }
  .file-badge.on .led { background:#38BDF8; }
  .file-label, .fbar-label { font-size:0.82rem; color:#94A3B8; font-weight:600; }
  .sstrip { display:flex; justify-content:center; gap:26px; flex-wrap:wrap; padding:10px 14px;
    background:#0F172A; border-bottom:1px solid #1E293B; }
  .sstat { font-size:0.85rem; color:#94A3B8; } .sstat b { color:#F8FAFC; }
  .lchip { display:inline-flex; align-items:center; gap:6px; background:#0F172A;
    border:1px solid #334155; border-radius:20px; padding:4px 11px; font-size:0.78rem; color:#CBD5E1; }
  .lswatch { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
  .sel-overlay { background:#0F172A; border:1px solid #334155; border-radius:8px;
    padding:12px 15px; margin:8px 0; font-size:0.88rem; color:#CBD5E1; line-height:1.6; }
  .sel-overlay b { color:#F8FAFC; } .sel-gid { color:#64748B; font-size:0.76rem; }
  div[data-testid="stFileUploader"] > label { display:none !important; }
  div[data-testid="stFileUploader"] section { padding:6px 10px !important; border-radius:6px !important;
    border:1px dashed #475569 !important; background:#0F172A !important; min-height:unset !important; }
  div[data-testid="stFileUploader"] section * { color:#CBD5E1 !important; }
  div[data-baseweb="select"] > div { background:#0F172A !important; border-color:#334155 !important; color:#E2E8F0 !important; }
  [data-baseweb="popover"], [data-baseweb="menu"] { background:#0F172A !important; color:#E2E8F0 !important; }
  [role="option"] { background:#0F172A !important; color:#E2E8F0 !important; }
  [role="option"]:hover { background:#1E293B !important; }
  .stMultiSelect [data-baseweb="tag"] { background:#082F49 !important; border:1px solid #0EA5E9 !important;
    color:#7DD3FC !important; border-radius:14px !important; }
  div[data-testid="stButton"] button { background:#0284C7 !important; border:1px solid #0284C7 !important;
    color:#FFFFFF !important; font-weight:600 !important; border-radius:6px !important; }
  div[data-testid="stButton"] button:hover { background:#0369A1 !important; border-color:#38BDF8 !important; }
  div[data-testid="stDownloadButton"] button { background:#1E293B !important; border:1px solid #475569 !important;
    color:#F8FAFC !important; font-weight:600 !important; width:100% !important; border-radius:6px !important; }
  div[data-testid="stDownloadButton"] button:hover { border-color:#38BDF8 !important; color:#7DD3FC !important; }
  [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] p { color:#CBD5E1; }
</style>
""", unsafe_allow_html=True)


def main():
    if st is None or go is None:
        raise RuntimeError("Streamlit UI requires the optional 'streamlit' and 'plotly' packages")
    _render_dashboard_css()

    DEFAULT_STATE = {
        "ifc_data": None,
        "ifc_traces": None,
        "space_meta": [],
        "element_meta": {},
        "selected_element_gid": None,
        "active_storeys": None,
        "active_types": None,
        "_ifc_key": None,
        "generated_idf_path": None,
        "generated_report_path": None,
        "idf_zone_info": None,
        "conversion_report": None,
    }
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    S = st.session_state

    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">🏗️ BIM <span>Viewer</span></div>'
        '<div class="app-subtitle">Upload an IFC model, explore it in 3D, and generate an adaptive EnergyPlus IDF · v14</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, col_upload, _ = st.columns([1, 2, 1], gap="small")
    with col_upload:
        st.markdown('<div style="padding-top:10px"><span class="file-label">IFC file</span></div>',
                    unsafe_allow_html=True)
        ifc_file = st.file_uploader("IFC", type=["ifc"], label_visibility="collapsed", key="ifc_upload_widget")
        has_file_badge = "on" if S["ifc_data"] else ""
        if S["ifc_data"]:
            badge_text = (f"✓ {S['ifc_data']['project']} — "
                           f"{len(S['ifc_data']['storeys'])} storeys, {len(S['ifc_data']['spaces'])} spaces")
        else:
            badge_text = "Drop an IFC file here"
        st.markdown(f'<div class="file-badge {has_file_badge}"><span class="led"></span>{badge_text}</div>',
                    unsafe_allow_html=True)

    if ifc_file:
        file_bytes = ifc_file.getvalue()
        file_key = hashlib.sha256(file_bytes).hexdigest()
        if S["_ifc_key"] != file_key:
            try:
                with st.spinner("Reading IFC file…"):
                    parsed = parse_ifc(file_bytes)
                    storey_names = [s["name"] for s in parsed["storeys"]]
                    type_labels = sorted(set(VISUAL_IFC_TYPE_LABELS.values()))
                    traces, spaces, elements = build_3d_traces(parsed["ifc"])
            except Exception as error:
                st.error(f"Couldn't read this IFC file: {error}")
                S["_ifc_key"] = file_key
                return
            S["ifc_data"] = parsed
            S["ifc_traces"] = traces
            S["space_meta"] = spaces
            S["element_meta"] = elements
            S["selected_element_gid"] = None
            S["active_storeys"] = storey_names[:]
            S["active_types"] = type_labels[:]
            S["generated_idf_path"] = None
            S["generated_report_path"] = None
            S["idf_zone_info"] = None
            S["conversion_report"] = None
            S["_ifc_key"] = file_key
            st.rerun()

    if not isinstance(S.get("ifc_data"), dict):
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:60vh;text-align:center;">
          <div style="font-size:2.6rem;margin-bottom:14px;opacity:0.5">🏗️</div>
          <div style="font-size:1.05rem;color:#CBD5E1;margin-bottom:6px;font-weight:600;">
            Drop an <span style="color:#38BDF8">IFC</span> file above to begin
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    ifc_d = S["ifc_data"]
    storey_names = [s["name"] for s in ifc_d["storeys"]]
    all_type_labels = sorted(set(VISUAL_IFC_TYPE_LABELS.values()))

    col_a, col_b, col_c, col_d = st.columns([0.3, 1.8, 0.28, 3.5], gap="small")
    with col_a:
        st.markdown('<div class="fbar-label" style="padding-top:8px">Floor</div>', unsafe_allow_html=True)
    with col_b:
        picked_storeys = st.multiselect(
            "floors", options=storey_names,
            default=[s for s in (S["active_storeys"] or storey_names) if s in storey_names],
            label_visibility="collapsed", key="floor_filter_widget",
        )
    with col_c:
        st.markdown('<div class="fbar-label" style="padding-top:8px">Types</div>', unsafe_allow_html=True)
    with col_d:
        picked_types = st.multiselect(
            "types", options=all_type_labels,
            default=[t for t in (S["active_types"] or all_type_labels) if t in all_type_labels],
            label_visibility="collapsed", key="type_filter_widget",
        )

    effective_storeys = picked_storeys if picked_storeys else storey_names
    effective_types = picked_types if picked_types else all_type_labels
    if (set(effective_storeys) != set(S["active_storeys"] or []) or
            set(effective_types) != set(S["active_types"] or [])):
        with st.spinner(f"Filtering — {len(effective_storeys)} floor(s), {len(effective_types)} type(s)…"):
            traces, spaces, elements = build_3d_traces(
                ifc_d["ifc"],
                storey_filter=effective_storeys if set(effective_storeys) != set(storey_names) else None,
                type_filter=effective_types if set(effective_types) != set(all_type_labels) else None,
            )
            S["ifc_traces"] = traces
            S["space_meta"] = spaces
            S["element_meta"] = elements
            S["active_storeys"] = effective_storeys[:]
            S["active_types"] = effective_types[:]
        st.rerun()

    st.markdown(
        f'<div class="sstrip">'
        f'<div class="sstat">Project: <b>{ifc_d["project"]}</b></div>'
        f'<div class="sstat">Storeys: <b>{len(ifc_d["storeys"])}</b></div>'
        f'<div class="sstat">Spaces: <b>{len(ifc_d["spaces"])}</b></div>'
        f'<div class="sstat">Meshes: <b>{len(S["ifc_traces"] or []):,}</b></div>'
        '</div>', unsafe_allow_html=True,
    )

    col_gen_button, col_gen_caption = st.columns([1.4, 3.6], gap="small")
    with col_gen_button:
        generate_clicked = st.button("⚡ Generate IDF from IFC", use_container_width=True)
    with col_gen_caption:
        st.caption("Runs the v14 adaptive space-boundary, adjacency, material, opening, confidence, and validation pipeline.")

    if generate_clicked:
        with st.spinner("Executing Physics & Geometry…"):
            try:
                idf_path, report_path, zone_info, _, report = generate_baseline_idf_from_ifc(ifc_d["ifc"], ifc_d)
                S["generated_idf_path"] = idf_path
                S["generated_report_path"] = report_path
                S["idf_zone_info"] = zone_info
                S["conversion_report"] = report
                st.success(f"Generated {report['zones_created']} Zones. {report['shared_surfaces_matched']} interior pairs matched.")
                gate = report.get("validation", {}).get("research_quality_gate", {})
                if not gate.get("passed"):
                    st.warning("Research quality gate failed: " + "; ".join(gate.get("failures", [])))
            except Exception:
                st.error(f"Generation Failed: {traceback.format_exc()}")

    if S.get("generated_idf_path"):
        col_idf, col_report = st.columns(2)
        with col_idf:
            st.download_button("⬇ Download generated IDF", Path(S["generated_idf_path"]).read_text(),
                               file_name="baseline_v14.idf", use_container_width=True)
        with col_report:
            st.download_button("⬇ Download conversion report", Path(S["generated_report_path"]).read_text(),
                               file_name="report_v14.json", use_container_width=True)

    if not S["ifc_traces"]:
        st.warning("No displayable meshes were found for the current filters.")
        return

    space_meta = S.get("space_meta") or []
    element_meta = S.get("element_meta") or {}
    if element_meta:
        NONE_LABEL = "— none selected —"

        def make_label(metadata):
            display_name = metadata.get("long_name") or metadata["name"]
            return f"{metadata['label']} · {display_name} ({metadata['storey']})"

        label_use_count = Counter(make_label(metadata) for metadata in element_meta.values())
        label_to_id = {NONE_LABEL: None}
        id_to_label = {}
        for gid, metadata in element_meta.items():
            base_label = make_label(metadata)
            final_label = base_label if label_use_count[base_label] == 1 else f"{base_label} [{gid[-6:]}]"
            label_to_id[final_label] = gid
            id_to_label[gid] = final_label
        dropdown_options = [NONE_LABEL] + sorted(label for label in label_to_id if label != NONE_LABEL)
        currently_selected_label = id_to_label.get(S["selected_element_gid"], NONE_LABEL)
        current_index = dropdown_options.index(currently_selected_label) if currently_selected_label in dropdown_options else 0
        col_find_label, col_find_box = st.columns([0.3, 4], gap="small")
        with col_find_label:
            st.markdown('<div class="fbar-label" style="padding-top:8px">Find</div>', unsafe_allow_html=True)
        with col_find_box:
            chosen_label = st.selectbox("find_element", dropdown_options, index=current_index,
                                        label_visibility="collapsed", key="find_element_widget")
        S["selected_element_gid"] = label_to_id.get(chosen_label)

    plotly_traces = []
    for trace_data in copy.deepcopy(S["ifc_traces"]):
        trace_data.pop("type", None)
        this_gid = trace_data.pop("global_id", None)
        if this_gid and this_gid == S["selected_element_gid"]:
            trace_data["color"] = "#FFE040"
            trace_data["opacity"] = 0.85
        plotly_traces.append(go.Mesh3d(**trace_data))

    if space_meta:
        dot_x, dot_y, dot_z, dot_color, dot_size, dot_text, dot_hover = [], [], [], [], [], [], []
        all_zone_info_for_hover = S.get("idf_zone_info")
        for space in space_meta:
            dot_x.append(space["centroid"][0])
            dot_y.append(space["centroid"][1])
            dot_z.append(space["centroid"][2])
            is_selected = space["global_id"] == S["selected_element_gid"]
            dot_color.append("#FFE040" if is_selected else "#FF2222")
            dot_size.append(28 if is_selected else 22)
            display_name = space.get("long_name") or space["name"]
            dot_text.append(display_name)
            zone = all_zone_info_for_hover.get(space["global_id"]) if all_zone_info_for_hover else None
            zone_line = ""
            if zone:
                zone_line = (f"<br><b>IDF Zone:</b> {zone['zone_name']}<br>"
                             f"Floor area: {zone['floor_area_m2']} m² · Volume: {zone['volume_m3']} m³<br>"
                             f"Confidence: {zone['confidence']} · Source: {zone['geometry_source']}")
            dot_hover.append(f"<b>{display_name}</b><br><b>Floor:</b> {space['storey']}<br>"
                             f"<span style='color:#9CA3AF'>{space['global_id']}</span>{zone_line}")
        plotly_traces.append(go.Scatter3d(
            x=dot_x, y=dot_y, z=dot_z, mode="markers+text",
            marker=dict(size=dot_size, color=dot_color, symbol="circle", opacity=1.0,
                        line=dict(color="#FFFFFF", width=2.5)),
            text=dot_text, textposition="top center", textfont=dict(color="#FFFFFF", size=9, family="monospace"),
            hovertemplate="%{customdata}<extra></extra>", customdata=dot_hover,
            name="Spaces", legendgroup="Spaces", showlegend=True,
        ))

    figure = go.Figure(data=plotly_traces)
    figure.update_layout(**_scene_layout(height=650))
    col_viewer, col_legend = st.columns([4, 1], gap="medium")
    with col_viewer:
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    with col_legend:
        st.markdown('<div class="fbar-label" style="margin-bottom:6px;">Legend</div>', unsafe_allow_html=True)
        seen_legend = {}
        for trace in S["ifc_traces"]:
            if trace.get("name", "?") not in seen_legend:
                seen_legend[trace.get("name", "?")] = trace.get("color", "#888")
        legend_rows_html = "".join(
            f'<div class="lchip" style="display:flex;width:100%;margin-bottom:6px;">'
            f'<div class="lswatch" style="background:{color}"></div>{label}</div>'
            for label, color in sorted(seen_legend.items())
        )
        st.markdown(legend_rows_html, unsafe_allow_html=True)

    selected_gid = S["selected_element_gid"]
    selected_element = element_meta.get(selected_gid) if selected_gid else None
    if selected_element:
        display_name = selected_element.get("long_name") or selected_element["name"]
        material_suffix = f" · {selected_element['material']}" if selected_element.get("material") else ""
        st.markdown(
            f'<div class="sel-overlay"><b>▶ {display_name}</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'{selected_element["label"]}&nbsp;&nbsp;·&nbsp;&nbsp;{selected_element["storey"]}{material_suffix}'
            f'<br><span class="sel-gid">{selected_element["global_id"]}</span></div>',
            unsafe_allow_html=True,
        )
        if selected_element["ifc_type"] == "IfcSpace":
            zone = (S.get("idf_zone_info") or {}).get(selected_gid)
            if zone:
                st.markdown(
                    f'<div class="sel-overlay" style="background:#082F49;border-color:#0EA5E9;">'
                    f'<b>⚡ IDF Zone: {zone["zone_name"]}</b><br>'
                    f'Floor area: <b>{zone["floor_area_m2"]} m²</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'Volume: <b>{zone["volume_m3"]} m³</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'Surfaces: <b>{zone["surface_count"]}</b><br>'
                    f'Confidence: <b>{zone["confidence"]}</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'Geometry source: <b>{zone["geometry_source"]}</b></div>', unsafe_allow_html=True,
                )
                with st.expander("Zone data as JSON"):
                    st.json(zone)
            elif S.get("idf_zone_info") is not None:
                st.caption("No IDF zone was generated for this space.")


def run_headless(ifc_path: str, idf_output: str, report_output: str) -> dict:
    """Batch conversion API used by validation corpora and CI pipelines."""
    source = Path(ifc_path)
    if not source.is_file():
        raise FileNotFoundError(f"IFC input not found: {source}")
    parsed = parse_ifc(source.read_bytes())
    generated_idf, generated_report, _, _, report = generate_baseline_idf_from_ifc(
        parsed["ifc"], parsed
    )
    idf_target = Path(idf_output)
    report_target = Path(report_output)
    idf_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated_idf, idf_target)
    shutil.copyfile(generated_report, report_target)
    return report


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive IFC-to-IDF converter")
    parser.add_argument("--ifc", help="Input IFC file; omit to start the Streamlit UI")
    parser.add_argument("--idf-output", default="baseline.idf", help="Output IDF path")
    parser.add_argument("--report-output", default="conversion_report.json", help="Output JSON report path")
    return parser


if __name__ == "__main__":
    args = _cli_parser().parse_args()
    if args.ifc:
        result = run_headless(args.ifc, args.idf_output, args.report_output)
        print(json.dumps({
            "zones_created": result["zones_created"],
            "quality_gate": result["validation"]["research_quality_gate"],
            "idf": str(Path(args.idf_output).resolve()),
            "report": str(Path(args.report_output).resolve()),
        }, indent=2))
    else:
        main()
