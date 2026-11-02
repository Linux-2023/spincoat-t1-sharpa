"""V34PourReference: receiver-relative planning references, unchanged native gates.

This plugin changes only planner input dictionaries and returned reference metadata.
The original V31 retargeter still performs every IK, wrist, joint-margin and native
collision test in the actual, unmodified world. Task corridors are intersected
with the original world corridors; their dimensions are never enlarged.
"""
from __future__ import annotations

import copy
from functools import wraps

import numpy as np

CANDIDATE_ID = "V34PourReference"
RECEIVER_BODY = "prep/shared_griffin_beaker"
# Frozen V31 preparation_config: receiver_xy_m and tabletop_z_m. The body origin
# is the exterior base, not the bottom-liquid plane or display-center site.
CANONICAL_ORIGIN_M = np.array([0.55, 0.20, 0.8000005])
REFERENCE_FIELDS = (
    "receiver_xy_m", "mouth_xy_m", "mouth_xy_offset_m",
    "mouth_xyz_waypoints_m", "mouth_height_waypoints_m",
    "v7_nominal_mouth_xyz_waypoints_m", "upright_retreat_mouth_xyz_m",
    "upright_retreat_mouth_z_m", "v7_preflow_mouth_box_m",
    "v7_preflow_front_x_limit_waypoints_m", "v7_preflow_zmin_waypoints_m",
)
PROVENANCE_KEY = "v34_pour_reference_provenance"
_INSTALLED = False


def receiver_references(model, data, profile):
    """Return a new profile; never write model, data, geometry, or the input."""
    result = copy.deepcopy(profile)
    body = model.body(RECEIVER_BODY).id
    origin = np.asarray(data.xpos[body], dtype=float).copy()
    rotation = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
    # SAME20 authorizes translations only. Do not approximate a rotated box by
    # an enlarged AABB or silently reinterpret world-aligned clearance planes.
    if not np.allclose(rotation, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("V34PourReference requires the canonical world-aligned receiver")
    if not np.all(np.isfinite(origin)):
        raise ValueError("Nonfinite actual receiver frame")
    previous = result.get(PROVENANCE_KEY)
    source = (copy.deepcopy(previous["canonical_source_fields"]) if previous
              else {key: copy.deepcopy(profile[key]) for key in REFERENCE_FIELDS
                    if key in profile})
    delta = origin - CANONICAL_ORIGIN_M
    changes = []

    def record(key, value, rule):
        result[key] = value
        changes.append({"field": key, "rule": rule})

    for key in ("mouth_xyz_waypoints_m", "v7_nominal_mouth_xyz_waypoints_m"):
        if key in source:
            rows = np.asarray(source[key], dtype=float).copy()
            rows[:, 1:4] += delta
            record(key, rows.tolist(), "angle unchanged; xyz_actual = xyz_canonical + receiver_delta")
    if "mouth_xy_m" in source:
        record("mouth_xy_m", (np.asarray(source["mouth_xy_m"]) + delta[:2]).tolist(),
               "xy_actual = xy_canonical + receiver_delta_xy")
    if "mouth_height_waypoints_m" in source:
        rows = np.asarray(source["mouth_height_waypoints_m"], dtype=float).copy()
        rows[:, 1] += delta[2]
        record("mouth_height_waypoints_m", rows.tolist(), "angle unchanged; z += receiver_delta_z")
    if "upright_retreat_mouth_xyz_m" in source:
        record("upright_retreat_mouth_xyz_m",
               (np.asarray(source["upright_retreat_mouth_xyz_m"]) + delta).tolist(),
               "xyz_actual = xyz_canonical + receiver_delta")
    if "upright_retreat_mouth_z_m" in source:
        record("upright_retreat_mouth_z_m", float(source["upright_retreat_mouth_z_m"] + delta[2]),
               "z_actual = z_canonical + receiver_delta_z")
    record("receiver_xy_m", origin[:2].tolist(), "actual data.xpos of shared Griffin beaker")
    if "mouth_xy_m" in result:
        record("mouth_xy_offset_m", (np.asarray(result["mouth_xy_m"]) - origin[:2]).tolist(),
               "receiver-local vector, not a point; no extra translation")

    corridor_audit = {}
    if "v7_preflow_mouth_box_m" in source:
        world = np.asarray(source["v7_preflow_mouth_box_m"], dtype=float)
        task = world + delta[:, None]
        effective = np.column_stack((np.maximum(world[:, 0], task[:, 0]),
                                     np.minimum(world[:, 1], task[:, 1])))
        if np.any(effective[:, 0] > effective[:, 1]):
            raise ValueError("Receiver-relative and original world corridors do not intersect")
        record("v7_preflow_mouth_box_m", effective.tolist(),
               "intersection(original_world_box, receiver_relative_task_box); never union")
        corridor_audit["mouth_box"] = {"world": world.tolist(), "task": task.tolist(),
                                       "effective": effective.tolist()}
    for key, axis, lower in (("v7_preflow_front_x_limit_waypoints_m", 0, False),
                             ("v7_preflow_zmin_waypoints_m", 2, True)):
        if key in source:
            world = np.asarray(source[key], dtype=float)
            task = world.copy()
            task[:, 1] += delta[axis]
            effective = world.copy()
            effective[:, 1] = (np.maximum(world[:, 1], task[:, 1]) if lower
                                else np.minimum(world[:, 1], task[:, 1]))
            record(key, effective.tolist(),
                   "max(original_world_floor, task_floor)" if lower
                   else "min(original_world_front_limit, task_front_limit)")
            corridor_audit[key] = {"world": world.tolist(), "task": task.tolist(),
                                    "effective": effective.tolist()}

    result[PROVENANCE_KEY] = {
        "candidate_id": CANDIDATE_ID,
        "frame": "world-aligned frame at actual receiver exterior-base body origin",
        "receiver_body": RECEIVER_BODY,
        "canonical_origin_world_m": CANONICAL_ORIGIN_M.tolist(),
        "actual_origin_world_m": origin.tolist(),
        "receiver_rotation_world": rotation.tolist(),
        "translation_world_m": delta.tolist(),
        "read_time_s": float(data.time),
        "canonical_source_fields": source,
        "field_transformations": changes,
        "corridor_intersections": corridor_audit,
        "unchanged": ["all world obstacles and model parameters", "tilt/yaw and joint seeds",
                      "original pose/wrist/native collision guards", "timing and liquid ledger",
                      "maximum_drain_wait_s", "original world corridor bounds"],
        "scope": "privileged actual receiver geometry; no perception accuracy claim",
    }
    return result


def synchronize_planned_references(profile, report):
    """Rebuild duplicate FK output views, without changing any planned command."""
    curve = np.asarray(profile["mouth_xyz_waypoints_m"], dtype=float)
    origin = np.asarray(profile[PROVENANCE_KEY]["actual_origin_world_m"])
    profile["mouth_height_waypoints_m"] = curve[:, [0, 3]].tolist()
    profile["mouth_xy_m"] = curve[-1, 1:3].tolist()
    profile["mouth_xy_offset_m"] = (curve[-1, 1:3] - origin[:2]).tolist()
    profile["upright_retreat_mouth_xyz_m"] = curve[0, 1:4].tolist()
    profile["upright_retreat_mouth_z_m"] = float(curve[0, 3])
    provenance = profile[PROVENANCE_KEY]
    provenance["planned_output_views"] = {
        "mouth_height_waypoints_m": "angle,z columns of actual replanned mouth_xyz_waypoints_m",
        "mouth_xy_m": "final actual replanned mouth point XY",
        "mouth_xy_offset_m": "final actual replanned mouth XY minus actual receiver XY",
        "upright_retreat_mouth_xyz_m": "first actual replanned upright mouth point",
        "upright_retreat_mouth_z_m": "first actual replanned upright mouth point Z",
        "v7_nominal_mouth_xyz_waypoints_m": "receiver-relative engineering goals, not overwritten by FK",
    }
    report[PROVENANCE_KEY] = copy.deepcopy(provenance)
    return profile, report


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_preturn_route_v31 as original
    previous = original.retarget_from_live

    @wraps(previous)
    def retarget_from_live(model, data, profile):
        rebased = receiver_references(model, data, profile)
        planned, report = previous(model, data, rebased)
        return synchronize_planned_references(planned, report)

    original.retarget_from_live = retarget_from_live
    _INSTALLED = True
