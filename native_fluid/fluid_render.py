"""Render mass-carrying parcels, wall films and floor pools from v2 replay data."""
import json
import numpy as np
import mujoco


def _ellipsoid(scene, position, size, matrix=None, alpha=1.):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.asarray(size), np.asarray(position),
                      (np.eye(3) if matrix is None else matrix).ravel(),
                      np.array([.04, .52, .95, alpha], dtype=np.float32))
    scene.ngeom += 1


def draw_fluid_v2(scene, archive, frame, target, blend=0.):
    if "fluid_parcel_birth_s" not in archive:
        return
    meta = json.loads(str(archive["metadata_json"]))
    p = meta["film_parameters"]
    next_frame = min(frame+1, len(archive["drop_film_time_s"])-1)
    t = (1-blend)*float(archive["drop_film_time_s"][frame])+blend*float(archive["drop_film_time_s"][next_frame])
    birth, hit = archive["fluid_parcel_birth_s"], archive["fluid_parcel_hit_s"]
    volume = archive["fluid_parcel_volume_m3"]
    position, velocity = archive["fluid_parcel_position_m"], archive["fluid_parcel_velocity_m_s"]
    g = p["gravity_m_s2"]
    for j in np.flatnonzero((birth <= t) & (t < hit)):
        age = t-birth[j]
        point = position[j]+age*velocity[j]
        point[2] -= .5*g*age*age
        radius = (3*volume[j]/(4*np.pi))**(1/3)
        _ellipsoid(scene, target+point, [radius]*3)
        # Explicit 1 ms exposure trail, rather than increasing parcel volume.
        previous_age = max(0., age-.001)
        previous = position[j]+previous_age*velocity[j]
        previous[2] -= .5*g*previous_age**2
        if np.linalg.norm(point-previous) > radius:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                              np.eye(3).ravel(), np.array([.05, .65, 1., .45], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius*.5, target+previous, target+point)
            scene.ngeom += 1
    landed = np.flatnonzero(hit <= t)
    wall, floor, weighted_z = np.zeros(16), np.zeros(16), np.zeros(16)
    for j in landed:
        pos = archive["fluid_parcel_hit_position_m"][j]
        b = min(int(np.mod(np.arctan2(pos[1],pos[0]), 2*np.pi)/(2*np.pi)*16),15)
        kind = int(archive["fluid_parcel_kind"][j])
        if kind == 1:
            remaining = volume[j]*np.exp(-(t-hit[j])/p["wall_drain_time_s"])
            wall[b] += remaining
            weighted_z[b] += remaining*pos[2]
            floor[b] += volume[j]-remaining
        elif kind == 2:
            floor[b] += volume[j]
    for b in range(16):
        angle = (b+.5)*2*np.pi/16
        radial, tangent = np.array([np.cos(angle),np.sin(angle),0.]), np.array([-np.sin(angle),np.cos(angle),0.])
        if wall[b] > 1e-13:
            upper = weighted_z[b]/wall[b]
            half_height = max(.001, .5*(upper-p["catcher_floor_z_m"]))
            half_width = .001
            thickness = 3*wall[b]/(4*np.pi*half_width*half_height)
            center = (p["catcher_radius_m"]-thickness)*radial
            center[2] = upper-half_height
            basis = np.column_stack((tangent, np.array([0.,0.,1.]), radial))
            _ellipsoid(scene, target+center, [half_width,half_height,thickness], basis)
        if floor[b] > 1e-13:
            radius = (4*floor[b]/(np.pi*np.tan(np.deg2rad(p["contact_angle_deg"]))))**(1/3)
            half_height = 3*floor[b]/(4*np.pi*radius**2)
            center = (p["catcher_radius_m"]-max(radius,.002))*radial
            center[2] = p["catcher_floor_z_m"]+half_height
            _ellipsoid(scene, target+center, [radius,radius,half_height])
