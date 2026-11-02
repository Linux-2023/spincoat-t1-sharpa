"""Explicit geometric pour targets; no fluid acceptance rules are changed."""
import numpy as np
from scipy.spatial.transform import Rotation


def resolve_pour_profile(plan, config):
    raw = plan.get('pour_trajectory')
    if raw is None:
        return dict(explicit=False, transport_yaw_deg=float(getattr(config, 'transport_yaw_deg', 0.)),
                    tilt_axis_azimuth_deg=0., max_tilt_deg=100, tilt_waypoints_deg=[25, 50, 75, 100], mouth_heights_m=None,
                    return_tilt_waypoints_deg=[75, 50, 25, 0], return_mouth_heights_m=None)
    yaw = float(raw['transport_yaw_deg'])
    axis_azimuth = float(raw.get('tilt_axis_azimuth_deg', 0.))
    if not np.isfinite(axis_azimuth):
        raise ValueError('Horizontal tilt-axis azimuth must be finite')
    maximum = float(raw['max_tilt_deg'])
    knots = np.asarray(raw['mouth_height_waypoints_m'], dtype=float)
    if not np.isfinite(yaw) or not np.isfinite(maximum) or not 80 <= maximum <= 110:
        raise ValueError('Explicit pour requires finite yaw and maximum tilt in80..110degrees')
    if (knots.ndim != 2 or knots.shape[1] != 2 or len(knots) < 2 or
            not np.all(np.isfinite(knots)) or np.any(np.diff(knots[:, 0]) <= 0) or
            abs(knots[0, 0]) > 1e-9 or abs(knots[-1, 0]-maximum) > 1e-9):
        raise ValueError('Pour height knots must span0..maximum tilt in increasing angle order')
    returning = raw.get('return_mouth_height_waypoints_m')
    if returning is not None:
        returning = np.asarray(returning, dtype=float)
        if (returning.ndim != 2 or returning.shape[1] != 2 or len(returning) < 2 or
                not np.all(np.isfinite(returning)) or np.any(np.diff(returning[:, 0]) >= 0) or
                abs(returning[0, 0]-maximum) > 1e-9 or abs(returning[-1, 0]) > 1e-9):
            raise ValueError('Return height knots must span maximum tilt..0 in decreasing angle order')
    return dict(explicit=True, transport_yaw_deg=yaw, tilt_axis_azimuth_deg=axis_azimuth, max_tilt_deg=maximum,
                tilt_waypoints_deg=knots[1:, 0].tolist(), mouth_heights_m=knots.tolist(),
                return_tilt_waypoints_deg=returning[1:, 0].tolist() if returning is not None else [75, 50, 25, 0],
                return_mouth_heights_m=returning.tolist() if returning is not None else None,
                scope='Geometric target schedule only; original fluid capture/tilt/volume gates retained')


def pour_endpoints(prepour, pourmouth, profile):
    first, last = np.array(prepour, copy=True), np.array(pourmouth, copy=True)
    if profile['mouth_heights_m'] is not None:
        first[2], last[2] = profile['mouth_heights_m'][0][1], profile['mouth_heights_m'][-1][1]
    return first, last


def pour_mouth_target(angle_deg, prepour, pourmouth, profile):
    target = prepour+(pourmouth-prepour)*angle_deg/profile['max_tilt_deg']
    if profile['mouth_heights_m'] is not None:
        knots = np.asarray(profile['mouth_heights_m'])
        target[2] = np.interp(angle_deg, knots[:, 0], knots[:, 1])
    return target


def post_lift_settle_seconds(plan):
    value = float(plan.get('post_lift_settle_s', 0.))
    if not np.isfinite(value) or not 0 <= value <= 2.:
        raise ValueError('Post-lift settling must be within0..2seconds')
    return value


def return_mouth_target(angle_deg, prepour, profile):
    target = np.array(prepour, copy=True)
    if profile['return_mouth_heights_m'] is not None:
        knots = np.asarray(profile['return_mouth_heights_m'])[::-1]
        target[2] = np.interp(angle_deg, knots[:, 0], knots[:, 1])
    return target


def pour_rotation(angle_deg, profile):
    """World-horizontal mouth-pivot rotation; zero azimuth preserves original Rx exactly."""
    azimuth = float(profile.get('tilt_axis_azimuth_deg', 0.))
    angle = float(angle_deg)
    if not np.isfinite(angle) or not np.isfinite(azimuth):
        raise ValueError('Pour rotation requires finite angle and azimuth')
    if azimuth == 0.:
        return Rotation.from_euler('x', angle, degrees=True).as_matrix()
    radians = np.deg2rad(azimuth)
    axis = np.array([np.cos(radians), np.sin(radians), 0.])
    return Rotation.from_rotvec(np.deg2rad(angle)*axis).as_matrix()
