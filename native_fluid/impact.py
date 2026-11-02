"""Point-droplet ballistic proxy; SI units, no fluid spreading or air drag."""
import torch


def plane_impact(position, velocity, plane_z, gravity=9.81):
    """Return flight time, XY intersection, validity for a horizontal plane.

    The droplet inherits measured nozzle velocity. No additional ejection
    velocity is assumed. This is a declared approximation, not a fluid solver.
    """
    height = position[..., 2] - plane_z
    vz = velocity[..., 2]
    discriminant = vz.square() + 2 * gravity * height
    flight_time = (vz + torch.sqrt(discriminant.clamp_min(0))) / gravity
    valid = (height > 0) & torch.isfinite(position).all(-1)
    valid &= torch.isfinite(velocity).all(-1) & (discriminant >= 0)
    xy = position[..., :2] + velocity[..., :2] * flight_time[..., None]
    return flight_time, xy, valid
