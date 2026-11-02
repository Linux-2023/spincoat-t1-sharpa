"""Conservative reduced film + finite delivery + ballistic parcel/catcher model.

Contact angle determines the spherical-cap source footprint, not a calibrated
dynamic wetting law. No resolved impact, breakup, air drag, evaporation or
fluid/robot reaction forces. Parcels represent outgoing volume, not a predicted
atomization spectrum. No task-tolerance threshold enters this model.
"""
from dataclasses import dataclass
import numpy as np
from .thin_film import ThinFilm, FilmParameters


@dataclass(frozen=True)
class FluidParameters(FilmParameters):
    contact_angle_deg: float = 10.
    delivery_duration_s: float = .08
    source_substep_s: float = .002
    parcel_volume_m3: float = .02e-9  # 20 nL representative parcels
    catcher_radius_m: float = .0572
    catcher_floor_z_m: float = -.032
    catcher_wall_top_z_m: float = .0065
    wall_drain_time_s: float = .3
    stability_roi_radius_fraction: float = .8
    stability_cv_limit: float = .05
    stability_shape_rate_limit_s: float = .05
    stability_hold_s: float = .5


class SpincoatFluid(ThinFilm):
    def __init__(self, parameters=None):
        super().__init__(parameters or FluidParameters())
        p = self.p
        if not 0 < p.contact_angle_deg < 60:
            raise ValueError("Reduced spherical-cap source supports contact angles 0..60 deg")
        if min(p.delivery_duration_s, p.source_substep_s, p.parcel_volume_m3, p.wall_drain_time_s) <= 0:
            raise ValueError("Delivery/parcel/drain parameters must be positive")
        self.doses = []
        self.rim_reservoir = np.zeros(16)
        self.direct_miss_m3 = 0.
        self.parcels = []
        self.phase_rad = 0.
        self._phase_start = 0.
        self._advance_start = 0.
        self.shape_rate_s = 0.
        self.stable_age_s = 0.
        self.first_uniform_time_s = None
        self.delivered_m3 = 0.
        self._last_shape = np.zeros_like(self.h)

    def _cap(self, xy, volume):
        if volume <= 0:
            return np.zeros_like(self.h)
        c = np.cos(np.deg2rad(self.p.contact_angle_deg))
        curvature_r = (3*volume/(np.pi*(2+c)*(1-c)**2))**(1/3)
        radial2 = (self.x-xy[0])**2+(self.y-xy[1])**2
        return np.maximum(np.sqrt(np.maximum(curvature_r**2-radial2, 0.))-curvature_r*c, 0.)

    def deposit(self, xy_m=(0., 0.), volume_m3=None):
        volume = self.p.volume_m3 if volume_m3 is None else float(volume_m3)
        if volume <= 0:
            raise ValueError("Dose volume must be positive")
        # Normalize on an extended virtual grid so off-substrate source volume
        # is lost honestly; never normalize an external dose back onto the disk.
        c = np.cos(np.deg2rad(self.p.contact_angle_deg))
        rc = (3*volume/(np.pi*(2+c)*(1-c)**2))**(1/3)
        base = rc*np.sin(np.deg2rad(self.p.contact_angle_deg))
        origin = -self.p.radius_m+self.dx/2
        axes = [origin+np.arange(int(np.floor((xy_m[j]-base-origin)/self.dx))-1,
                                int(np.ceil((xy_m[j]+base-origin)/self.dx))+2)*self.dx for j in (0, 1)]
        xx, yy = np.meshgrid(axes[0]-xy_m[0], axes[1]-xy_m[1])
        full = np.maximum(np.sqrt(np.maximum(rc**2-xx**2-yy**2, 0.))-rc*c, 0.)
        full_volume = float(full.sum()*self.dx**2)
        profile = self._cap(xy_m, volume)*(volume/max(full_volume, 1e-100))
        # Translation changes quadrature slightly; keep injected mass bounded.
        profile *= min(1., volume/max(float(profile.sum()*self.dx**2), 1e-100))
        kept = profile*self.mask
        kept_volume = float(kept.sum()*self.dx**2)
        missed = max(0., volume-kept_volume)
        self.initial_volume_m3 += volume
        self.doses.append({"start": self.time_s, "profile": kept, "volume": volume,
                           "missed": missed, "fraction": 0.})

    def _deliver(self, now):
        for dose in self.doses:
            u = np.clip((now-dose["start"])/self.p.delivery_duration_s, 0., 1.)
            fraction = u*u*(3-2*u)
            increment = fraction-dose["fraction"]
            if increment <= 0:
                continue
            self.h += increment*dose["profile"]
            self.delivered_m3 += increment*dose["volume"]
            self.direct_miss_m3 += increment*dose["missed"]
            self.escaped_volume_m3 += increment*dose["missed"]
            dose["fraction"] = fraction

    def _capture(self, position, velocity):
        p = self.p
        radial_speed2 = float(velocity[:2]@velocity[:2])
        b = 2*float(position[:2]@velocity[:2])
        c = float(position[:2]@position[:2])-p.catcher_radius_m**2
        wall_t = (-b+np.sqrt(max(b*b-4*radial_speed2*c, 0.)))/(2*radial_speed2) if radial_speed2 > 0 else np.inf
        floor_t = (velocity[2]+np.sqrt(velocity[2]**2+2*p.gravity_m_s2*(position[2]-p.catcher_floor_z_m)))/p.gravity_m_s2
        wall_z = position[2]+wall_t*velocity[2]-.5*p.gravity_m_s2*wall_t**2 if np.isfinite(wall_t) else -np.inf
        if wall_t <= floor_t and p.catcher_floor_z_m <= wall_z <= p.catcher_wall_top_z_m:
            duration, kind = wall_t, 1
        else:
            duration = floor_t
            at_floor = position[:2]+floor_t*velocity[:2]
            kind = 2 if np.linalg.norm(at_floor) <= p.catcher_radius_m else 3
        hit = position+duration*velocity
        hit[2] -= .5*p.gravity_m_s2*duration**2
        return float(duration), kind, hit

    def _on_rim_flux(self, volumes_m3, dt, omega_rad_s):
        self.rim_reservoir += volumes_m3
        angle = self._phase_start+omega_rad_s*(self.time_s-self._advance_start)
        for b in np.flatnonzero(self.rim_reservoir >= self.p.parcel_volume_m3):
            volume = np.floor(self.rim_reservoir[b]/self.p.parcel_volume_m3)*self.p.parcel_volume_m3
            self.rim_reservoir[b] -= volume
            local_angle = (b+.5)*2*np.pi/16
            theta = local_angle+angle
            radial = np.array([np.cos(theta), np.sin(theta), 0.])
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.])
            edge = self.mask & (self.rim_bin == b) & (self.x**2+self.y**2 > (self.p.radius_m-2*self.dx)**2)
            height = float(self.h[edge].mean()) if edge.any() else 0.
            radial_speed = volumes_m3[b]/max(dt*self.p.radius_m*(2*np.pi/16)*max(height, 1e-6), 1e-100)
            position = self.p.radius_m*radial
            velocity = radial_speed*radial+omega_rad_s*self.p.radius_m*tangent
            duration, kind, hit = self._capture(position, velocity)
            self.parcels.append({"birth_s": self.time_s, "volume_m3": float(volume),
                                 "position_m": position, "velocity_m_s": velocity,
                                 "hit_s": self.time_s+duration, "kind": kind, "hit_position_m": hit})

    def advance(self, duration_s, omega_rad_s, rotor_angle_start=None):
        if duration_s < 0:
            raise ValueError("Duration must be nonnegative")
        old_shape = self.h/max(float(self.h[self.mask].mean()), 1e-30)
        self._phase_start = self.phase_rad if rotor_angle_start is None else float(rotor_angle_start)
        self._advance_start = self.time_s
        end = self.time_s+duration_s
        while self.time_s < end-1e-12:
            pending = any(d["fraction"] < 1. for d in self.doses)
            step = min(end-self.time_s, self.p.source_substep_s if pending else .02)
            self._deliver(self.time_s+step/2)
            super().advance(step, omega_rad_s)
            self._deliver(self.time_s)
        self.phase_rad = self._phase_start+omega_rad_s*duration_s
        shape = self.h/max(float(self.h[self.mask].mean()), 1e-30)
        self.shape_rate_s = float(np.mean(abs(shape[self.mask]-old_shape[self.mask]))/max(duration_s,1e-30))
        roi = self.x**2+self.y**2 < (self.p.stability_roi_radius_fraction*self.p.radius_m)**2
        mean = float(self.h[roi].mean())
        cv = float(self.h[roi].std()/max(mean,1e-30))
        covered = float((self.h[self.mask]>1e-6).mean())
        stable = covered >= .95 and cv <= self.p.stability_cv_limit and self.shape_rate_s <= self.p.stability_shape_rate_limit_s and not any(d["fraction"] < 1. for d in self.doses)
        self.stable_age_s = self.stable_age_s+duration_s if stable else 0.
        if self.first_uniform_time_s is None and self.stable_age_s >= self.p.stability_hold_s:
            self.first_uniform_time_s = self.time_s-self.stable_age_s

    def compartments(self, at_time=None):
        now = self.time_s if at_time is None else at_time
        air, wall, floor, outside = 0., np.zeros(16), np.zeros(16), self.direct_miss_m3
        for parcel in self.parcels:
            if now < parcel["birth_s"]:
                continue
            v = parcel["volume_m3"]
            if now < parcel["hit_s"]:
                air += v
                continue
            angle = np.mod(np.arctan2(parcel["hit_position_m"][1], parcel["hit_position_m"][0]), 2*np.pi)
            b = min(int(angle/(2*np.pi)*16), 15)
            if parcel["kind"] == 1:
                remaining = v*np.exp(-(now-parcel["hit_s"])/self.p.wall_drain_time_s)
                wall[b] += remaining
                floor[b] += v-remaining
            elif parcel["kind"] == 2:
                floor[b] += v
            else:
                outside += v
        return {"airborne_m3": air, "wall_m3": wall, "floor_m3": floor,
                "uncontained_m3": outside, "rim_reservoir_m3": self.rim_reservoir.copy()}

    def metrics(self):
        m = super().metrics()
        c = self.compartments()
        film = float(self.h.sum()*self.dx**2)
        pending = max(0., self.initial_volume_m3-self.delivered_m3)
        total = film+pending+c["airborne_m3"]+c["wall_m3"].sum()+c["floor_m3"].sum()+c["uncontained_m3"]+c["rim_reservoir_m3"].sum()
        roi = self.x**2+self.y**2 < (self.p.stability_roi_radius_fraction*self.p.radius_m)**2
        roi_mean = float(self.h[roi].mean())
        m.update({"pending_ul": pending*1e9, "airborne_ul": c["airborne_m3"]*1e9,
                  "catcher_wall_ul": float(c["wall_m3"].sum())*1e9,
                  "catcher_floor_ul": float(c["floor_m3"].sum())*1e9,
                  "uncontained_ul": c["uncontained_m3"]*1e9,
                  "rim_reservoir_ul": float(c["rim_reservoir_m3"].sum())*1e9,
                  "mass_balance_relative_error": abs(total-self.initial_volume_m3)/max(self.initial_volume_m3,1e-30),
                  "film_roi_cv": float(self.h[roi].std()/max(roi_mean,1e-30)),
                  "film_shape_change_rate_s": self.shape_rate_s,
                  "first_uniform_time_s": self.first_uniform_time_s,
                  "parcel_count": len(self.parcels)})
        return m
