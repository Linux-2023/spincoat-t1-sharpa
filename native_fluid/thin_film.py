"""Conservative 2-D lubrication approximation for a Newtonian spin-coated film.

SI units throughout. In a substrate-fixed frame:
  dh/dt + div(q) = 0
  q = h^3/(3*mu) * [rho*omega^2*(x,y) - grad(rho*g*h-gamma*laplacian(h))]

No Coriolis, evaporation, disjoining pressure, contact-angle hysteresis, or
resolved droplet-impact stage. Gaussian post-impact footprint is an assumption.
The 0.7 mm task tolerance is deliberately absent from this fluid solver.

Reference: https://arxiv.org/abs/2405.16983 (see source URL in metadata;
the implemented flat-substrate model is a reduced approximation, not the
full curved-substrate/Coriolis model of that paper).
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FilmParameters:
    radius_m: float = .018
    resolution: int = 64
    density_kg_m3: float = 1000.
    viscosity_pa_s: float = .01
    surface_tension_n_m: float = .03
    gravity_m_s2: float = 9.81
    volume_m3: float = 30e-9
    initial_sigma_m: float = .0035
    max_substep_s: float = .01


class ThinFilm:
    def __init__(self, parameters=None):
        self.p = parameters or FilmParameters()
        p = self.p
        if p.resolution < 8 or min(p.radius_m, p.viscosity_pa_s, p.initial_sigma_m, p.max_substep_s) <= 0:
            raise ValueError("Invalid film geometry, viscosity, footprint or step size")
        if min(p.density_kg_m3, p.surface_tension_n_m, p.gravity_m_s2) < 0:
            raise ValueError("Material parameters must be nonnegative")
        self.dx = 2*p.radius_m/p.resolution
        a = np.linspace(-p.radius_m+self.dx/2, p.radius_m-self.dx/2, p.resolution)
        self.x, self.y = np.meshgrid(a, a)
        self.mask = self.x**2+self.y**2 <= p.radius_m**2
        faces = np.linspace(-p.radius_m, p.radius_m, p.resolution+1)
        self.x_faces, self.y_faces = faces[None, :], faces[:, None]
        self.h = np.zeros_like(self.x)
        self.initial_volume_m3 = 0.
        self.escaped_volume_m3 = 0.
        self.time_s = 0.
        self.substeps = 0
        self.rim_outflow_bins_m3 = np.zeros(16)
        angle = np.mod(np.arctan2(self.y, self.x), 2*np.pi)
        self.rim_bin = np.floor(angle/(2*np.pi)*16).astype(int)
        self.boundary_x = np.pad(self.mask.astype(int), ((0,0),(1,0)))-np.pad(self.mask.astype(int), ((0,0),(0,1)))
        self.boundary_y = np.pad(self.mask.astype(int), ((1,0),(0,0)))-np.pad(self.mask.astype(int), ((0,1),(0,0)))
        self.face_bins_x = (np.mod(np.arctan2(a[:,None], faces[None,:]), 2*np.pi)/(2*np.pi)*16).astype(int)
        self.face_bins_y = (np.mod(np.arctan2(faces[:,None], a[None,:]), 2*np.pi)/(2*np.pi)*16).astype(int)

    def _on_rim_flux(self, volumes_m3, dt, omega_rad_s):
        """Optional coupling hook; called at the end of each conservative step."""
        pass

    def deposit(self, xy_m=(0., 0.), volume_m3=None):
        volume = self.p.volume_m3 if volume_m3 is None else volume_m3
        if volume <= 0:
            raise ValueError("Deposited volume must be positive")
        profile = np.exp(-((self.x-xy_m[0])**2+(self.y-xy_m[1])**2)/(2*self.p.initial_sigma_m**2))
        # A deposit far beyond the substrate is lost, not snapped to its edge.
        # Normalize over the infinite plane, not over this finite computational
        # square: otherwise an off-disk deposit would be pulled back onto it.
        profile *= volume/(2*np.pi*self.p.initial_sigma_m**2)
        kept = profile*self.mask
        kept *= min(1., volume/max(float(kept.sum()*self.dx**2), 1e-100))
        self.h += kept
        self.initial_volume_m3 += volume
        self.escaped_volume_m3 += volume-float(kept.sum()*self.dx**2)

    def _fluxes(self, omega_rad_s):
        p, h, dx = self.p, self.h, self.dx
        pad = np.pad(h, 1, mode="edge")
        lap = (pad[1:-1, 2:]+pad[1:-1, :-2]+pad[2:, 1:-1]+pad[:-2, 1:-1]-4*h)/dx**2
        pressure = p.density_kg_m3*p.gravity_m_s2*h - p.surface_tension_n_m*lap
        hl, hr = np.pad(h, ((0,0),(1,0))), np.pad(h, ((0,0),(0,1)))
        hb, ht = np.pad(h, ((1,0),(0,0))), np.pad(h, ((0,1),(0,0)))
        pl, pr = np.pad(pressure, ((0,0),(1,0))), np.pad(pressure, ((0,0),(0,1)))
        pb, pt = np.pad(pressure, ((1,0),(0,0))), np.pad(pressure, ((0,1),(0,0)))
        c = p.density_kg_m3*omega_rad_s**2/(3*p.viscosity_pa_s)
        fx = c*self.x_faces*np.where(self.x_faces >= 0, hl, hr)**3
        fy = c*self.y_faces*np.where(self.y_faces >= 0, hb, ht)**3
        fx -= ((hl+hr)/2)**3/(3*p.viscosity_pa_s)*(pr-pl)/dx
        fy -= ((hb+ht)/2)**3/(3*p.viscosity_pa_s)*(pt-pb)/dx
        return fx, fy

    def advance(self, duration_s, omega_rad_s):
        if duration_s < 0:
            raise ValueError("Duration must be nonnegative")
        remaining, p = float(duration_s), self.p
        while remaining > 1e-12:
            hmax = float(self.h.max())
            if hmax <= 1e-15:
                self.time_s += remaining
                break
            mobility = hmax**3/(3*p.viscosity_pa_s)
            adv_speed = p.density_kg_m3*omega_rad_s**2*p.radius_m*hmax**2/p.viscosity_pa_s
            dt = min(remaining, p.max_substep_s,
                     .20*self.dx/max(adv_speed, 1e-30),
                     .10*self.dx**2/max(p.density_kg_m3*p.gravity_m_s2*mobility, 1e-30),
                     .01*self.dx**4/max(p.surface_tension_n_m*mobility, 1e-30))
            fx, fy = self._fluxes(omega_rad_s)
            # Outgoing-volume limiter, applied once per shared face. It keeps
            # cells nonnegative without independently clipping their fluxes.
            outgoing = np.maximum(fx[:,1:],0)+np.maximum(-fx[:,:-1],0)
            outgoing += np.maximum(fy[1:,:],0)+np.maximum(-fy[:-1,:],0)
            scale = np.minimum(1., self.h*self.dx/(dt*outgoing+1e-100))
            left, right = np.pad(scale, ((0,0),(1,0))), np.pad(scale, ((0,0),(0,1)))
            bottom, top = np.pad(scale, ((1,0),(0,0))), np.pad(scale, ((0,1),(0,0)))
            fx *= np.where(fx >= 0, left, right)
            fy *= np.where(fy >= 0, bottom, top)
            old_volume = float(self.h.sum()*self.dx**2)
            updated = self.h - dt/self.dx*(fx[:,1:]-fx[:,:-1]+fy[1:,:]-fy[:-1,:])
            # Count every face crossing the disk boundary, including where the
            # disk touches the bounding square. Outside-cell counting missed
            # the latter, although the scalar escaped-volume ledger included it.
            vx = np.maximum(fx*self.boundary_x, 0.)*dt*self.dx
            vy = np.maximum(fy*self.boundary_y, 0.)*dt*self.dx
            rim = np.bincount(self.face_bins_x.ravel(), weights=vx.ravel(), minlength=16)
            rim += np.bincount(self.face_bins_y.ravel(), weights=vy.ravel(), minlength=16)
            self.rim_outflow_bins_m3 += rim
            self.h = np.maximum(updated, 0.)*self.mask
            new_volume = float(self.h.sum()*self.dx**2)
            self.escaped_volume_m3 += float(rim.sum())
            self.time_s += dt
            remaining -= dt
            self.substeps += 1
            self._on_rim_flux(rim, dt, omega_rad_s)

    def metrics(self):
        volume = float(self.h.sum()*self.dx**2)
        wet = self.mask & (self.h > 1e-6)
        return {"time_s": self.time_s, "on_substrate_ul": volume*1e9,
                "escaped_ul": self.escaped_volume_m3*1e9,
                "wet_area_fraction": float(wet.sum()/self.mask.sum()),
                "peak_thickness_um": float(self.h.max()*1e6),
                "mean_thickness_um": float(self.h[self.mask].mean()*1e6),
                "mass_balance_relative_error": abs(volume+self.escaped_volume_m3-self.initial_volume_m3)/max(self.initial_volume_m3,1e-30),
                "substeps": self.substeps}
