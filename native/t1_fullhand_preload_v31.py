"""Preserve a fitted full-hand clearance posture while applying radial preload.
Idle fingers stay at their physically acquired targets until deliberate release.
"""
import numpy as np
from t1_radial_preload_v30 import RadialPreload
from t1_spincoat_scene import hand_indices

class FullHandPreload(RadialPreload):
    def __init__(self,m,extra_force_N=3.):
        super().__init__(m,extra_force_N)
        joints=hand_indices(m,'left')[0][13:]
        self.idle_acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints])
        self.idle_command=None
    def apply(self,m,d,phase):
        if self.idle_command is None and phase.startswith('pre_mix_clearance'):
            self.idle_command=d.ctrl[self.idle_acts].copy()
        super().apply(m,d,phase)
        if self.idle_command is not None and not self.released:
            d.ctrl[self.idle_acts]=self.idle_command
