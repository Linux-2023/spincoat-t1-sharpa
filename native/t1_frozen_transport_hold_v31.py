"""V27 short shake preload followed by fixed acquired actuator commands.
Freezes position targets, not joints or object pose; original actuators and caps remain.
"""
import numpy as np
from t1_radial_preload_v30 import RadialPreload
from t1_spincoat_scene import hand_indices

class FrozenTransportHold(RadialPreload):
    def __init__(self,m,force):
        super().__init__(m,force)
        js,_,_=hand_indices(m,'left')
        self.acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js])
        self.acquired=None
    def apply(self,m,d,phase):
        if phase=='release_tube':self.released=True
        if self.released:return
        if self.acquired is not None:
            d.ctrl[self.acts]=self.acquired
            return
        super().apply(m,d,phase)
        if phase=='coarse_carry_to_receiver':self.acquired=d.ctrl[self.acts].copy()
