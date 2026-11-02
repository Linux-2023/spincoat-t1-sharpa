"""Five-pad acquired posture with actuator-side gravity-load redistribution.
Preserves original model, caps and live object dynamics; no tactile force input.
"""
import mujoco
import numpy as np
from t1_spincoat_scene import hand_indices

class FivePadHold:
    def __init__(self,m,force=0.):
        if force!=0:raise ValueError('Five-pad pressure is defined by the frozen grasp plan')
        self.pb=m.body('sharpa_left/left_hand_C_MC').id;self.tb=m.body('prep/reagent_tube').id;js,qa,va=hand_indices(m,'left');self.acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js]);self.groups=[]
        for digit in ['thumb','index','middle','ring','pinky']:
            ix=np.array([i for i,j in enumerate(js) if '/left_'+digit+'_' in m.joint(int(j)).name]);self.groups.append(dict(acts=self.acts[ix],va=va[ix],geom=m.geom('sharpa_left/left_'+digit+'_elastomer').id))
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.start=None;self.released=False;self.max_clipping_rad=0.
    def apply(self,m,d,phase):
        if phase=='release_tube':self.released=True
        if self.released:return
        H=d.xmat[self.pb].reshape(3,3)
        gravity=-float(m.body_mass[self.tb])*m.opt.gravity/5.
        if self.start is None:
            if not phase.startswith('pre_mix_clearance'):return
            self.start=float(d.time);self.nominal=d.ctrl[self.acts].copy();self.gravity_local=H.T@gravity
            for g in self.groups:
                points=[c.pos.copy() for c in d.contact if g['geom'] in c.geom and self.tb in m.geom_bodyid[c.geom] and c.dist<0]
                if not points:raise RuntimeError('Five-pad holder requires all five actual pad contacts')
                g['local']=d.geom_xmat[g['geom']].reshape(3,3).T@(np.mean(points,axis=0)-d.geom_xpos[g['geom']])
        d.ctrl[self.acts]=self.nominal
        correction=gravity-H@self.gravity_local
        for g in self.groups:
            point=d.geom_xpos[g['geom']]+d.geom_xmat[g['geom']].reshape(3,3)@g['local'];mujoco.mj_jac(m,d,self.jp,self.jr,point,int(m.geom_bodyid[g['geom']]));acts=g['acts'];raw=d.ctrl[acts]+self.jp[:,g['va']].T@correction/m.actuator_gainprm[acts,0];bounded=np.clip(raw,m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1]);self.max_clipping_rad=max(self.max_clipping_rad,float(np.max(abs(raw-bounded))));d.ctrl[acts]=bounded
