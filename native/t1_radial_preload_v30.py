"""Bounded radial grasp preload using actual initial pad geometry.
Adds position-actuator torque demand; never changes model gains, force limits,
friction or object pose. Simulator geometry is not deployed tactile sensing.
"""
import mujoco
import numpy as np
from t1_spincoat_scene import hand_indices

class RadialPreload:
    def __init__(self,m,extra_force_N):
        self.force=float(extra_force_N)
        if not 0<=self.force<=12:raise ValueError('Preload outside diagnostic range')
        self.pb=m.body('sharpa_left/left_hand_C_MC').id
        self.tb=m.body('prep/reagent_tube').id
        js,qa,va=hand_indices(m,'left');self.groups=[]
        for digit in ['thumb','index','middle']:
            ix=np.array([i for i,j in enumerate(js) if '/left_'+digit+'_' in m.joint(int(j)).name])
            acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js[ix]])
            self.groups.append(dict(va=va[ix],acts=acts,geom=m.geom('sharpa_left/left_'+digit+'_elastomer').id))
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.start=None
        self.max_clipping_rad=0.;self.released=False
    def apply(self,m,d,phase):
        if phase=='release_tube':self.released=True
        if self.released:return
        if self.start is None:
            if not phase.startswith('pre_mix_clearance'):return
            self.start=float(d.time);H=d.xmat[self.pb].reshape(3,3);R=d.xmat[self.tb].reshape(3,3)
            for g in self.groups:
                points=[c.pos.copy() for c in d.contact if g['geom'] in c.geom and self.tb in m.geom_bodyid[c.geom] and c.dist<0]
                if not points:raise RuntimeError('Preload requires actual three-pad contact')
                point=np.mean(points,axis=0);local=R.T@(point-d.xpos[self.tb]);radial=R@np.r_[local[:2],0.];radial/=np.linalg.norm(radial)
                g['normal_palm']=H.T@(-radial);g['point_local']=d.geom_xmat[g['geom']].reshape(3,3).T@(point-d.geom_xpos[g['geom']])
        if any(phase.startswith(s) for s in ['release_tube','withdraw_empty','return_to_original','settle_original','empty_hand']):return
        u=np.clip((d.time-self.start)/.6,0,1);blend=u**3*(10+u*(-15+6*u));H=d.xmat[self.pb].reshape(3,3)
        for g in self.groups:
            point=d.geom_xpos[g['geom']]+d.geom_xmat[g['geom']].reshape(3,3)@g['point_local']
            mujoco.mj_jac(m,d,self.jp,self.jr,point,int(m.geom_bodyid[g['geom']]))
            torque=self.jp[:,g['va']].T@(H@g['normal_palm']*self.force*blend)
            acts=g['acts'];target=d.ctrl[acts]+torque/m.actuator_gainprm[acts,0]
            limited=np.clip(target,m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1]);self.max_clipping_rad=max(self.max_clipping_rad,float(np.max(abs(target-limited))));d.ctrl[acts]=limited
