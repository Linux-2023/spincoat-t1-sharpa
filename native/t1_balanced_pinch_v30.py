"""Three-pad Cartesian force/posture control under unchanged actuator limits.
Uses simulator geometry and encoders, not tactile force measurements.
"""
import mujoco
import numpy as np
from t1_spincoat_scene import hand_indices

class BalancedPinch:
    def __init__(self,m,force=9.):
        self.force=max(4.,float(force));self.pb=m.body('sharpa_left/left_hand_C_MC').id;self.tb=m.body('prep/reagent_tube').id
        js,qa,va=hand_indices(m,'left');self.groups=[]
        for digit in ['thumb','index','middle']:
            ix=np.array([i for i,j in enumerate(js) if '/left_'+digit+'_' in m.joint(int(j)).name]);self.groups.append(dict(qa=qa[ix],va=va[ix],acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js[ix]]),geom=m.geom('sharpa_left/left_'+digit+'_elastomer').id))
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.pj=np.zeros((3,m.nv));self.start=None;self.released=False;self.max_clipping_rad=0.
    def apply(self,m,d,phase):
        if phase=='release_tube':self.released=True
        if self.released:return
        H=d.xmat[self.pb].reshape(3,3);origin=d.xpos[self.pb]
        if self.start is None:
            if not phase.startswith('pre_mix_clearance'):return
            self.start=float(d.time);R=d.xmat[self.tb].reshape(3,3);normals=[]
            for g in self.groups:
                points=[c.pos.copy() for c in d.contact if g['geom'] in c.geom and self.tb in m.geom_bodyid[c.geom] and c.dist<0]
                if not points:raise RuntimeError('Pinch requires actual three-pad contact')
                point=np.mean(points,axis=0);local=R.T@(point-d.xpos[self.tb]);radial=R@np.r_[local[:2],0.];radial/=np.linalg.norm(radial);n=H.T@(-radial);normals.append(n)
                g.update(normal=n,point_local=d.geom_xmat[g['geom']].reshape(3,3).T@(point-d.geom_xpos[g['geom']]),position=H.T@(point-origin),qref=d.qpos[g['qa']].copy())
            _,_,vh=np.linalg.svd(np.array(normals).T);weights=vh[-1];weights*=np.sign(weights[0])
            if np.any(weights<=0):raise RuntimeError('No positive three-normal force closure')
            self.weights=weights/weights.max()
        u=np.clip((d.time-self.start)/.6,0,1);blend=u**3*(10+u*(-15+6*u))
        for g,weight in zip(self.groups,self.weights):
            point=d.geom_xpos[g['geom']]+d.geom_xmat[g['geom']].reshape(3,3)@g['point_local'];target=origin+H@g['position'];normal=H@g['normal'];tangent=np.eye(3)-np.outer(normal,normal)
            mujoco.mj_jac(m,d,self.jp,self.jr,point,int(m.geom_bodyid[g['geom']]));J=self.jp[:,g['va']].copy();velocity=self.jp@d.qvel
            mujoco.mj_jac(m,d,self.pj,self.jr,target,self.pb);target_velocity=self.pj@d.qvel
            F=normal*self.force*weight+tangent@(1200.*(target-point)+8.*(target_velocity-velocity))
            N=np.eye(len(g['va']))-np.linalg.pinv(J,rcond=1e-5)@J
            tau=J.T@F-10.*(N@(d.qpos[g['qa']]-g['qref']))-.3*(N@d.qvel[g['va']])-d.qfrc_passive[g['va']]
            acts=g['acts'];bias=m.actuator_biasprm[acts];gain=m.actuator_gainprm[acts,0];raw=(tau-bias[:,0]-bias[:,1]*d.qpos[g['qa']])/gain
            bounded=np.clip(raw,m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1]);self.max_clipping_rad=max(self.max_clipping_rad,float(np.max(abs(raw-bounded))));d.ctrl[acts]=(1-blend)*d.ctrl[acts]+blend*bounded
