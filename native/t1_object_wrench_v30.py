"""Bounded soft-wrench allocation with explicit unilateral friction constraints.
Geometry/encoder feedback only. No model edits or object-state writes.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import hand_indices

def skew(v):
 x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

class ObjectWrenchHold:
 def __init__(self,m,force=12.):
  self.force=float(force);self.pb=m.body('sharpa_left/left_hand_C_MC').id;self.tb=m.body('prep/reagent_tube').id;js,qa,va=hand_indices(m,'left');self.groups=[]
  for digit in ['thumb','index','middle']:
   ix=np.array([i for i,j in enumerate(js) if '/left_'+digit+'_' in m.joint(int(j)).name]);self.groups.append(dict(qa=qa[ix],va=va[ix],acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js[ix]]),geom=m.geom('sharpa_left/left_'+digit+'_elastomer').id))
  self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.pj=np.zeros((3,m.nv));self.vp=np.zeros(6);self.vt=np.zeros(6);self.start=None;self.last=None;self.released=False;self.max_clipping_rad=0.;self.allocated=None
 def apply(self,m,d,phase):
  if phase=='release_tube':self.released=True
  if self.released:return
  H=d.xmat[self.pb].reshape(3,3);origin=d.xpos[self.pb];R=d.xmat[self.tb].reshape(3,3);com=d.xipos[self.tb]
  if self.start is None:
   if not phase.startswith('pre_mix_clearance'):return
   self.start=float(d.time);self.ref=H.T@(com-origin);self.refR=H.T@R
   for g in self.groups:
    points=[c.pos.copy() for c in d.contact if g['geom'] in c.geom and self.tb in m.geom_bodyid[c.geom] and c.dist<0]
    if not points:raise RuntimeError('Wrench control requires actual three-pad contact')
    g['local']=d.geom_xmat[g['geom']].reshape(3,3).T@(np.mean(points,axis=0)-d.geom_xpos[g['geom']]);g['qref']=d.qpos[g['qa']].copy()
  target=origin+H@self.ref;mujoco.mj_jac(m,d,self.pj,self.jr,target,self.pb);targetv=self.pj@d.qvel;mujoco.mj_jacBodyCom(m,d,self.jp,self.jr,self.tb);velocity=self.jp@d.qvel
  mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.pb,self.vp,0);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.tb,self.vt,0)
  desired=np.r_[-float(m.body_mass[self.tb])*m.opt.gravity+1200.*(target-com)+12.*(targetv-velocity),1.5*Rotation.from_matrix(H@self.refR@R.T).as_rotvec()+.04*(self.vp[:3]-self.vt[:3])]
  points=[];normals=[];Jlist=[]
  for g in self.groups:
   point=d.geom_xpos[g['geom']]+d.geom_xmat[g['geom']].reshape(3,3)@g['local'];points.append(point);local=R.T@(point-d.xpos[self.tb]);n=-R@np.r_[local[:2],0.];n/=np.linalg.norm(n);normals.append(n)
   mujoco.mj_jac(m,d,self.jp,self.jr,point,int(m.geom_bodyid[g['geom']]));Jlist.append(self.jp[:,g['va']].copy())
  G=np.vstack([np.tile(np.eye(3),(1,3)),np.hstack([skew(point-com) for point in points])])
  if self.last is None or d.time-self.last>=.01-1e-9:
   C=[];lo=[];hi=[]
   for i,n in enumerate(normals):
    row=np.zeros(9);row[3*i:3*i+3]=n;C.append(row);lo.append(1.);hi.append(self.force)
    tangent=R[:,2];other=np.cross(n,tangent)
    for theta in np.arange(8)*np.pi/4:
     row=np.zeros(9);row[3*i:3*i+3]=.8*np.cos(np.pi/8)*n-np.cos(theta)*tangent-np.sin(theta)*other;C.append(row);lo.append(0.);hi.append(np.inf)
   seed=np.array(normals).ravel()*self.force*.7;scale=np.r_[np.ones(3),np.full(3,40.)];A=scale[:,None]*G;b=scale*desired
   fit=minimize(lambda f:float(np.sum((A@f-b)**2)+.002*np.sum((f-seed)**2)),seed if self.allocated is None else self.allocated.ravel(),jac=lambda f:2*A.T@(A@f-b)+.004*(f-seed),method='SLSQP',constraints=[LinearConstraint(np.array(C),lo,hi)],options=dict(maxiter=80,ftol=1e-8))
   if not fit.success:raise RuntimeError('Constrained soft-wrench allocation failed: '+fit.message)
   self.allocated=fit.x.reshape(3,3);self.last=float(d.time)
  u=np.clip((d.time-self.start)/.6,0,1);blend=u**3*(10+u*(-15+6*u))
  for g,J,F in zip(self.groups,Jlist,self.allocated):
   N=np.eye(len(g['va']))-np.linalg.pinv(J,rcond=1e-5)@J;tau=J.T@F-100.*(N@(d.qpos[g['qa']]-g['qref']))-1.*(N@d.qvel[g['va']])-d.qfrc_passive[g['va']];acts=g['acts'];bias=m.actuator_biasprm[acts];raw=(tau-bias[:,0]-bias[:,1]*d.qpos[g['qa']])/m.actuator_gainprm[acts,0];bounded=np.clip(raw,m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1]);self.max_clipping_rad=max(self.max_clipping_rad,float(np.max(abs(raw-bounded))));d.ctrl[acts]=(1-blend)*d.ctrl[acts]+blend*bounded
