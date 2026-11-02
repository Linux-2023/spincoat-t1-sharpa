"""V4 fixed geometric path plus joint-arclength C2 timing.
Pure workData IK; no live steps/state writes, no contact input, physical edits,
or gate changes. Same .1mm/.1deg pose, .05rad margin and 2rad/s guards.
"""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from scipy.interpolate import CubicSpline,PchipInterpolator
from t1_spincoat_scene import arm_indices
from t1_human_pour_v4_geometric_ik import solve_geometric

def plan_segment(m,d,profile,*,segment,entry_arm_command,duration_s):
 from t1_human_pour_profile_v4 import mouth_position,axial_yaw_deg,smoothstep5
 _,qa,_=arm_indices(m,'left');ids,_,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body(profile.get('tube_body','prep/reagent_tube')).id;ms=m.site(profile.get('tube_mouth_site','prep/tube_mouth')).id;mujoco.mj_kinematics(m,w);Ha=w.xmat[pb].reshape(3,3).copy();Pa=w.xpos[pb].copy();E=np.asarray(profile.get('measured_hand_to_tube_R',Ha.T@w.xmat[tb].reshape(3,3)));ml=np.asarray(profile.get('measured_mouth_in_hand_m',Ha.T@(w.site_xpos[ms]-Pa)))
 seed=np.asarray(entry_arm_command).copy();w.qpos[qa]=seed;mujoco.mj_kinematics(m,w);H0=w.xmat[pb].reshape(3,3).copy();P0=w.xpos[pb].copy();M0=P0+H0@ml;maximum=float(profile.get('max_tilt_deg',100));az=np.deg2rad(profile['tilt_axis_azimuth_deg']);axis=np.array([np.cos(az),np.sin(az),0.]);info=[]
 if segment=='return':
  cache=profile.get('_v4_tilt_cache')
  if cache is None:raise ValueError('V4 return requires same-sequence tilt geometry cache')
  coord=np.linspace(maximum,0,max(2,int(np.ceil(maximum/.2))+1));curve=CubicSpline(cache['angle_deg'],cache['arm_commands'],axis=0);commands=curve(coord)
  if np.max(abs(commands[0]-seed))>1e-6:raise RuntimeError('Reverse-path entry command discontinuity')
 else:
  coord=np.linspace(0,maximum,max(2,int(np.ceil(maximum/.2))+1)) if segment=='tilt' else np.linspace(0,1,201)
  commands=[seed.copy()]
  if segment in ['approach','retreat']:
   goalH=Rotation.from_euler('z',axial_yaw_deg(0,profile),degrees=True).as_matrix()@E.T;goalM=mouth_position(0,profile) if segment=='approach' else np.asarray(profile.get('upright_retreat_mouth_xyz_m',mouth_position(0,profile)));slerp=Slerp([0,1],Rotation.from_matrix(np.array([H0,goalH])))
  for u in coord[1:]:
   if segment=='tilt':
    H=Rotation.from_rotvec(np.deg2rad(u)*axis).as_matrix()@Rotation.from_euler('z',axial_yaw_deg(u,profile),degrees=True).as_matrix()@E.T;M=mouth_position(u,profile)
   else:H=slerp(u).as_matrix();M=M0+(goalM-M0)*u
   P=M-H@ml
   if segment!='tilt' and np.linalg.norm(goalM-M0)<1e-9 and np.linalg.norm(Rotation.from_matrix(goalH@H0.T).as_rotvec())<1e-9:q=seed.copy();ii={}
   else:q,ii=solve_geometric(m,w,P,H,seed=seed)
   seed=q;commands.append(q.copy());info.append(ii);w.qpos[qa]=q;mujoco.mj_kinematics(m,w)
  commands=np.asarray(commands)
  if segment=='tilt':profile['_v4_tilt_cache']=dict(angle_deg=coord.copy(),arm_commands=commands.copy())
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(commands,axis=0),axis=1))];times=np.arange(1,int(round(duration_s/m.opt.timestep))+1)*m.opt.timestep;g=smoothstep5(times/duration_s)
 keep=np.r_[True,np.diff(arc)>1e-12]
 if arc[-1]<1e-10:samplecoord=np.full(len(times),coord[0]);denseq=np.tile(commands[0],(len(times),1))
 else:
  samplecoord=PchipInterpolator(arc[keep],coord[keep])(arc[-1]*g);denseq=CubicSpline(coord if segment!='return' else coord[::-1],commands if segment!='return' else commands[::-1],axis=0)(samplecoord)
 spline=CubicSpline(np.r_[0,times],np.vstack([commands[0],denseq]),axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))));peak=float(abs(spline(times,1)).max());margin=float(np.minimum(denseq-m.jnt_range[ids,0],m.jnt_range[ids,1]-denseq).min())
 if peak>2 or margin<.05-1e-6:raise RuntimeError('V4 arclength spline rejected '+str((peak,margin)))
 return dict(time=times,arm_commands=denseq,angle_deg=samplecoord if segment in ['tilt','return'] else None,segment=segment,duration_s=float(duration_s),peak_joint_speed_rad_s=peak,min_joint_margin_rad=margin,joint_path_length_rad=float(arc[-1]),geometric_knot_count=len(coord),time_law='joint_arclength_C2',measured_hand_to_tube_R=E.tolist(),measured_mouth_in_hand_m=ml.tolist())

