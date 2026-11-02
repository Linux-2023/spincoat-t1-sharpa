"""Independent v4 natural-q reference motion.
Control writes occur only in the caller. Inline q nodes define the geometric
path; FK-derived source pose replaces old Cartesian goals. The coordinate
0..100 is a nominal path label, not actual tube tilt. No gain/physics changes.
"""
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator,CubicSpline
from t1_spincoat_scene import arm_indices

def entry_target(m,d,profile):
 """Read-only work-FK target for root coarse carry from its own current grasp."""
 _,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w);pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body(profile.get('tube_body','prep/reagent_tube')).id;ms=m.site(profile.get('tube_mouth_site','prep/tube_mouth')).id;H=w.xmat[pb].reshape(3,3).copy();E=H.T@w.xmat[tb].reshape(3,3);ml=H.T@(w.site_xpos[ms]-w.xpos[pb]);q=np.asarray(profile['joint_reference_keyframes_rad'][0][1:]);w.qpos[qa]=q;mujoco.mj_kinematics(m,w);H=w.xmat[pb].reshape(3,3).copy();P=w.xpos[pb].copy()
 return dict(mouth_world_m=P+H@ml,hand_rotation_world=H,tube_rotation_world=H@E,arm_command=q.copy(),measured_hand_to_tube_R=E,measured_mouth_in_hand_m=ml)

def plan_segment(m,d,profile,*,segment,entry_arm_command,duration_s):
 from t1_human_pour_profile_v4 import smoothstep5
 if profile.get('expected_torso_q_rad') is not None:
  tq=np.array([d.qpos[m.joint('astribot_torso_joint_'+str(i)).qposadr[0]] for i in range(1,5)])
  if np.max(abs(tq-np.asarray(profile['expected_torso_q_rad'])))>.005:raise RuntimeError('Natural joint-path requires qualified declared torso pose')
 ids,qa,_=arm_indices(m,'left');seed=np.asarray(entry_arm_command).copy();nodes=np.asarray(profile['joint_reference_keyframes_rad'],dtype=float);reference=PchipInterpolator(nodes[:,0],nodes[:,1:],axis=0);maximum=float(profile.get('max_tilt_deg',100.))
 if segment=='tilt':
  coord=np.linspace(0,maximum,2001);commands=reference(coord)
  if np.max(abs(commands[0]-seed))>1e-6:raise RuntimeError('Joint-reference tilt requires qualified approach ending at first q node')
  profile['_v4_joint_cache']=dict(coordinate=coord.copy(),commands=commands.copy())
 elif segment=='return':
  coord=np.linspace(maximum,0,2001);commands=reference(coord)
  if np.max(abs(commands[0]-seed))>1e-6:raise RuntimeError('Reverse joint-reference entry discontinuity')
 elif segment=='approach':
  coord=np.linspace(0,1,201);commands=seed[None,:]+coord[:,None]*(nodes[0,1:]-seed)[None,:]
 elif segment=='retreat':
  coord=np.array([0.,1.]);commands=np.tile(seed,(2,1))
 else:raise ValueError(segment)
 # Explicit ergonomic planning guard, independent of the unchanged physical gates.
 w=mujoco.MjData(m);w.qpos[:]=d.qpos;el=m.body('astribot_arm_left_link_5').id;wr=m.body('astribot_arm_left_link_6').id;pb=m.body('sharpa_left/left_hand_C_MC').id;mc=m.body('sharpa_left/left_middle_MCP_VL').id;wrist=[]
 for qq in commands:
  w.qpos[qa]=qq;mujoco.mj_kinematics(m,w);u=w.xpos[wr]-w.xpos[el];v=w.xpos[mc]-w.xpos[pb];wrist.append(float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))))
 if max(wrist)>50.+1e-6:raise RuntimeError('Natural joint-reference wrist exceeds declared50degree geometric bound')
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(commands,axis=0),axis=1))];times=np.arange(1,int(round(duration_s/m.opt.timestep))+1)*m.opt.timestep;keep=np.r_[True,np.diff(arc)>1e-12]
 if arc[-1]<1e-10:samplecoord=np.full(len(times),coord[0]);q=np.tile(seed,(len(times),1))
 else:
  samplecoord=PchipInterpolator(arc[keep],coord[keep])(arc[-1]*smoothstep5(times/duration_s))
  if segment in ['tilt','return']:q=reference(samplecoord)
  else:q=seed[None,:]+samplecoord[:,None]*(nodes[0,1:]-seed)[None,:]
 spline=CubicSpline(np.r_[0,times],np.vstack([seed,q]),axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))));peak=float(abs(spline(times,1)).max());margin=float(np.minimum(q-m.jnt_range[ids,0],m.jnt_range[ids,1]-q).min())
 if peak>2. or margin<.05-1e-6:raise RuntimeError('Natural joint-reference motion rejected '+str((peak,margin)))
 measured=entry_target(m,d,profile)
 return dict(time=times,arm_commands=q,angle_deg=samplecoord if segment in ['tilt','return'] else None,segment=segment,duration_s=float(duration_s),peak_joint_speed_rad_s=peak,min_joint_margin_rad=margin,joint_path_length_rad=float(arc[-1]),geometric_knot_count=len(coord),time_law='joint_reference_arclength_C2',coordinate_semantics='nominal path coordinate, not actual tube tilt',planned_forearm_palm_max_deg=float(max(wrist)),measured_hand_to_tube_R=np.asarray(profile.get('measured_hand_to_tube_R',measured['measured_hand_to_tube_R'])).tolist(),measured_mouth_in_hand_m=np.asarray(profile.get('measured_mouth_in_hand_m',measured['measured_mouth_in_hand_m'])).tolist())

