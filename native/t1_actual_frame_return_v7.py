"""Single privileged actual-E read at return entry; no live pose/state writes."""
import numpy as np,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator,CubicSpline
from t1_spincoat_scene import arm_indices

def plan_return(m,d,profile,*,entry_arm_command,duration_s=10.):
 ids,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w)
 pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body('prep/reagent_tube').id;ms=m.site('prep/tube_mouth').id
 H=w.xmat[pb].reshape(3,3).copy();T=w.xmat[tb].reshape(3,3).copy();E=H.T@T;ml=H.T@(w.site_xpos[ms]-w.xpos[pb]);M0=w.site_xpos[ms].copy();rel=H.T@(w.xpos[tb]-w.xpos[pb]);q0=w.qpos[qa].copy();offset=np.asarray(entry_arm_command)-q0
 theta=float(np.arccos(np.clip(T[2,2],-1,1)));axis=np.cross([0.,0.,1.],T[:,2]);axis/=max(np.linalg.norm(axis),1e-12);gR=Rotation.from_rotvec(-theta*axis).as_matrix()@T;gamma=float(np.arctan2(gR[1,0],gR[0,0]));v=np.r_[q0,gamma];el=m.body('astribot_arm_left_link_5').id;wr=m.body('astribot_arm_left_link_6').id;mc=m.body('sharpa_left/left_middle_MCP_VL').id
 lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;rows=[];qs=[];states=[]
 def geometry(x):
  w.qpos[qa]=x[:7];mujoco.mj_kinematics(m,w);h=w.xmat[pb].reshape(3,3);M=w.xpos[pb]+h@ml;root=w.xpos[pb]+h@rel;a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[pb];angle=float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))));return h,M,root,angle
 for u in np.linspace(0.,1.,101):
  prior=v.copy();target_angle=theta*(1-u);tilt=Rotation.from_rotvec(target_angle*axis).as_matrix();goal=M0+u*np.array([-.03,.06,.17]);low=np.r_[np.maximum(lo,v[:7]-.15),v[7]-.15];high=np.r_[np.minimum(hi,v[:7]+.15),v[7]+.15]
  def eq(x):
   h,*_=geometry(x);R=tilt@Rotation.from_euler('z',x[7]).as_matrix()@E.T;return 100*Rotation.from_matrix(h@R.T).as_rotvec()
  def ine(x):
   h,M,root,angle=geometry(x);return np.r_[50-angle,M-[.32,.10,min(.89,M0[2]-.001)],np.array([.75,.65,1.16])-M,root[2]-.82]
  def obj(x):
   h,M,root,angle=geometry(x);return 100*np.sum((M-goal)**2)+.04*np.sum((x[:7]-prior[:7])**2)+.02*(x[7]-prior[7])**2+.5*(1-np.cos(np.radians(angle)))
  if u>0:
   fit=minimize(obj,v,method='SLSQP',bounds=list(zip(low,high)),constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=ine)],options=dict(maxiter=180,ftol=1e-10));v=fit.x
  h,M,root,angle=geometry(v);err=float(np.degrees(np.linalg.norm(eq(v))/100));row=dict(progress=float(u),tube_target_tilt_deg=float(np.degrees(target_angle)),wrist_deg=angle,orientation_error_deg=err,mouth_world_m=M.tolist(),root_world_m=root.tolist(),gamma_deg=float(np.degrees(v[7])));rows.append(row)
  if err>.01 or np.min(ine(v))<-.00001:raise RuntimeError('Actual-E free-yaw return rejected '+str(row))
  qs.append(v[:7].copy());states.append(w.qpos.copy())
 qs=np.array(qs);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(qs,axis=0),axis=1))];keep=np.r_[True,np.diff(arc)>1e-12];interp=PchipInterpolator(arc[keep],qs[keep],axis=0);times=np.arange(1,round(duration_s/m.opt.timestep)+1)*m.opt.timestep;x=times/duration_s;progress=x*x*x*(10+x*(-15+6*x));commands=interp(arc[-1]*progress)+offset;seed=np.asarray(entry_arm_command);spline=CubicSpline(np.r_[0.,times],np.vstack([seed,commands]),axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))));speed=float(abs(spline(times,1)).max());margin=float(np.minimum(commands-m.jnt_range[ids,0],m.jnt_range[ids,1]-commands).min())
 acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0]for j in ids]);limited=m.actuator_ctrllimited[acts].astype(bool);control_margin=float(np.minimum(commands[:,limited]-m.actuator_ctrlrange[acts[limited],0],m.actuator_ctrlrange[acts[limited],1]-commands[:,limited]).min())if limited.any()else None
 if speed>2 or margin<.05 or(control_margin is not None and control_margin<0):raise RuntimeError('Original return speed/margin/control-range failed '+str((speed,margin,control_margin)))
 return dict(time=times,arm_commands=commands,angle_deg=None,segment='return',duration_s=float(duration_s),peak_joint_speed_rad_s=speed,min_joint_margin_rad=margin,planned_forearm_palm_max_deg=max(r['wrist_deg']for r in rows),source_state_time_s=float(d.time),reference_scope='Single actual return-entry E/ml read; free axial yaw, orientation hard, explicit mouth box and root floor; no live pose writes',measured_hand_to_tube_R=E.tolist(),measured_mouth_in_hand_m=ml.tolist(),rows=rows,geometric_arm_q=qs,conditional_qpos=np.array(states),preserved_servo_offset_rad=offset.tolist())


def install_return_planner():
 """Explicit process-local opt-in; existing API/file defaults remain unchanged."""
 import t1_human_pour_profile_v4 as api
 if getattr(api.plan_pour_segment_v4,'_actual_return_v7',False):return
 original=api.plan_pour_segment_v4
 def dispatch(m,d,profile,*,segment,ik_solver,entry_arm_command=None,duration_s=4.8,knot_dt_s=.08):
  if segment=='return' and profile.get('v7_actual_return_reanchor',False):
   plan=plan_return(m,d,profile,entry_arm_command=entry_arm_command,duration_s=duration_s)
   # Keep native arrays only in explicit conditional artifacts, not JSON runtime reports.
   plan.pop('geometric_arm_q',None);plan.pop('conditional_qpos',None)
   plan['privileged_actual_E_read_count']=1
   plan['actual_grip_success_not_inferred']=True
   return plan
  return original(m,d,profile,segment=segment,ik_solver=ik_solver,entry_arm_command=entry_arm_command,duration_s=duration_s,knot_dt_s=knot_dt_s)
 dispatch._actual_return_v7=True
 api.plan_pour_segment_v4=dispatch
 import t1_deferred_idle_restore_v5 as deferred
 deferred._DEFERRED_POUR.__globals__['plan_pour_segment_v4']=dispatch


def advance_human_pour_reanchored_return_v7(*args,**kwargs):
 install_return_planner()
 from t1_deferred_idle_restore_v5 import advance_human_pour_deferred_restore_v5
 return advance_human_pour_deferred_restore_v5(*args,**kwargs)

def adapt_launch_source(source):
 old='from t1_deferred_idle_restore_v5 import advance_human_pour_deferred_restore_v5 as advance_human_pour_v2'
 new='from t1_actual_frame_return_v7 import advance_human_pour_reanchored_return_v7 as advance_human_pour_v2'
 if source.count(old)!=1:raise ValueError('Actual-return API import must occur exactly once')
 return source.replace(old,new,1)


def plan_actual_root_descent_free_yaw(m,d,handles,arm_command,target_bottom_world_m,duration_s=2.5,ik_solver=None):
 """Original vertical rack insertion target, axial yaw free; fresh actual E at entry."""
 ids,qa,_=arm_indices(m,'left');w=mujoco.MjData(m);w.qpos[:]=d.qpos;mujoco.mj_kinematics(m,w);pb=m.body('sharpa_left/left_hand_C_MC').id;tb=m.body(handles['tube_body']).id
 H=w.xmat[pb].reshape(3,3).copy();T=w.xmat[tb].reshape(3,3).copy();E=H.T@T;rel=H.T@(w.xpos[tb]-w.xpos[pb]);P0=w.xpos[tb].copy();target=np.asarray(target_bottom_world_m);q0=w.qpos[qa].copy();offset=np.asarray(arm_command)-q0
 axis=np.cross([0.,0.,1.],T[:,2]);theta=float(np.arccos(np.clip(T[2,2],-1,1)));axis/=max(np.linalg.norm(axis),1e-12);G=Rotation.from_rotvec(-theta*axis).as_matrix()@T;v=np.r_[q0,np.arctan2(G[1,0],G[0,0])];lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;el=m.body('astribot_arm_left_link_5').id;wr=m.body('astribot_arm_left_link_6').id;mc=m.body('sharpa_left/left_middle_MCP_VL').id;qs=[];rows=[]
 for u in np.linspace(0.,1.,101):
  prior=v.copy();P=P0+u*(target-P0);tilt=Rotation.from_rotvec(theta*(1-u)*axis).as_matrix();low=np.r_[np.maximum(lo,v[:7]-.15),v[7]-.15];high=np.r_[np.minimum(hi,v[:7]+.15),v[7]+.15]
  def geo(x):
   w.qpos[qa]=x[:7];mujoco.mj_kinematics(m,w);h=w.xmat[pb].reshape(3,3);root=w.xpos[pb]+h@rel;a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[pb];wa=np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)));return h,root,wa
  def eq(x):
   h,root,_=geo(x);R=tilt@Rotation.from_euler('z',x[7]).as_matrix()@E.T;return np.r_[1000*(root-P),100*Rotation.from_matrix(h@R.T).as_rotvec()]
  def obj(x):
   h,root,wa=geo(x);return .05*(1-np.cos(np.radians(wa)))+np.sum((x[:7]-prior[:7])**2)+(x[7]-prior[7])**2
  if u>0:v=minimize(obj,v,method='SLSQP',bounds=list(zip(low,high)),constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=lambda x:50-geo(x)[2])],options=dict(maxiter=200,ftol=1e-10)).x
  er=eq(v);h,root,wa=geo(v);row=dict(progress=float(u),position_error_mm=float(np.linalg.norm(er[:3])),orientation_error_deg=float(np.degrees(np.linalg.norm(er[3:])/100)),wrist_deg=float(wa),gamma_deg=float(np.degrees(v[7])));rows.append(row)
  if row['position_error_mm']>.1 or row['orientation_error_deg']>.01 or wa>50.00001:raise RuntimeError('Free-axial descent rejected '+str(row))
  qs.append(v[:7].copy())
 ts=np.linspace(0.,duration_s,101);curve=PchipInterpolator(ts,np.array(qs),axis=0);dense=np.arange(0.,duration_s+m.opt.timestep*.5,m.opt.timestep);u=dense/duration_s;prog=u*u*u*(10+u*(-15+6*u));commands=curve(duration_s*prog)+offset;spline=CubicSpline(dense,commands,axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))));speed=float(abs(spline(dense,1)).max());margin=float(np.minimum(commands-m.jnt_range[ids,0],m.jnt_range[ids,1]-commands).min())
 if speed>2 or margin<.05:raise RuntimeError('Original descent speed/margin reject '+str((speed,margin)))
 report=dict(scope=plan_actual_root_descent_free_yaw.__doc__,target_bottom_world_m=target.tolist(),actual_start_bottom_world_m=P0.tolist(),source_time_s=float(d.time),actual_E_privileged_simulation_geometry=True,command_peak_velocity_rad_s=speed,command_margin_rad=margin,rows=rows,geometry_or_physical_success_not_inferred=True)
 return dict(time=dense,arm_commands=commands,geometric_arm_q=curve(duration_s*prog),spline=spline,report=report)


def adapt_final_stage_source(source):
 """Run AFTER original coarse/fine transformation; all acceptance/release blocks kept."""
 from t1_adaptive_return_corner_v7 import adapt_stage_source
 source=adapt_stage_source(source)
 old="descent=plan_tube_descent(m,d,handles,armcmd,tube_initial+np.array([0,0,.001]),duration_s=2.5,ik_solver=continuous_left_solve_arm)"
 if source.count(old)!=1:raise ValueError('Original final Cartesian insertion anchor must occur once')
 new="from t1_actual_frame_return_v7 import plan_actual_root_descent_free_yaw\n   "+old.replace('plan_tube_descent','plan_actual_root_descent_free_yaw',1)
 result=source.replace(old,new,1);compile(result,'<actual_return_and_yaw_free_insertion_v7>','exec');return result

def adapt_workflow_launch_source(source,profile=None):
 from t1_left_opposite_workflow_v7 import adapt_launch_source as original_left
 source=original_left(source,profile=profile)
 source=adapt_launch_source(source)
 anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
 if source.count(anchor)!=1:raise ValueError('Final generated stage archive anchor must occur once')
 replacement="from t1_actual_frame_return_v7 import adapt_final_stage_source\\n        source=adapt_final_stage_source(source)\\n        "+anchor
 result=source.replace(anchor,replacement,1);compile(result,'<v7_actual_return_workflow_outer>','exec');return result
