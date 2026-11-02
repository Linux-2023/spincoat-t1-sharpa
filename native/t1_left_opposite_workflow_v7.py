"""Opt-in v7 LEFT opposite/right-side pour; no v6 default changes or physical/gate tuning."""
import copy,json,numpy as np,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator
from t1_spincoat_scene import arm_indices
from t1_left_standing_workflow_v6 import _native,_work_pose,make_standing_transport

def _plan_frame_legacy(m,d,profile,E,ml,rel):
 p=copy.deepcopy(profile);ids,qa,_=arm_indices(m,"left");w=mujoco.MjData(m);w.qpos[:]=d.qpos;pb=m.body("sharpa_left/left_hand_C_MC").id;tb=m.body("prep/reagent_tube").id;tq=int(m.joint("prep/reagent_tube_free").qposadr[0]);ms=m.site("prep/tube_mouth").id;el=m.body("astribot_arm_left_link_5").id;wr=m.body("astribot_arm_left_link_6").id;mc=m.body("sharpa_left/left_middle_MCP_VL").id;lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;maximum=float(p["max_tilt_deg"]);az=np.radians(float(p["tilt_axis_azimuth_deg"]));axis=np.r_[np.cos(az),np.sin(az),0.];M=np.array(p.get("v7_nominal_mouth_xyz_waypoints_m",p["mouth_xyz_waypoints_m"]));curve=PchipInterpolator(M[:,0],M[:,1:],axis=0);seed=np.array(p["joint_reference_keyframes_rad"][-1][1:]);gamma0=np.radians(float(p.get("v7_axial_yaw_end_deg",-38)));tilt=Rotation.from_rotvec(np.radians(maximum)*axis).as_matrix();mouth=curve(maximum)
 def wrist():
  a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[pb];return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
 def endpoint(v):
  R=tilt@Rotation.from_euler("z",v[7]).as_matrix()@E.T;return _work_pose(m,w,qa,pb,v[:7],mouth-R@ml,R)
 def endobj(v):endpoint(v);return 1-np.cos(np.radians(wrist()))+.002*np.sum((v[:7]-seed)**2)+.001*(v[7]-gamma0)**2
 fit=minimize(endobj,np.r_[np.clip(seed,lo,hi),gamma0],method="SLSQP",bounds=list(zip(np.r_[lo,-np.pi],np.r_[hi,np.pi])),constraints=[dict(type="eq",fun=endpoint),dict(type="ineq",fun=lambda v: (-endpoint(v)[0]*0-.25-w.xmat[pb].reshape(3,3)[2,0]))],options=dict(maxiter=350,ftol=1e-11));err=endpoint(fit.x);q=fit.x[:7].copy();gamma_end=float(np.degrees(fit.x[7]));bad=_native(m,w,pb,"prep/reagent_tube",tq,rel,E)
 if np.linalg.norm(err[:3])>.1 or np.degrees(np.linalg.norm(err[3:])/100)>.01 or wrist()>50 or bad:raise RuntimeError("V7 own-live opposite endpoint rejected "+json.dumps(dict(error=err.tolist(),wrist=wrist(),bad=bad)))
 rows=[];states=[];box=np.array(p.get("v7_preflow_mouth_box_m",[[.40,.70],[.15,.48],[.93,1.15]]));free_below=float(p.get("v7_free_mouth_below_tilt_deg",65.))
 for angle in np.linspace(maximum,0,int(round(maximum))+1):
  localbox=box.copy();front=np.array(p.get("v7_preflow_front_x_limit_waypoints_m",[[0,.53],[45,.53],[65,.61]]));localbox[0,1]=min(localbox[0,1],float(np.interp(angle,front[:,0],front[:,1])))
  u=np.clip(angle/50.,0,1);gam=gamma_end*u**3*(10+u*(-15+6*u));T=Rotation.from_rotvec(np.radians(angle)*axis).as_matrix()@Rotation.from_euler("z",gam,degrees=True).as_matrix();R=T@E.T;nom=curve(angle);prior=q.copy();lower=np.maximum(lo,q-.15);upper=np.minimum(hi,q+.15)
  def update(v):w.qpos[qa]=v;mujoco.mj_kinematics(m,w);return w.xpos[pb]+w.xmat[pb].reshape(3,3)@ml
  def ori(v):update(v);return 100*Rotation.from_matrix(w.xmat[pb].reshape(3,3)@R.T).as_rotvec()
  if angle>=free_below:
   def pose(v):return _work_pose(m,w,qa,pb,v,nom-R@ml,R)
   def obj(v):pose(v);return 1-np.cos(np.radians(wrist()))+.01*np.sum((v-prior)**2)
   cons=[dict(type="eq",fun=pose)]
  else:
   def obj(v):pos=update(v);return 40*np.sum((pos-nom)**2)+.00002*np.sum(1/np.minimum(v-m.jnt_range[ids,0],m.jnt_range[ids,1]-v)**2)+1-np.cos(np.radians(wrist()))+.01*np.sum((v-prior)**2)
   def bounds(v):pos=update(v);return np.r_[pos-localbox[:,0],localbox[:,1]-pos]
   cons=[dict(type="eq",fun=ori),dict(type="ineq",fun=bounds),dict(type="ineq",fun=lambda v:50-wrist_after(v))]
  def wrist_after(v):update(v);return wrist()
  fit=minimize(obj,q,method="SLSQP",bounds=list(zip(lower,upper)),constraints=cons,options=dict(maxiter=250,ftol=1e-11));q=fit.x;pos=update(q).copy();oe=float(np.degrees(np.linalg.norm(ori(q))/100));pe=float(np.linalg.norm(pos-nom)*1000);wrdeg=wrist();bad=_native(m,w,pb,"prep/reagent_tube",tq,rel,E);up=float(w.xmat[pb].reshape(3,3)[2,0]);row=dict(angle_deg=float(angle),gamma_deg=float(gam),arm_q_rad=q.tolist(),mouth_world_m=pos.tolist(),nominal_mouth_deviation_mm=pe,orientation_error_deg=oe,wrist_deg=wrdeg,palm_normal_world_z=up,free_position=bool(angle<free_below),bad=bad);rows.append(row);states.append(w.qpos.copy());outside=bool(np.any(pos<localbox[:,0]-1e-6)or np.any(pos>localbox[:,1]+1e-6))if angle<free_below else False
  if oe>.01 or (angle>=free_below and pe>.1)or wrdeg>50.000001 or bad or outside or (angle>=75 and up>-.25):raise RuntimeError("V7 own-live opposite path rejected "+json.dumps(row))
 p.update(joint_reference_keyframes_rad=[[r["angle_deg"],*r["arm_q_rad"]]for r in rows[::-1]],mouth_xyz_waypoints_m=[[r["angle_deg"],*r["mouth_world_m"]]for r in rows[::-1]],measured_hand_to_tube_R=E.tolist(),measured_mouth_in_hand_m=ml.tolist(),v7_axial_yaw_end_deg=gamma_end,source_state_time_s=float(d.time),source_state_path="Own same-live actual postmix/fold geometry",dynamic_status="PLANNED_FROM_OWN_LIVE_STATE_NOT_ACTUAL_EXECUTION");report=dict(scope="Opposite v7 exact orientation with original physical criteria; preflow mouth is explicit free-corridor FK-derived reference, flow>=65 exact mouth",actual_reference_time_s=float(d.time),node_count=len(rows),wrist_max_deg=max(r["wrist_deg"]for r in rows),gamma_end_deg=gamma_end,free_mouth_below_tilt_deg=free_below,declared_preflow_mouth_box_m=box.tolist(),rows=rows);return p,report,np.array(states)

def _plan_frame(m,d,profile,E,ml,rel):
 if profile.get("v7_planning_direction")=="forward_free_gamma":
  from t1_left_anatomical_forward_v7 import plan_frame
  return plan_frame(m,d,profile,E,ml,rel)
 return _plan_frame_legacy(m,d,profile,E,ml,rel)

def retarget_from_live(m,d,profile):
 pb=m.body("sharpa_left/left_hand_C_MC").id;tb=m.body("prep/reagent_tube").id;ms=m.site("prep/tube_mouth").id;H=d.xmat[pb].reshape(3,3).copy();E=H.T@d.xmat[tb].reshape(3,3);ml=H.T@(d.site_xpos[ms]-d.xpos[pb]);rel=H.T@(d.xpos[tb]-d.xpos[pb]);p,r,_=_plan_frame(m,d,profile,E,ml,rel);return p,r

def adapt_generated_stage_source(source):
 from t1_left_standing_workflow_v6 import adapt_generated_stage_source as original
 source=original(source);old="from t1_left_standing_workflow_v6 import retarget_from_live,make_standing_transport"
 if source.count(old)!=1:raise RuntimeError("V7 retarget import anchor mismatch")
 source=source.replace(old,"from t1_left_opposite_workflow_v7 import retarget_from_live,make_standing_transport")
 old="dock.station_xy=np.array([0.,.30]);dock.target=dock.station_xy.copy()"
 if source.count(old)!=1:raise RuntimeError("V7 initial dock anchor mismatch")
 source=source.replace(old,"dock.station_xy=np.array([0.,.15]);dock.target=dock.station_xy.copy()")
 from t1_left_com_grip_stage_v7 import adapt_source as adapt_com_grip
 return adapt_com_grip(source)

def adapt_launch_source(source,profile=None):
 from t1_left_standing_workflow_v6 import adapt_launch_source as original
 if profile is not None:
  from pathlib import Path
  p=json.loads(Path(profile).read_text())if not isinstance(profile,dict)else profile
  if not any(np.allclose(p["receiver_xy_m"],xy) for xy in [[.60,.15],[.55,.20]]):raise ValueError("V7 cup must match an explicit reviewed layout: (.60,.15) or(.55,.20)")
 source=original(source);old="from t1_left_standing_workflow_v6 import adapt_generated_stage_source"
 if source.count(old)!=1:raise RuntimeError("V7 outer stage adapter anchor mismatch")
 return source.replace(old,"from t1_left_opposite_workflow_v7 import adapt_generated_stage_source")
