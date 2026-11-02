"""Conditional/same-live v7 forward planner for the corrected provisional anatomical LEFT mount.
All changes occur in private FK data; runtime control/physics/held gates remain in original stage.
"""
import copy,json,numpy as np,mujoco
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices
from t1_left_standing_workflow_v6 import _native

def plan_frame(m,d,profile,E,ml,rel):
 p=copy.deepcopy(profile);ids,qa,_=arm_indices(m,"left");w=mujoco.MjData(m);w.qpos[:]=d.qpos;pb=m.body("sharpa_left/left_hand_C_MC").id;tq=int(m.joint("prep/reagent_tube_free").qposadr[0]);el=m.body("astribot_arm_left_link_5").id;wr=m.body("astribot_arm_left_link_6").id;mc=m.body("sharpa_left/left_middle_MCP_VL").id;lo=m.jnt_range[ids,0]+.0501;hi=m.jnt_range[ids,1]-.0501;axis=np.r_[np.cos(np.radians(p["tilt_axis_azimuth_deg"])),np.sin(np.radians(p["tilt_axis_azimuth_deg"])),0.];curve=PchipInterpolator(np.array(p["v7_nominal_mouth_xyz_waypoints_m"])[:,0],np.array(p["v7_nominal_mouth_xyz_waypoints_m"])[:,1:],axis=0);v=np.r_[np.array(p["joint_reference_keyframes_rad"])[0,1:],np.radians(p["axial_yaw_waypoints_deg"][0][1])];rows=[];states=[];free_below=float(p["v7_free_mouth_below_tilt_deg"]);box0=np.array(p["v7_preflow_mouth_box_m"])
 def wrist():
  a=w.xpos[wr]-w.xpos[el];b=w.xpos[mc]-w.xpos[pb];return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
 for angle in np.linspace(0,p["max_tilt_deg"],int(round(p["max_tilt_deg"]))+1):
  prior=v.copy();free=angle<free_below;nom=curve(angle);tilt=Rotation.from_rotvec(np.radians(angle)*axis).as_matrix();box=box0.copy();zz=np.array(p["v7_preflow_zmin_waypoints_m"]);xx=np.array(p["v7_preflow_front_x_limit_waypoints_m"]);box[2,0]=np.interp(angle,zz[:,0],zz[:,1]);box[0,1]=min(box[0,1],np.interp(angle,xx[:,0],xx[:,1]));lower=np.r_[lo,-np.pi]if angle==0 else np.r_[np.maximum(lo,v[:7]-.15),v[7]-.15];upper=np.r_[hi,np.pi]if angle==0 else np.r_[np.minimum(hi,v[:7]+.15),v[7]+.15]
  def calc(x):
   w.qpos[qa]=x[:7];mujoco.mj_kinematics(m,w);H=w.xmat[pb].reshape(3,3);pos=w.xpos[pb]+H@ml;R=tilt@Rotation.from_euler("z",x[7]).as_matrix()@E.T;return pos,100*Rotation.from_matrix(H@R.T).as_rotvec()
  def eq(x):
   pos,ori=calc(x);return ori if free else np.r_[1000*(pos-nom),ori]
  def inequality(x):
   pos,_=calc(x);up=w.xmat[pb].reshape(3,3)[2,0];base=np.r_[50-wrist(),(-.25 if angle>=75 else .25)-up];return np.r_[base,pos-box[:,0],box[:,1]-pos]if free else base
  def obj(x):
   pos,_=calc(x);weight=40+40000*np.clip((angle-74)/6,0,1)**2 if free else 0.;return weight*np.sum((pos-nom)**2)+1-np.cos(np.radians(wrist()))+.01*np.sum((x[:7]-prior[:7])**2)+.02*(x[7]-prior[7])**2+.00002*np.sum(1/np.minimum(x[:7]-m.jnt_range[ids,0],m.jnt_range[ids,1]-x[:7])**2)
  fit=minimize(obj,v,method="SLSQP",bounds=list(zip(lower,upper)),constraints=[dict(type="eq",fun=eq),dict(type="ineq",fun=inequality)],options=dict(maxiter=250,ftol=1e-10));v=fit.x;pos,ori=calc(v);bad=_native(m,w,pb,"prep/reagent_tube",tq,rel,E);row=dict(angle_deg=float(angle),arm_q_rad=v[:7].tolist(),gamma_deg=float(np.degrees(v[7])),mouth_world_m=pos.tolist(),position_error_mm=float(np.linalg.norm(pos-nom)*1000),orientation_error_deg=float(np.degrees(np.linalg.norm(ori)/100)),wrist_deg=wrist(),palm_normal_world_z=float(w.xmat[pb].reshape(3,3)[2,0]),bad=bad);rows.append(row);states.append(w.qpos.copy())
  if row["orientation_error_deg"]>.01 or(not free and row["position_error_mm"]>.1)or row["wrist_deg"]>50.000001 or bad or np.min(inequality(v))<-.00001:raise RuntimeError("Own-live anatomical v7 forward retarget rejected "+json.dumps(row))
 p.update(joint_reference_keyframes_rad=[[r["angle_deg"],*r["arm_q_rad"]]for r in rows],mouth_xyz_waypoints_m=[[r["angle_deg"],*r["mouth_world_m"]]for r in rows],axial_yaw_waypoints_deg=[[r["angle_deg"],r["gamma_deg"]]for r in rows],measured_hand_to_tube_R=E.tolist(),measured_mouth_in_hand_m=ml.tolist(),source_state_time_s=float(d.time),source_state_path="Own same-live actual mixed/folded geometry on new anatomical mount",dynamic_status="OWN_LIVE_PLANNED_REFERENCE_NOT_EXECUTED")
 return p,dict(scope=__doc__,actual_reference_time_s=float(d.time),reference_is_privileged_simulation_geometry=True,no_live_pose_or_qpos_writes=True,wrist_max_deg=max(r["wrist_deg"]for r in rows),node_count=len(rows),free_mouth_below_tilt_deg=free_below,declared_preflow_mouth_box_m=box0.tolist(),rows=rows),np.array(states)
