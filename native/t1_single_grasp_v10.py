"""V10: single synchronized final approach and closure, no independent preshape."""
import numpy as np
from t1_transition_flow_v8 import once

def closing_target(start,closed,u):
 h=np.asarray(start)+(np.asarray(closed)-start)*u
 h[2]-=.5*np.sin(np.pi*u)
 h[4]=start[4]+(closed[4]-start[4])*u**3
 return h

def adapt_left_stage(source):
 source=once(source,"fine_started[0]=float(d.time);phase('open_hand',.6,goalhand=openq);reach=", "fine_started[0]=float(d.time);reach=")
 begin="phase('seat_from_side',3.5,goal(pick,R));"
 end="phase('close_on_tube',3.,goalhand=closed);"
 a=source.index(begin);b=source.index(end,a)+len(end)
 replacement="events.append(dict(time_s=float(d.time),phase='v10_single_grasp_start',scope='Single4.5s Cartesian final approach plus C2 finger closure; no intermediate open/seat/realignment phases',initial_hand_command_rad=handcmd.tolist(),final_hand_command_rad=closed.tolist(),target_palm_world_m=pick.tolist()));phase('close_on_tube',4.5,goalarm=goal(pick,R),goalhand=closed);"
 source=source[:a]+replacement+source[b:]
 old="['seat_from_side','seat_open_realignment','lift_tube']"
 if source.count(old)!=3:raise ValueError('Expected3 pregrip Cartesian anchors')
 source=source.replace(old,"['seat_from_side','seat_open_realignment','lift_tube','close_on_tube']")
 old="table=np.asarray(plan['close_joint_commands_rad']);grid=np.asarray(plan['close_joint_progress_u']);handcmd=np.array([np.interp(u,grid,table[:,j]) for j in range(len(h0))])"
 source=once(source,old,"from t1_single_grasp_v10 import closing_target;handcmd=closing_target(h0,closed,u)")
 source=once(source,"if name in ['close_on_tube','hold_before_tube_lift'] and plan.get('gt_geometry_tracking_during_close',False):", "if name=='hold_before_tube_lift' and plan.get('v10_track_tube_during_hold',False):")
 old="phase('pre_mix_clearance_'+str(pre_index),2.,goal(np.asarray(waypoint)-measured_R@measured_root,measured_R))"
 new="tube_axis=d.xmat[body].reshape(3,3)[:,2].copy();upright_axis=np.cross(tube_axis,np.array([0.,0.,1.]));upright_angle=float(np.arccos(np.clip(tube_axis[2],-1,1)));upright_R=Rotation.from_rotvec(upright_axis/max(np.linalg.norm(upright_axis),1e-12)*upright_angle).as_matrix()@measured_R;events.append(dict(time_s=float(d.time),phase='v10_upright_during_existing_clearance',initial_tube_tilt_deg=float(np.degrees(upright_angle)),scope='Physical arm orientation correction during existing2s clearance; actual hand-tube transform read once, no object pose write or extra hold'));phase('pre_mix_clearance_'+str(pre_index),2.,goal(np.asarray(waypoint)-upright_R@measured_root,upright_R))"
 source=once(source,old,new)
 old="np.asarray(transport['spline']((k+1)*m.opt.timestep)) if transport is not None"
 new="np.asarray(transport['spline'](min(duration,(k+1)*m.opt.timestep/1.0) if name=='close_on_tube' else (k+1)*m.opt.timestep)) if transport is not None"
 source=once(source,old,new)
 old="(out/('cartesian_transport_'+name+'_report.json')).write_text(json.dumps(transport['report'],indent=2))"
 new="if name=='close_on_tube':\n    transport['report']['v10_arm_duration_fraction']=1.0\n    transport['report']['v10_retimed_peak_joint_speed_rad_s']=transport['report']['sampled_1ms_command_peak_velocity_rad_s']/1.0\n    if transport['report']['v10_retimed_peak_joint_speed_rad_s']>2.:raise RuntimeError('Retimed approach exceeds2rad/s')\n   "+old
 source=once(source,old,new)
 compile(source,'<v10 single synchronized grasp>','exec');return source

def adapt_launch(source):
 anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
 return once(source,anchor,"from t1_single_grasp_v10 import adapt_left_stage as adapt_single_grasp\\n        source=adapt_single_grasp(source)\\n        "+anchor)
