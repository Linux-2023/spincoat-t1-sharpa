"""Independent traditional-glass stage factories, side-aware physical actuation.
Acceptedrecipe02 and its original preparation source remain unchanged.
"""
import inspect,json
from pathlib import Path
import t1_reagent_preparation_stage as legacy
from t1_glass_testtube_preparation import GlassPreparationConfig,GlassPreparationLedger
from t1_human_tube_mixing import advance_mouth_pivot_shake,MixingConfig
from t1_glass_pour_trajectory import resolve_pour_profile,pour_endpoints,pour_mouth_target,post_lift_settle_seconds,return_mouth_target,pour_rotation
from t1_glass_rack_clearance import rack_clearance_lift_m,rack_return_lift_m,route_waypoints,cartesian_transport_phase,left_palm_target_fk


def build_glass_stage(*,side,config_class,ledger_class,out,control_hook=None):
 if side not in ['left','right']:raise ValueError('Invalid manipulation side')
 source=inspect.getsource(legacy.advance_reagent_preparation);start=source.index('  # Gentle actual orientation oscillation');end=source.index('  # Capture actual grasp transform',start)
 block="""  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))
  if clear_lift>1e-6:phase('raise_clear_of_rack',2.,goal(d.xpos[palm].copy()+np.array([0,0,clear_lift]),d.xmat[palm].reshape(3,3).copy()))
  events.append(dict(time_s=float(d.time),phase='gentle_shake_mouth_pivot'))
  def mixing_step(qtarget,phase_name,allow_mix):
   nonlocal armcmd
   armcmd=np.asarray(qtarget).copy();sample(phase_name,mix=allow_mix)
  armcmd,mixing_report,mixing_trace=advance_mouth_pivot_shake(m,d,handles,armcmd,mixing_step,config=MixingConfig(tube_length_m=c.tube_length_m),side=MANIPULATION_SIDE,ik_solver=(continuous_left_solve_arm if MANIPULATION_SIDE=='left' and plan.get('continuous_left_ik',False) else None))
  (out/'mixing_report.json').write_text(json.dumps(mixing_report,indent=2));np.savez_compressed(out/'mixing_actual_trace.npz',**mixing_trace)
"""
 source=source[:start]+block+source[end:]
 source=source.replace(' armcmd=d.ctrl[armacts].copy()'," if 'open_command_hand_q_rad' in plan:openq=np.asarray(plan['open_command_hand_q_rad'],dtype=float).copy()\n armcmd=d.ctrl[armacts].copy()")
 source=source.replace('dock.step(d);d.qfrc_applied[:]', 'dock.step(d);CONTROL_HOOK(m,d,phase) if CONTROL_HOOK is not None else None;d.qfrc_applied[:]')
 source=source.replace("if 'thumb' not in m.joint(j).name:openq[i]*=.40","if any(digit in m.joint(j).name for digit in ['ring_','pinky_']):openq[i]=closed[i]\n  elif 'thumb' not in m.joint(j).name:openq[i]*=.90")
 source=source.replace("phase('approach_above_tube',3.,goal(pick+[0,0,.15],R));phase('lower_to_tube',2.5,goal(pick,R));","phase('approach_outside_tube',3.,goal(pick+np.asarray(plan.get('approach_offset_world_m',[-.10,0,0]))+[0,0,.08],R));phase('seat_from_side',3.5,goal(pick,R));")
 source=source.replace("phase('seat_from_side',3.5,goal(pick,R));","actual_tube_R=d.xmat[body].reshape(3,3).copy();R=actual_tube_R@R;pick=d.xpos[body].copy()+actual_tube_R@np.array([0,0,c.tube_grasp_height_m])-R@np.asarray(reach['mirrored_contact_offset_palm_m']);events.append(dict(time_s=float(d.time),phase='tube_6d_reanchor_before_seat',reference='Actual simulation tube pose GT baseline, no pose write or RGB detection claimed',tube_position_world_m=d.xpos[body].tolist(),tube_rotation_world=actual_tube_R.tolist()));phase('seat_from_side',3.5,goal(pick,R));")
 assert 'tube_6d_reanchor_before_seat' in source, 'Actual tube6D reanchor insertion failed'
 if side=='left':
  source=source.replace("phase('close_on_tube',3.,goalhand=closed);", "actual_tube_R=d.xmat[body].reshape(3,3).copy();R=actual_tube_R@np.asarray(reach['pickup_palm_R_world']);pick=d.xpos[body].copy()+actual_tube_R@np.array([0,0,c.tube_grasp_height_m])-R@np.asarray(reach['mirrored_contact_offset_palm_m']);events.append(dict(time_s=float(d.time),phase='tube_6d_reanchor_open_before_close',reference='GT geometric upper-bound baseline; actual free tube never pose-written',tube_position_world_m=d.xpos[body].tolist(),tube_rotation_world=actual_tube_R.tolist()));phase('seat_open_realignment',1.2,goal(pick,R));phase('close_on_tube',3.,goalhand=closed);")

 if side=='left':
  source=source.replace("   if goalhand is not None:handcmd=(1-u)*h0+u*np.asarray(goalhand)", "   if goalhand is not None:\n    handcmd=(1-u)*h0+u*np.asarray(goalhand)\n    if name=='close_on_tube' and 'close_joint_commands_rad' in plan:\n     table=np.asarray(plan['close_joint_commands_rad']);grid=np.asarray(plan['close_joint_progress_u']);handcmd=np.array([np.interp(u,grid,table[:,j]) for j in range(len(h0))])")
 if side=='left':
  source=source.replace("   if name=='pour_hold':", "   if name in ['close_on_tube','hold_before_tube_lift'] and plan.get('gt_geometry_tracking_during_close',False):\n    if k%100==0:\n     actual_tube_R=d.xmat[body].reshape(3,3).copy();grip_R=actual_tube_R@np.asarray(reach['pickup_palm_R_world']);grip_p=d.xpos[body].copy()+actual_tube_R@np.array([0,0,c.tube_grasp_height_m])-grip_R@np.asarray(reach['mirrored_contact_offset_palm_m']);feedback_anchor=armcmd.copy();feedback_goal=goal(grip_p,grip_R);events.append(dict(time_s=float(d.time),phase='gt_geometry_tracking_target',reference='10Hz full tube6D simulation GT upper-bound baseline, not image perception or tactile feedback',point_world_m=grip_p.tolist(),tube_root_world_m=d.xpos[body].tolist()))\n    blend=smooth((k%100+1)/100);armcmd=(1-blend)*feedback_anchor+blend*feedback_goal\n   if name=='pour_hold':")
  source=source.replace("beforelift=d.xpos[body].copy();", "pick=d.xpos[palm].copy();R=d.xmat[palm].reshape(3,3).copy();beforelift=d.xpos[body].copy();")
 source=source.replace("beforelift=d.xpos[body].copy();phase('lift_tube',3.,goal(pick+[0,0,.10],R));", "beforelift=d.xpos[body].copy();required_lift_m=max(.06,float(handles['rack_top_world_z_m'])+.025-float(tube_initial[2]));planned_lift_m=max(.10,required_lift_m+.020);phase('lift_tube',3.,goal(pick+[0,0,planned_lift_m],R));")
 source=source.replace("any(n.startswith(('prep/tube_holder','prep/receiver_nest','aspiration_source','stand/','spincoater/','lab_bench','pipette/')) for n in names)","any((n.startswith(c.prefix) and n != handles['tube_body']) or n.startswith(('aspiration_source','stand/','spincoater/','lab_bench','pipette/')) for n in names)")
 source=source.replace("upright_hand=bodyRrelative.T","upright_hand=Rotation.from_euler('z',getattr(c,'transport_yaw_deg',0.),degrees=True).as_matrix()@bodyRrelative.T")
 source=source.replace("desiredR=Rotation.from_euler('x',100,degrees=True).as_matrix()@local_body.T","desiredR=Rotation.from_euler('x',100,degrees=True).as_matrix()@Rotation.from_euler('z',getattr(c,'transport_yaw_deg',0.),degrees=True).as_matrix()@local_body.T")
 source=source.replace('required_lift_mm=60.','required_lift_mm=required_lift_m*1000,required_bottom_above_rack_top_mm=25.').replace('lift>=.06 and opposed and stable','lift>=required_lift_m and opposed and stable').replace('Actual lift>=60mm;','Actual lift>=max(60mm, rack-top+25mm minus initial tube-bottom);')
 source=source.replace('ledger=PreparationLedger(c);',"ledger=PreparationLedger(c);pour_profile=resolve_pour_profile(plan,c);(out/'used_pour_trajectory.json').write_text(json.dumps(pour_profile,indent=2));")
 source=source.replace("getattr(c,'transport_yaw_deg',0.)","pour_profile['transport_yaw_deg']")
 source=source.replace("Rotation.from_euler('x',100,degrees=True)","Rotation.from_euler('x',pour_profile['max_tilt_deg'],degrees=True)")
 source=source.replace('for angle in [25,50,75,100]:',"for angle in pour_profile['tilt_waypoints_deg']:")
 source=source.replace('mp=prepour+(pourmouth-prepour)*angle/100;', 'mp=pour_mouth_target(angle,prepour,pourmouth,pour_profile);')
 source=source.replace(";phase('carry_above_receiver',", ";prepour,pourmouth=pour_endpoints(prepour,pourmouth,pour_profile);phase('carry_above_receiver',")
 source=source.replace(';window=full[-500:];', ";settle_s=post_lift_settle_seconds(plan);phase('stabilize_lifted_tube',settle_s) if settle_s else None;window=full[-500:];")
 source=source.replace('for angle in [75,50,25,0]:',"for angle in pour_profile['return_tilt_waypoints_deg']:")
 source=source.replace('goal(prepour-Rp@mouth_local,Rp)', 'goal(return_mouth_target(angle,prepour,pour_profile)-Rp@mouth_local,Rp)')
 source=source.replace("goal(prepour-Rotation.from_euler('x',pour_profile['max_tilt_deg'],degrees=True).as_matrix()@upright_hand@mouth_local,", "goal(return_mouth_target(pour_profile['max_tilt_deg'],prepour,pour_profile)-Rotation.from_euler('x',pour_profile['max_tilt_deg'],degrees=True).as_matrix()@upright_hand@mouth_local,")
 source=source.replace('  # Place tube on the original holder using measured current grip transform.', "  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))\n  if clear_lift>1e-6:phase('raise_upright_clear_of_rack_before_return',2.,goal(d.xpos[palm].copy()+np.array([0,0,clear_lift]),d.xmat[palm].reshape(3,3).copy()))\n  # Place tube on the original holder using measured current grip transform.")
 source=source.replace('goal(tube_initial+[0,0,.10]-returnR@root_local,returnR)', 'goal(tube_initial+[0,0,rack_return_lift_m(plan,float(tube_initial[2]))]-returnR@root_local,returnR)')
 source=source.replace(";phase('carry_above_receiver',", "\n  for waypoint_index,mouth_waypoint in enumerate(route_waypoints(plan,'carry_mouth_waypoints_world_m')):\n   phase('carry_clearance_waypoint_'+str(waypoint_index),2.,goal(np.asarray(mouth_waypoint)-upright_hand@mouth_local,upright_hand))\n  phase('carry_above_receiver',")
 source=source.replace(";phase('return_above_holder',", "\n  for waypoint_index,root_waypoint in enumerate(route_waypoints(plan,'return_tube_bottom_waypoints_world_m')):\n   phase('return_clearance_waypoint_'+str(waypoint_index),2.,goal(np.asarray(root_waypoint)-returnR@root_local,returnR))\n  phase('return_above_holder',")
 descent_legacy="phase('lower_into_holder',2.5,goal(tube_initial+[0,0,.001]-returnR@root_local,returnR))"
 descent_block="""
  if plan.get('cartesian_rack_descent',False):
   if MANIPULATION_SIDE!='left':raise RuntimeError('Cartesian rack descent is LEFT-only')
   descent=plan_tube_descent(m,d,handles,armcmd,tube_initial+np.array([0,0,.001]),duration_s=2.5,ik_solver=continuous_left_solve_arm)
   events.append(dict(time_s=float(d.time),phase='lower_into_holder',target_mode='Actual-transform Cartesian target spline; no object state writes'))
   (out/'cartesian_rack_descent_report.json').write_text(json.dumps(descent['report'],indent=2))
   np.savez_compressed(out/'cartesian_rack_descent_plan.npz',time=descent['time'],arm_commands=descent['arm_commands'],geometric_arm_q=descent['geometric_arm_q'])
   for descent_step in range(round(2.5/m.opt.timestep)):
    armcmd=np.asarray(descent['spline']((descent_step+1)*m.opt.timestep));sample('lower_into_holder')
  else:"""+descent_legacy+"\n  "
 source=source.replace(';'+descent_legacy+';',descent_block)
 if side=='left':
  held_accept="  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')"
  assert source.count(held_accept)==1
  source=source.replace(held_accept,held_accept+"\n  if plan.get('freeze_left_integrator_after_held',False):\n   frozen_offset=FREEZE_LEFT_ENCODER();events.append(dict(time_s=float(d.time),phase='freeze_left_integrator_after_accepted_held',existing_offset_rad=frozen_offset.tolist(),scope='Retain existing encoder integral; no ctrl jump or force feedback'))")
 if side=='left':
  phase_anchor="  for k in range(int(round(duration/m.opt.timestep))):"
  duration_anchor="  nonlocal armcmd,handcmd\n"
  duration_block="""  if name in plan.get('phase_duration_overrides_s',{}):
   requested_duration=float(duration)
   override=float(plan['phase_duration_overrides_s'][name])
   if not cartesian_transport_phase(name) or not np.isfinite(override) or not requested_duration<=override<=30.:
    raise ValueError('Invalid LEFT Cartesian transport duration override: '+name)
   duration=override
   events.append(dict(time_s=float(d.time),phase='left_transport_duration_override',transport_phase=name,requested_duration_s=requested_duration,executed_duration_s=duration,scope='Continuous trajectory timing only; no hold, grasp, physical parameter, or acceptance gate changes'))
"""
  assert source.count(duration_anchor)==1
  source=source.replace(duration_anchor,duration_anchor+duration_block)
  transport_block="""  transport=None
  if plan.get('cartesian_left_transport',False) and held_gate is not None and held_gate['pass_'] and goalarm is not None and cartesian_transport_phase(name):
   target_p,target_R=left_palm_target_fk(m,d,goalarm)
   transport=plan_palm_cartesian_move(m,d,armcmd,target_p,target_R,duration,ik_solver=continuous_left_solve_arm)
   (out/('cartesian_transport_'+name+'_report.json')).write_text(json.dumps(transport['report'],indent=2))
   np.savez_compressed(out/('cartesian_transport_'+name+'_plan.npz'),time=transport['time'],arm_commands=transport['arm_commands'],geometric_arm_q=transport['geometric_arm_q'])
"""
  assert source.count(phase_anchor)==1
  source=source.replace(phase_anchor,transport_block+phase_anchor)
  source=source.replace('if goalarm is not None:armcmd=(1-u)*q0+u*np.asarray(goalarm)', "if goalarm is not None:armcmd=np.asarray(transport['spline']((k+1)*m.opt.timestep)) if transport is not None else (1-u)*q0+u*np.asarray(goalarm)")
 if side=='left':
  rotation_max="Rotation.from_euler('x',pour_profile['max_tilt_deg'],degrees=True).as_matrix()"
  rotation_angle="Rotation.from_euler('x',angle,degrees=True).as_matrix()"
  assert source.count(rotation_max)==3 and source.count(rotation_angle)==2, 'LEFT tilt-axis wiring changed'
  source=source.replace(rotation_max,"pour_rotation(pour_profile['max_tilt_deg'],pour_profile)").replace(rotation_angle,"pour_rotation(angle,pour_profile)")
 if side=='left':source=source.replace('arm_indices(m)','arm_indices(m,\'left\')').replace('hand_indices(m)','hand_indices(m,\'left\')').replace('sharpa_right/','sharpa_left/').replace('/right_','/left_').replace("side='right'","side='left'")
 if side=='left':
  source=source.replace('q,err=solve_arm(',"q,err=(continuous_left_solve_arm if plan.get('continuous_left_ik',False) and held_gate is not None and held_gate['pass_'] else solve_arm)(")
 encoder=None
 if side=='left':
  from t1_left_hand_encoder_tracking import LeftHandEncoderTracker
  encoder=LeftHandEncoderTracker(out)
 def combined_hook(m,d,phase):
  if control_hook is not None:control_hook(m,d,phase)
  if encoder is not None:encoder.step(m,d,phase)
 def freeze_left_encoder():
  if encoder is None:raise RuntimeError('Integral freeze is available only for the new LEFT tracker')
  return encoder.freeze_integrator()
 out=Path(out);out.mkdir(parents=True,exist_ok=True);filename=out/('executed_'+side+'_glass_stage.py');filename.write_text(source);env=dict(legacy.advance_reagent_preparation.__globals__);env.update(PreparationConfig=config_class,PreparationLedger=ledger_class,advance_mouth_pivot_shake=advance_mouth_pivot_shake,MixingConfig=MixingConfig,MANIPULATION_SIDE=side,CONTROL_HOOK=combined_hook,resolve_pour_profile=resolve_pour_profile,pour_endpoints=pour_endpoints,pour_mouth_target=pour_mouth_target,post_lift_settle_seconds=post_lift_settle_seconds,return_mouth_target=return_mouth_target,pour_rotation=pour_rotation,rack_clearance_lift_m=rack_clearance_lift_m,rack_return_lift_m=rack_return_lift_m,route_waypoints=route_waypoints,cartesian_transport_phase=cartesian_transport_phase,left_palm_target_fk=left_palm_target_fk)
 if side=='left':
  from t1_left_continuous_ik import solve_arm as continuous_left_solve_arm
  from t1_left_cartesian_segment import plan_tube_descent,plan_palm_cartesian_move
  env['continuous_left_solve_arm']=continuous_left_solve_arm
  env['plan_tube_descent']=plan_tube_descent
  env['plan_palm_cartesian_move']=plan_palm_cartesian_move
  env['FREEZE_LEFT_ENCODER']=freeze_left_encoder
 exec(compile(source,str(filename),'exec'),env);fn=env['advance_reagent_preparation'];fn.encoder_tracker=encoder;return fn


def advance_reagent_preparation(m,d,dock,handles,out,*,plan=None,return_to_ready_joint_target=None):
 if handles.get('manipulation_side',handles.get('preparation_side','right'))=='left':
  from t1_left_glass_preparation_stage import advance_reagent_preparation as left_advance
  return left_advance(m,d,dock,handles,out,plan=plan,return_to_ready_joint_target=return_to_ready_joint_target)
 if handles.get('preparation_asset_type')!='traditional_glass_testtube':raise ValueError('Traditional glass/rack handles required')
 if plan is None:raise ValueError('Explicit newly validated traditional-tube grasp plan required')
 fn=build_glass_stage(side='right',config_class=GlassPreparationConfig,ledger_class=GlassPreparationLedger,out=out);return fn(m,d,dock,handles,out,plan=plan,return_to_ready_joint_target=return_to_ready_joint_target)
