"""Opt-in LEFT free-corridor whole-arm transport, with explicit fine endpoints.

Targets only. No live qpos, object, base, gains, friction or acceptance edits.
Spatial waypoint interpolation is C2 and has nonzero internal tangents. One
minimum-jerk time law replaces independent stop/start at every free waypoint.
This is an engineering trajectory, not motion captured from a human operator.
"""
from dataclasses import dataclass, asdict
import numpy as np
import mujoco
from scipy.interpolate import BPoly, make_interp_spline
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices


@dataclass(frozen=True)
class CoarseFineConfig:
    mean_translation_speed_m_s: float = .075
    minimum_duration_s: float = 1.2
    maximum_duration_s: float = 6.
    rotation_speed_rad_s: float = .35
    knot_dt_s: float = .04
    maximum_waypoint_deviation_m: float = .012
    maximum_command_speed_rad_s: float = 2.
    minimum_command_joint_margin_rad: float = .049


def transport_duration(distance_m, rotation_rad=0., *, config=None):
    c = config or CoarseFineConfig()
    # Average speed, not instantaneous cruise speed; minimum jerk peaks 1.875x.
    return min(c.maximum_duration_s, max(c.minimum_duration_s,
               float(distance_m)/c.mean_translation_speed_m_s,
               abs(float(rotation_rad))/c.rotation_speed_rad_s))


def _minimum_jerk(u):
    return u*u*u*(10.+u*(-15.+6.*u))


def _spatial_path(points):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (lengths < 1e-6).any():
        raise ValueError('Consecutive duplicate free-corridor points must be removed')
    arc = np.r_[0., np.cumsum(lengths)]
    directions = np.diff(points, axis=0)/lengths[:, None]
    tangent = np.empty_like(points)
    tangent[0], tangent[-1] = directions[0], directions[-1]
    for i in range(1, len(points)-1):
        v = directions[i-1]+directions[i]
        if np.linalg.norm(v) < .1:
            raise ValueError('Reversal requires a separate stopped fine maneuver')
        tangent[i] = .35*v/np.linalg.norm(v)
    curve = BPoly.from_derivatives(arc, [[p, v, np.zeros(3)]
                                         for p,v in zip(points,tangent)])
    return arc, curve


def _distance_to_polyline(P, points):
    best = np.full(len(P), np.inf)
    for a,b in zip(points[:-1], points[1:]):
        ab = b-a
        u = np.clip(((P-a)@ab)/(ab@ab),0.,1.)
        best = np.minimum(best,np.linalg.norm(P-(a+u[:,None]*ab),axis=1))
    return best


def plan_free_corridor_transport(m, d, arm_command, palm_waypoints_world_m,
                                 target_palm_R_world, *, config=None,
                                 duration_s=None, ik_solver=None):
    c = config or CoarseFineConfig()
    if ik_solver is None:
        from t1_left_continuous_ik import solve_arm
        ik_solver = solve_arm
    ids, qa, _ = arm_indices(m, 'left')
    palm = m.body('sharpa_left/left_hand_C_MC').id
    work = mujoco.MjData(m)
    work.qpos[:] = d.qpos
    mujoco.mj_kinematics(m,work)
    qref = np.asarray(d.qpos[qa]).copy()
    current = np.asarray(arm_command,dtype=float).copy()
    initial_P = work.xpos[palm].copy()
    initial_R = work.xmat[palm].reshape(3,3).copy()
    points = [initial_P]
    skipped = []
    for i, p in enumerate(np.asarray(palm_waypoints_world_m,dtype=float)):
        if np.linalg.norm(p-points[-1]) < .0005:
            skipped.append(i)
        else:
            points.append(p.copy())
    if len(points) < 2:
        raise ValueError('No meaningful free-corridor displacement; caller must skip explicitly')
    points = np.asarray(points)
    arc, curve = _spatial_path(points)
    rotation_vector = Rotation.from_matrix(np.asarray(target_palm_R_world)@initial_R.T).as_rotvec()
    duration = transport_duration(arc[-1],np.linalg.norm(rotation_vector),config=c) if duration_s is None else float(duration_s)
    if not .5 <= duration <= 30.:
        raise ValueError('Invalid independent free-corridor duration')
    dense = np.linspace(0.,duration,round(duration/m.opt.timestep)+1)
    progress = _minimum_jerk(dense/duration)
    dense_P = curve(arc[-1]*progress)
    deviation = _distance_to_polyline(dense_P,points)
    if deviation.max() > c.maximum_waypoint_deviation_m:
        raise RuntimeError('C2 free-corridor corner leaves explicitly bounded 12mm path tube')
    times = np.linspace(0.,duration,int(np.ceil(duration/c.knot_dt_s))+1)
    q = qref.copy()
    qs, errors, targets = [q.copy()], [], [initial_P]
    for t in times[1:]:
        u = _minimum_jerk(t/duration)
        P = curve(arc[-1]*u)
        R = Rotation.from_rotvec(rotation_vector*u).as_matrix()@initial_R
        q,e = ik_solver(m,work,P,R,side='left',seed=q)
        if e['position_error_mm'] > .1 or e['orientation_error_deg'] > .1:
            raise RuntimeError('Original geometric pose accuracy guard failed')
        qs.append(q.copy()); errors.append(e); targets.append(P)
    qs = np.asarray(qs)
    commands = qs+(current-qref)[None,:]
    bc = ([(1,np.zeros(7)),(2,np.zeros(7))],[(1,np.zeros(7)),(2,np.zeros(7))])
    spline = make_interp_spline(times,commands,k=5,bc_type=bc)
    sampled = spline(dense)
    velocity = spline(dense,1)
    margin = np.minimum(sampled-m.jnt_range[ids,0],m.jnt_range[ids,1]-sampled).min()
    acts = np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in ids])
    limited = m.actuator_ctrllimited[acts].astype(bool)
    control_margin = float(np.minimum(sampled[:,limited]-m.actuator_ctrlrange[acts[limited],0],
                                     m.actuator_ctrlrange[acts[limited],1]-sampled[:,limited]).min()) if limited.any() else None
    if margin < c.minimum_command_joint_margin_rad or abs(velocity).max() > c.maximum_command_speed_rad_s:
        raise RuntimeError('Original joint margin or 2rad/s planning guard failed')
    if control_margin is not None and control_margin < 0:
        raise RuntimeError('Original actuator control range failed')
    report = dict(config=asdict(c),duration_s=duration,polyline_distance_m=float(arc[-1]),
                  corner_deviation_max_mm=float(deviation.max()*1000),
                  planned_palm_waypoints_world_m=points.tolist(), skipped_redundant_waypoint_indices=skipped,
                  maximum_knot_position_error_mm=max(e['position_error_mm'] for e in errors),
                  maximum_knot_rotation_error_deg=max(e['orientation_error_deg'] for e in errors),
                  command_joint_margin_rad=float(margin),
                  actuator_control_margin_rad=control_margin,
                  maximum_command_speed_rad_s=float(abs(velocity).max()),
                  initial_command_jump_rad=float(abs(commands[0]-current).max()),
                  preserved_servo_offset_rad=(current-qref).tolist(),
                  internal_waypoint_stops=False, endpoint_velocity_and_acceleration_zero=True,
                  geometry_or_contact_validated=False, physical_success_not_inferred=True,
                  scope='Free corridor only. Separate fine rack entry/descent, grip gate, mixing, precision pour and base stop remain unchanged.')
    return dict(time=times,arm_commands=commands,geometric_arm_q=qs,
                target_palm_world_m=np.asarray(targets),spline=spline,report=report)


def advance_free_corridor_transport(m,d,arm_command,palm_waypoints_world_m,
                                    target_palm_R_world,step_callback,*,phase_name,
                                    config=None,duration_s=None,ik_solver=None):
    plan = plan_free_corridor_transport(m,d,arm_command,palm_waypoints_world_m,
                                       target_palm_R_world,config=config,
                                       duration_s=duration_s,ik_solver=ik_solver)
    duration = plan['report']['duration_s']
    start = float(d.time)
    for k in range(round(duration/m.opt.timestep)):
        before = float(d.time)
        step_callback(np.asarray(plan['spline'](min((k+1)*m.opt.timestep,duration))),phase_name)
        if abs(d.time-before-m.opt.timestep)>1e-9:
            raise RuntimeError('Caller must advance exactly one original physics step')
    plan['report'].update(execution_start_s=start,execution_end_s=float(d.time),execution_completed=True)
    return np.asarray(plan['spline'](duration)),plan['report'],dict(time=plan['time'],arm_commands=plan['arm_commands'],geometric_arm_q=plan['geometric_arm_q'],target_palm_world_m=plan['target_palm_world_m'])


def adapt_left_stage_source(source):
    """Adapt the independently generated human-v2 LEFT stage before exec.

    Strict, single-match anchors fail closed when upstream source changes.
    The last rack side-entry and descent remain original fine phases. A
    redundant rack-overhead move is skipped only when actual Cartesian pose
    already meets explicit 0.5mm/0.2deg tolerances; all grasp/pour gates remain.
    """
    carry = """  for waypoint_index,mouth_waypoint in enumerate(route_waypoints(plan,'carry_mouth_waypoints_world_m')):
   phase('carry_clearance_waypoint_'+str(waypoint_index),2.,goal(np.asarray(mouth_waypoint)-upright_hand@mouth_local,upright_hand))
  phase('carry_above_receiver',3.,goal(prepour-upright_hand@mouth_local,upright_hand))"""
    new_carry = """  def coarse_transport_step(qtarget,phase_name):
   nonlocal armcmd
   armcmd=np.asarray(qtarget).copy();sample(phase_name)
  if HUMAN_POUR_PROFILE.get('idle_fold_before_coarse_carry',False):
   if not HUMAN_POUR_PROFILE.get('idle_fold_already_applied',False):raise RuntimeError('Pre-carry idle fold requires explicit pour duplicate-fold suppression')
   for idle_phase in ['human_pour_idle_extend','human_pour_idle_fold']:
    for idle_step in range(round(float(HUMAN_POUR_PROFILE.get('idle_segment_s',1.2))/m.opt.timestep)):coarse_transport_step(armcmd,idle_phase)
   events.append(dict(time_s=float(d.time),phase='idle_fold_before_coarse_carry_complete',scope='Original idle ring/pinky overlay moved after mixing and before free carry; three primary grasp fingers unchanged'))
  coarse_receiver_mouth=prepour.copy()
  if 'mouth_xy_m' in HUMAN_POUR_PROFILE:coarse_receiver_mouth[:2]=np.asarray(HUMAN_POUR_PROFILE['mouth_xy_m'])
  if 'mouth_height_waypoints_m' in HUMAN_POUR_PROFILE:
   coarse_height=np.asarray(HUMAN_POUR_PROFILE['mouth_height_waypoints_m']);coarse_receiver_mouth[2]=np.interp(0.,coarse_height[:,0],coarse_height[:,1])
  carry_points=[np.asarray(P)-upright_hand@mouth_local for P in route_waypoints(plan,'carry_mouth_waypoints_world_m')]+[coarse_receiver_mouth-upright_hand@mouth_local]
  events.append(dict(time_s=float(d.time),phase='coarse_carry_to_receiver',scope='One C2 free-corridor trajectory; original precision entry gate follows'))
  armcmd,coarse_report,coarse_trace=V3_ADVANCE_COARSE(m,d,armcmd,carry_points,upright_hand,coarse_transport_step,phase_name='coarse_carry_to_receiver',ik_solver=continuous_left_solve_arm)
  (out/'coarse_carry_report.json').write_text(json.dumps(coarse_report,indent=2));np.savez_compressed(out/'coarse_carry_plan.npz',**coarse_trace)"""
    ret = """  for waypoint_index,root_waypoint in enumerate(route_waypoints(plan,'return_tube_bottom_waypoints_world_m')):
   phase('return_clearance_waypoint_'+str(waypoint_index),2.,goal(np.asarray(root_waypoint)-returnR@root_local,returnR))
  phase('return_above_holder',3.,goal(tube_initial+[0,0,rack_return_lift_m(plan,float(tube_initial[2]))]-returnR@root_local,returnR))"""
    new_ret = """  return_points=[np.asarray(P)-returnR@root_local for P in route_waypoints(plan,'return_tube_bottom_waypoints_world_m')]
  if len(return_points)<3:raise RuntimeError('Coarse/fine v3 requires explicit free return waypoints plus final fine rack side-entry')
  events.append(dict(time_s=float(d.time),phase='coarse_return_free_corridor',scope='Merge free return waypoints; preserve separate final rack side-entry and vertical descent'))
  armcmd,coarse_report,coarse_trace=V3_ADVANCE_COARSE(m,d,armcmd,return_points[:-1],returnR,coarse_transport_step,phase_name='coarse_return_free_corridor',ik_solver=continuous_left_solve_arm)
  (out/'coarse_return_report.json').write_text(json.dumps(coarse_report,indent=2));np.savez_compressed(out/'coarse_return_plan.npz',**coarse_trace)
  phase('return_clearance_waypoint_'+str(len(return_points)-1),2.,goal(return_points[-1],returnR))
  rack_above_target=tube_initial+[0,0,rack_return_lift_m(plan,float(tube_initial[2]))]-returnR@root_local
  rack_above_position_error=float(np.linalg.norm(d.xpos[palm]-rack_above_target))
  rack_above_orientation_error=float(np.degrees(Rotation.from_matrix(d.xmat[palm].reshape(3,3)@returnR.T).magnitude()))
  if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2:
   events.append(dict(time_s=float(d.time),phase='skip_redundant_return_above_holder',position_error_mm=rack_above_position_error*1000,orientation_error_deg=rack_above_orientation_error,maximum_position_error_mm=.5,maximum_orientation_error_deg=.2,scope='Skip already reached transport waypoint only; descent, release and acceptance unchanged'))
  else:phase('return_above_holder',3.,goal(rack_above_target,returnR))"""
    result = source.replace('\r\n','\n')
    for label,old,new in [('carry',carry,new_carry),('return',ret,new_ret)]:
        if result.count(old)!=1:
            raise ValueError('Coarse/fine v3 '+label+' source anchor must occur exactly once')
        result=result.replace(old,new,1)
    compile(result,'<coarse_fine_v3_adapted_left_stage>','exec')
    return result,dict(V3_ADVANCE_COARSE=advance_free_corridor_transport)
