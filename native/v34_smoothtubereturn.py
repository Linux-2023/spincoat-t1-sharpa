"""Stage-owned LEFT continuity adapter; native adapters and hard boundaries remain."""
from functools import wraps

CANDIDATE_ID = 'SmoothTubeReturn'
PATCHED_SYMBOLS = ('t1_empty_hand_ready_v7.adapt_final_stage_source',)
_INSTALLED = False

_LOADED = """  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))
  if clear_lift>1e-6:phase('raise_upright_clear_of_rack_before_return',2.,goal(d.xpos[palm].copy()+np.array([0,0,clear_lift]),d.xmat[palm].reshape(3,3).copy()))
  # Place tube on the original holder using measured current grip transform.
  Rh=d.xmat[palm].reshape(3,3);root_local=Rh.T@(d.xpos[body]-d.xpos[palm]);bodyRrelative=Rh.T@d.xmat[body].reshape(3,3);returnR=bodyRrelative.T
  return_points=[np.asarray(P)-returnR@root_local for P in route_waypoints(plan,'return_tube_bottom_waypoints_world_m')]
  if len(return_points)<3:raise RuntimeError('Coarse/fine v3 requires explicit free return waypoints plus final fine rack side-entry')
  events.append(dict(time_s=float(d.time),phase='coarse_return_free_corridor',scope='Merge free return waypoints; preserve separate final rack side-entry and vertical descent'))
  from t1_adaptive_return_corner_v7 import advance_return_transport
  armcmd,coarse_report,coarse_trace=advance_return_transport(m,d,armcmd,return_points[:-1],returnR,coarse_transport_step,phase_name='coarse_return_free_corridor',ik_solver=continuous_left_solve_arm)
  (out/'coarse_return_report.json').write_text(json.dumps(coarse_report,indent=2));np.savez_compressed(out/'coarse_return_plan.npz',**coarse_trace)
  phase('return_clearance_waypoint_'+str(len(return_points)-1),2.,goal(return_points[-1],returnR))"""

_LOADED_NEW = """  from v34_smoothtubereturn_reference import plan_loaded_return as _smooth_loaded_plan,advance_loaded as _smooth_loaded_advance
  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))
  # One actual grasp frame at this hard boundary; never rebase in the traversed join.
  Rh=d.xmat[palm].reshape(3,3);root_local=Rh.T@(d.xpos[body]-d.xpos[palm]);bodyRrelative=Rh.T@d.xmat[body].reshape(3,3);returnR=bodyRrelative.T
  return_points=[np.asarray(P)-returnR@root_local for P in route_waypoints(plan,'return_tube_bottom_waypoints_world_m')]
  if len(return_points)<3:raise RuntimeError('Coarse/fine v3 requires explicit free return waypoints plus final fine rack side-entry')
  _smooth_durations={}
  _smooth_names=(['raise_upright_clear_of_rack_before_return'] if clear_lift>1e-6 else [])+['return_clearance_waypoint_'+str(len(return_points)-1)]
  for _smooth_name in _smooth_names:
   if _smooth_name in plan.get('phase_duration_overrides_s',{}):
    _smooth_override=float(plan['phase_duration_overrides_s'][_smooth_name])
    if not cartesian_transport_phase(_smooth_name) or not np.isfinite(_smooth_override) or not 2.<=_smooth_override<=30.:raise ValueError('Invalid LEFT Cartesian transport duration override: '+_smooth_name)
    _smooth_durations[_smooth_name]=_smooth_override
  _smooth_loaded=_smooth_loaded_plan(m,d,handles,armcmd,clear_lift,return_points,returnR,ik_solver=continuous_left_solve_arm,audit_directory=out,minimum_clearance_duration_s=_smooth_durations.get('raise_upright_clear_of_rack_before_return',0.),side_entry_duration_s=_smooth_durations.get('return_clearance_waypoint_'+str(len(return_points)-1),2.))
  for _smooth_name,_smooth_override in _smooth_durations.items():
   events.append(dict(time_s=float(d.time),phase='left_transport_duration_override',transport_phase=_smooth_name,requested_duration_s=2.,executed_duration_s=float(_smooth_loaded['reference'].durations[_smooth_loaded['phases'].index(_smooth_name)]),native_minimum_override_s=_smooth_override,scope='Original validated duration override retained in coherent reference; no command clipping'))
  armcmd,coarse_report,coarse_trace=_smooth_loaded_advance(m,d,_smooth_loaded,coarse_transport_step,events)
  (out/'coarse_return_report.json').write_text(json.dumps(coarse_report,indent=2));np.savez_compressed(out/'coarse_return_plan.npz',**coarse_trace)"""

_EMPTY = """  if HUMAN_POUR_PROFILE.get('v7_empty_ready_clearance',False):
   empty_P=d.xpos[palm].copy()+np.array([-.10,0.,0.]);empty_R=d.xmat[palm].reshape(3,3).copy();phase('empty_hand_clearance_before_ready',1.5,goal(empty_P,empty_R))
  phase('return_to_original_ready_joints',3.,goalarm=ready_target[qa],goalhand=ready_target[hq]);phase('settle_original_ready',.7)"""

_EMPTY_NEW = """  if HUMAN_POUR_PROFILE.get('v7_empty_ready_clearance',False):
   from v34_smoothtubereturn_reference import plan_empty_ready as _smooth_empty_plan,advance_empty as _smooth_empty_advance
   empty_P=d.xpos[palm].copy()+np.array([-.10,0.,0.]);empty_R=d.xmat[palm].reshape(3,3).copy()
   _smooth_empty=_smooth_empty_plan(m,d,armcmd,goal(empty_P,empty_R),ready_target[qa],handcmd,ready_target[hq],ik_solver=continuous_left_solve_arm)
   def _smooth_empty_step(qtarget,htarget,phase_name):
    nonlocal handcmd
    handcmd=np.asarray(htarget).copy()
    coarse_transport_step(qtarget,phase_name)
   armcmd,handcmd,_smooth_empty_report,_smooth_empty_trace=_smooth_empty_advance(m,d,_smooth_empty,_smooth_empty_step,events)
   (out/'continuous_empty_ready_report.json').write_text(json.dumps(_smooth_empty_report,indent=2));np.savez_compressed(out/'continuous_empty_ready_plan.npz',**_smooth_empty_trace)
  else:
   phase('return_to_original_ready_joints',3.,goalarm=ready_target[qa],goalhand=ready_target[hq])
  phase('settle_original_ready',.7)"""

# These anchors are asserted, not replaced. Exact original failure boundaries
# remain between the spans, and native sample/holder/ledger code stays untouched.
_RETAINED = (
    "  if ledger.delivered_m3<max(c.target_pour_volume_m3,c.tube_initial_volume_m3)*.99:raise RuntimeError('Gravity pour delivered less than original99percent target; no flow valve/cap')",
    "  if rack_above_position_error<=.0005 and rack_above_orientation_error<=.2:",
    "   descent=plan_actual_root_descent_free_yaw(m,d,handles,armcmd,tube_initial+np.array([0,0,.001]),duration_s=2.5,ik_solver=continuous_left_solve_arm)",
    "  phase('release_tube',2.,goalhand=openq);phase('withdraw_empty_hand',2.,goal(tube_initial+[0,0,.15]-returnR@root_local,returnR));phase('human_pour_idle_restore_fold',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)));phase('human_pour_idle_restore_extend',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)))",
    "  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')",
    "  if not dock.stopped:raise RuntimeError('Actual base did not settle before precision pour')",
)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import t1_empty_hand_ready_v7 as empty

    previous = empty.adapt_final_stage_source

    @wraps(previous)
    def adapt_final_stage_source(source):
        source = previous(source)
        for anchor in (_LOADED, _EMPTY, *_RETAINED):
            if source.count(anchor) != 1:
                raise ValueError('SmoothTubeReturn requires exactly one executed-R05 anchor: ' + repr(anchor[:110]))
        source = source.replace(_LOADED, _LOADED_NEW, 1)
        source = source.replace(_EMPTY, _EMPTY_NEW, 1)
        compile(source, '<v34_smoothtubereturn_left_stage>', 'exec')
        return source

    adapt_final_stage_source._v34_smooth_tube_return = True
    empty.adapt_final_stage_source = adapt_final_stage_source
    _INSTALLED = True
