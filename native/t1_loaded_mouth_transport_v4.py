"""Independent operative-mouth transport; preserves v3 numeric motion guards.

Unlike converting every waypoint with the final hand orientation, this follows
the declared world mouth curve throughout axial rolling: P=M-R*mouth_local.
Only planning MjData is pose-written; execution delegates unchanged to the
original actuator-step callback and never writes the live robot/object state.
"""
import inspect,types,hashlib
import numpy as np
import t1_coarse_fine_motion_v3 as v3

def _build():
 s=inspect.getsource(v3.plan_free_corridor_transport)
 def change(old,new):
  nonlocal s
  if s.count(old)!=1:raise RuntimeError('V4 mouth transport source anchor mismatch '+repr(old[:90]))
  s=s.replace(old,new)
 change('    points = [initial_P]',"    mouth_id=m.site('prep/tube_mouth').id\n    mouth_local=initial_R.T@(work.site_xpos[mouth_id]-initial_P)\n    initial_M=initial_P+initial_R@mouth_local\n    points = [initial_M]")
 change('    for i, p in enumerate(np.asarray(palm_waypoints_world_m,dtype=float)):', '    for i, palm_point in enumerate(np.asarray(palm_waypoints_world_m,dtype=float)):\n        p=palm_point+np.asarray(target_palm_R_world)@mouth_local')
 change('        P = curve(arc[-1]*u)\n        R = Rotation.from_rotvec(rotation_vector*u).as_matrix()@initial_R', '        M = curve(arc[-1]*u)\n        R = Rotation.from_rotvec(rotation_vector*u).as_matrix()@initial_R\n        P = M-R@mouth_local')
 change('planned_palm_waypoints_world_m=points.tolist()', 'planned_mouth_waypoints_world_m=points.tolist()')
 change('    return dict(time=times,arm_commands=commands,geometric_arm_q=qs,', "    report.update(operative_point='Measured tube mouth; exact world path during geometric hand rotation',planner_adapter_sha256=V4_MOUTH_ADAPTER_HASH,scope='V4 free-corridor mouth-pivot geometry only; original numeric margin/speed/control-range/corner guards retained; no physical/contact success inferred.')\n    return dict(time=times,target_mouth_world_m=curve(arc[-1]*_minimum_jerk(times/duration)),arm_commands=commands,geometric_arm_q=qs,")
 ns=dict(v3.plan_free_corridor_transport.__globals__,V4_MOUTH_ADAPTER_HASH=hashlib.sha256(inspect.getsource(_build).encode()).hexdigest())
 exec(compile(s,__file__+' [isolated mouth-pivot planner]','exec'),ns)
 plan=ns['plan_free_corridor_transport']
 env=dict(v3.advance_free_corridor_transport.__globals__,plan_free_corridor_transport=plan)
 advance=types.FunctionType(v3.advance_free_corridor_transport.__code__,env,name='advance_free_mouth_corridor_transport',argdefs=v3.advance_free_corridor_transport.__defaults__,closure=v3.advance_free_corridor_transport.__closure__)
 advance.__kwdefaults__=v3.advance_free_corridor_transport.__kwdefaults__
 return plan,advance,s

plan_free_mouth_corridor_transport,advance_free_mouth_corridor_transport,GENERATED_PLAN_SOURCE=_build()

def advance_loaded_transport(*args,**kwargs):
 """New mouth-aware carry; preserve the established standing rack return."""
 if kwargs.get('phase_name')=='coarse_carry_to_receiver':return advance_free_mouth_corridor_transport(*args,**kwargs)
 return v3.advance_free_corridor_transport(*args,**kwargs)
