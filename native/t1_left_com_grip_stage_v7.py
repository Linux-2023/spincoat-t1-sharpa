"""Opt-in COM-nearer controller grasp height and constant-palm pregrip path.
No physical parameter, primary nominal shape, force/gain, or acceptance changes.
"""
def adapt_source(source):
 old=" c=PreparationConfig(**handles['config']);"
 if source.count(old)!=1:raise RuntimeError("COM grip config anchor must occur once")
 new=""" if 'v7_grip_height_m' in plan:
  requested_height=float(plan['v7_grip_height_m'])
  if not np.isfinite(requested_height) or not 0.<requested_height<.150:raise ValueError('Invalid explicit tube controller grasp height')
  handles=dict(handles);handles['config']=dict(handles['config']);handles['config']['tube_grasp_height_m']=requested_height
  (out/'explicit_com_grasp_height_v7.json').write_text(json.dumps(dict(controller_tube_grasp_height_m=requested_height,scope='Explicit geometric controller target; tube/rack mass,geometry,friction,gains,caps and acceptance unchanged'),indent=2))
 c=PreparationConfig(**handles['config']);"""
 source=source.replace(old,new)
 old="  if plan.get('cartesian_left_transport',False) and held_gate is not None and held_gate['pass_'] and goalarm is not None and cartesian_transport_phase(name):"
 if source.count(old)!=1:raise RuntimeError("COM grip phase transport anchor must occur once")
 new="  if goalarm is not None and ((plan.get('v7_cartesian_pregrip',False) and name in ['seat_from_side','seat_open_realignment','lift_tube']) or (plan.get('cartesian_left_transport',False) and held_gate is not None and held_gate['pass_'] and cartesian_transport_phase(name))):"
 source=source.replace(old,new)
 return bind_pregrip_solver(source)

def bind_pregrip_solver(source):
 """Use exactly the validated neutral solver pregrip; preserve postheld dispatcher."""
 lines=source.splitlines();matches=[x for x in lines if x.startswith('   transport=plan_palm_cartesian_move(m,d,armcmd,target_p,target_R,duration,ik_solver=')]
 if len(matches)!=1:raise RuntimeError('COM pregrip solver call anchor must occur once')
 line=matches[0];prefix,expr=line.rsplit('ik_solver=',1)
 if not expr.endswith(')'):raise RuntimeError('COM pregrip transport call shape changed')
 expr=expr[:-1]
 condition="(plan.get('v7_cartesian_pregrip',False) and name in ['seat_from_side','seat_open_realignment','lift_tube'])"
 replacement=prefix+'ik_solver=(solve_arm if '+condition+' else ('+expr+')))'
 event="   if "+condition+":events.append(dict(time_s=float(d.time),phase='v7_pregrip_solver_selected',transport_phase=name,solver_module=getattr(solve_arm,'__module__','unknown'),solver_qualname=getattr(solve_arm,'__qualname__','unknown'),continuous_solver_module=getattr(continuous_left_solve_arm,'__module__','unknown'),scope='Same neutral pregrip solver as independently validated fresh43; postheld loaded solver unchanged'))"
 return source.replace(line,event+'\n'+replacement)
