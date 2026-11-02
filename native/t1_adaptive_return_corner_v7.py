"""V7 opt-in return-only C2 corner reduction; original 12mm guard retained."""
import types
import numpy as np
from scipy.interpolate import BPoly
import t1_coarse_fine_motion_v3 as base

def _bounded_path(points):
    points=np.asarray(points,float)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    if (lengths<1e-6).any():raise ValueError('Duplicate free-corridor points')
    arc=np.r_[0.,np.cumsum(lengths)]
    directions=np.diff(points,axis=0)/lengths[:,None]
    for attempt in range(13):
        scale=.35*(.5**attempt) if attempt<12 else 0.
        tangent=np.empty_like(points);tangent[0]=directions[0];tangent[-1]=directions[-1]
        for i in range(1,len(points)-1):
            v=directions[i-1]+directions[i]
            if np.linalg.norm(v)<.1:raise ValueError('Reversal requires separate stopped maneuver')
            tangent[i]=scale*v/np.linalg.norm(v)
        curve=BPoly.from_derivatives(arc,[[p,v,np.zeros(3)] for p,v in zip(points,tangent)])
        # Dense geometric selection, followed by the unchanged original 1ms guard.
        deviation=base._distance_to_polyline(curve(np.linspace(0.,arc[-1],20001)),points)
        if deviation.max()<=.0119:
            curve.v7_corner_review=dict(interior_tangent_scale=float(scale),halving_attempt=attempt,dense_selection_deviation_mm=float(deviation.max()*1000),selection_samples=20001,original_limit_mm=12.)
            return arc,curve
    raise RuntimeError('No C2 corner reduction satisfies unchanged 12mm corridor')

def advance_return_transport(m,d,arm_command,palm_waypoints_world_m,target_palm_R_world,step_callback,*,phase_name,**kwargs):
    if phase_name!='coarse_return_free_corridor':
        return base.advance_free_corridor_transport(m,d,arm_command,palm_waypoints_world_m,target_palm_R_world,step_callback,phase_name=phase_name,**kwargs)
    env=dict(base.plan_free_corridor_transport.__globals__)
    chosen={}
    def path(points):
        arc,curve=_bounded_path(points);chosen.update(curve.v7_corner_review);return arc,curve
    env['_spatial_path']=path
    planner=types.FunctionType(base.plan_free_corridor_transport.__code__,env);planner.__kwdefaults__=base.plan_free_corridor_transport.__kwdefaults__
    adv_env=dict(base.advance_free_corridor_transport.__globals__);adv_env['plan_free_corridor_transport']=planner
    advance=types.FunctionType(base.advance_free_corridor_transport.__code__,adv_env);advance.__kwdefaults__=base.advance_free_corridor_transport.__kwdefaults__
    result=advance(m,d,arm_command,palm_waypoints_world_m,target_palm_R_world,step_callback,phase_name=phase_name,**kwargs)
    result[1]['adaptive_return_corner_v7']=chosen
    return result

def plan_return_transport(m,d,arm_command,points,R,**kwargs):
    env=dict(base.plan_free_corridor_transport.__globals__);review={}
    def path(points):
        arc,curve=_bounded_path(points);review.update(curve.v7_corner_review);return arc,curve
    env['_spatial_path']=path
    planner=types.FunctionType(base.plan_free_corridor_transport.__code__,env);planner.__kwdefaults__=base.plan_free_corridor_transport.__kwdefaults__
    result=planner(m,d,arm_command,points,R,**kwargs);result['report']['adaptive_return_corner_v7']=review
    return result

def adapt_stage_source(source):
    old="armcmd,coarse_report,coarse_trace=V3_ADVANCE_COARSE(m,d,armcmd,return_points[:-1],returnR,coarse_transport_step,phase_name='coarse_return_free_corridor',ik_solver=continuous_left_solve_arm)"
    if source.count(old)!=1:raise ValueError('V7 adaptive return anchor must occur once')
    new="from t1_adaptive_return_corner_v7 import advance_return_transport\n  "+old.replace('V3_ADVANCE_COARSE','advance_return_transport',1)
    result=source.replace(old,new,1);compile(result,'<adaptive_return_v7>','exec');return result
