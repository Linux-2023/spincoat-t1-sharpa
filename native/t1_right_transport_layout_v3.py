"""RESET-only pipette station relocation and grasp-plan translation.

Moves the tall holder away from the low cup-to-coater transport corridor.
Never moves a live held object; all robot poses stay unchanged at reset.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import inspect
import numpy as np,mujoco

@dataclass(frozen=True)
class RightStationLayoutConfig:
    station_shift_world_m: tuple = (-.075,0.,0.)
    translate_grasp_targets_from_s: float = 2.9

def relocate_station_at_reset(scene_tuple,config=None):
    c=config or RightStationLayoutConfig();spec,m,d,children=scene_tuple
    if float(d.time)!=0.:raise ValueError('Pipette station may move only at scene reset')
    delta=np.asarray(c.station_shift_world_m,dtype=float);mujoco.mj_forward(m,d);before=d.qpos.copy()
    stand=m.body('stand/pipette_stand').id;parent=int(m.body_parentid[stand]);local_delta=d.xmat[parent].reshape(3,3).T@delta
    old=m.body_pos[stand].copy();m.body_pos[stand]+=local_delta;spec.body('stand/pipette_stand').pos[:]=np.asarray(spec.body('stand/pipette_stand').pos)+local_delta
    pipe=m.body('pipette/pipette').id;j=int(m.body_jntadr[pipe]);q=int(m.jnt_qposadr[j]);assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE
    d.qpos[q:q+3]+=delta;spec.body('pipette/pipette').pos[:]=np.asarray(spec.body('pipette/pipette').pos)+delta
    mask=np.ones(m.nq,bool);mask[q:q+7]=False;assert np.array_equal(before[mask],d.qpos[mask]);mujoco.mj_forward(m,d)
    report=dict(config=asdict(c),phase='RESET ONLY before simulation',stand_body='stand/pipette_stand',old_stand_local_pos_m=old.tolist(),new_stand_local_pos_m=m.body_pos[stand].tolist(),pipette_free_qpos=d.qpos[q:q+7].tolist(),robot_qpos_unchanged=True,physics_parameters_unchanged=True)
    return (spec,m,d,children),report

def translated_station_preparer(baseline,source_path,config=None):
    """Clone original station_prepare with translated legacy grasp world targets.

    Install this clone as t1_continuous_approach.baseline_station_prepare before
    the normal continuous/relative station factory is called.
    """
    c=config or RightStationLayoutConfig();source=inspect.getsource(baseline)
    anchor='def transform_point(point):return center+turn@(np.asarray(point)-center)'
    if source.count(anchor)!=1:raise ValueError('Unsupported station_prepare transform')
    source=source.replace(anchor,"def transform_point(point,phase_time=None):\n  transformed=center+turn@(np.asarray(point)+V3_STATION_DELTA-center)\n  if phase_time is not None and phase_time<V3_TRANSLATE_TARGETS_FROM_S:transformed-=V3_STATION_DELTA\n  return transformed")
    planner='if time>=1.9:point=station_point(point);quat=station_quat(quat)'
    if source.count(planner)!=1:raise ValueError('Unsupported station waypoint transform')
    source=source.replace(planner,'if time>=1.9:point=station_point(point,time);quat=station_quat(quat)')
    path=Path(source_path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    env=dict(baseline.__globals__,V3_STATION_DELTA=np.asarray(c.station_shift_world_m,dtype=float),V3_TRANSLATE_TARGETS_FROM_S=float(c.translate_grasp_targets_from_s));exec(compile(source,str(path),'exec'),env)
    fn=env[baseline.__name__]
    for name in ['angle','closing_scale']:
        if hasattr(baseline,name):setattr(fn,name,getattr(baseline,name))
    return fn
