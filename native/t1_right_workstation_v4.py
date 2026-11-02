"""RESET-only, fully table-supported RIGHT workstation candidates for v4.

Uses v3's tested reset/target translation mechanism, but rejects any stand
footprint outside the physical bench. Does not modify controller/physics gates.
Squat actuation is owned by the separate root coarse-posture adapter.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np,mujoco
from t1_right_transport_layout_v3 import RightStationLayoutConfig,relocate_station_at_reset as _relocate,translated_station_preparer as _preparer
from audit_t1_workstation_support_v4 import footprint,support

@dataclass(frozen=True)
class RightWorkstationV4:
    # Relative to original supported station center approximately (.399913,-.080021).
    station_shift_world_m: tuple=(.04,.08,0.)
    # Design constraint for reset layout; never changes any physical success gate.
    minimum_table_edge_margin_m: float=.020
    translate_grasp_targets_from_s: float=2.9
    preferred_squat_a_rad: float=.5
    # Low transport can pass behind the supported original stand after cup withdrawal.
    corridor_tip_x_world_m: float=.52

    def station_config(self):
        return RightStationLayoutConfig(station_shift_world_m=self.station_shift_world_m,translate_grasp_targets_from_s=self.translate_grasp_targets_from_s)

def audit_station_support(model,data):
    mujoco.mj_forward(model,data);g=model.geom('bench').id;c=data.geom_xpos[g];R=data.geom_xmat[g].reshape(3,3)
    if not np.allclose(R,np.eye(3),atol=1e-9):raise ValueError('Expected measured axis-aligned bench support')
    s=model.geom_size[g];bounds=[float(c[0]-s[0]),float(c[0]+s[0]),float(c[1]-s[1]),float(c[1]+s[1])]
    result=support(footprint(model,data,'stand/stand_base'),bounds);result.update(table_support_xy_bounds_m=bounds,table_top_z_m=float(c[2]+s[2]),stand_center_xy_m=data.geom_xpos[model.geom('stand/stand_base').id,:2].tolist());return result

def relocate_workstation_at_reset(scene_tuple,config=None):
    cfg=config or RightWorkstationV4();scene,report=_relocate(scene_tuple,cfg.station_config());s,m,d,children=scene;support_report=audit_station_support(m,d)
    if not support_report['whole_footprint_supported']:raise ValueError('v4 stand footprint is not wholly supported by bench')
    if min(support_report['edge_margins_m'].values())<cfg.minimum_table_edge_margin_m:raise ValueError('v4 stand footprint has less than configured table edge margin')
    report.update(v4_config=asdict(cfg),actual_reset_support_audit=support_report,scope='RESET scene layout only; no robot pose changes; not physical squat evidence');return scene,report

def translated_station_preparer(baseline,source_path,config=None):
    cfg=config or RightWorkstationV4();fn=_preparer(baseline,source_path,cfg.station_config())
    from pathlib import Path
    path=Path(source_path);source=path.read_text();anchor=' center,turn=reset_station(m,d,station_prepare.angle,out)'
    if source.count(anchor)!=1:raise ValueError('Unsupported station reset anchor')
    source=source.replace(anchor,anchor+'\n V4_VALIDATE_SUPPORTED_STATION(m,d)')
    def validate(m,d):
        audit=audit_station_support(m,d)
        if not audit['whole_footprint_supported'] or min(audit['edge_margins_m'].values())<cfg.minimum_table_edge_margin_m:raise ValueError('Final prepared station footprint fails v4 tabletop support')
        return audit
    path.write_text(source);env=dict(fn.__globals__,V4_VALIDATE_SUPPORTED_STATION=validate);exec(compile(source,str(path),'exec'),env);result=env[fn.__name__]
    for name in ['angle','closing_scale']:
        if hasattr(fn,name):setattr(result,name,getattr(fn,name))
    return result

def adapt_supported_aspiration_source(source,config=None):
    """Fixed world-X low corridor, including when the shared cup is moved."""
    from t1_right_transport_low_v3 import adapt_aspiration_source,RightTransportLowConfig,RightLowWithdrawalPolicy
    cfg=config or RightWorkstationV4();text,env,report=adapt_aspiration_source(source,RightTransportLowConfig())
    def policy_for_shared_cup(geometry,low_config):
        return RightLowWithdrawalPolicy(geometry,replace(low_config,stand_bypass_x_m=cfg.corridor_tip_x_world_m-float(geometry.center_xy_m[0])))
    env['RightLowWithdrawalPolicy']=policy_for_shared_cup
    report.pop('stand_bypass_x_m',None)
    report.update(workstation=asdict(cfg),stand_bypass_x_rule='corridor_tip_x_world_m - source.center_xy_m[0]',corridor_scope='Tip moves vertically clear of cup before lateral move to fixed world X corridor; original physics/gates unchanged')
    return text,env,report



def synchronize_source_parent_at_reset(scene_tuple,glass_config):
    """Position the complete source subtree before augmentation/compilation.

    Bimanual asset_builder already applies this same parent position. This
    helper makes the precompile scene coherent as well. Never translate the
    shared beaker child separately: its position is relative to this parent.
    """
    spec,m,d,children=scene_tuple
    if float(d.time)!=0.:raise ValueError('Source parent placement is RESET-only')
    mujoco.mj_forward(m,d);body=m.body('aspiration_source').id;parent=int(m.body_parentid[body]);old=d.xpos[body].copy();target=old.copy();target[:2]=glass_config.receiver_xy_m
    local_delta=d.xmat[parent].reshape(3,3).T@(target-old);m.body_pos[body]+=local_delta;sb=spec.body('aspiration_source');sb.pos[:]=np.asarray(sb.pos)+local_delta;mujoco.mj_forward(m,d)
    return scene_tuple,dict(scope='RESET parent-only source subtree placement before asset compile',old_source_world_m=old.tolist(),new_source_world_m=d.xpos[body].tolist(),children_separately_translated=False,robot_and_free_object_poses_unchanged=True)

def audit_shared_receiver_coherence(model,data,receiver_xy_m,prefix='prep/'):
    """Read-only actual FK check of source, nested beaker, nest and liquid XY."""
    mujoco.mj_forward(model,data);target=np.asarray(receiver_xy_m,float);centers={}
    for name in ['aspiration_source',prefix+'shared_griffin_beaker',prefix+'receiver_nest']:
        centers[name]=data.xpos[model.body(name).id,:2].copy()
    geom=prefix+'receiver_liquid_proxy_visual';centers[geom]=data.geom_xpos[model.geom(geom).id,:2].copy()
    errors={name:float(np.linalg.norm(xy-target)) for name,xy in centers.items()};ok=all(value<1e-8 for value in errors.values())
    result=dict(coherent=bool(ok),expected_receiver_xy_m=target.tolist(),actual_centers_xy_m={k:v.tolist() for k,v in centers.items()},errors_m=errors,scope='Actual compiled model FK, parent-child transforms included; no physical rollout')
    if not ok:raise ValueError('Shared source/beaker/nest/liquid center mismatch: '+str(result))
    return result
