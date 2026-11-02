"""Opt-in v6 interior workstation proposal; RESET-only scene changes.

Reach candidates are conditional geometry, never an actual grasp certificate.
Original v3/v4/v5 defaults, physical parameters and acceptance gates stay intact.
"""
from dataclasses import dataclass,asdict
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_right_workstation_v4 import RightWorkstationV4,relocate_workstation_at_reset
from t1_right_transport_low_v3 import RightTransportLowConfig,adapt_aspiration_source as _low

@dataclass(frozen=True)
class RightInteriorWorkstationV6:
    cup_xy_m:tuple=(.60,.30)
    tube_xy_m:tuple=(.60,.50)
    stand_center_xy_m:tuple=(.68,0.)
    grasp_verticalize_extra_lift_m:float=0.
    skip_camera_clearance_motion:bool=True
    stand_yaw_deg:float=60.
    station_already_prepared:bool=False
    coating_target_xy_m:tuple=(.55,-.34)
    grasp_base_xy_m:tuple=(.02,.20)
    aspiration_base_xy_m:tuple=(0.,.30)
    coating_base_xy_m:tuple=(0.,-.10)
    standing_torso_q_rad:tuple=(0.,0.,float(np.pi/15),0.)
    aspiration_tool_world_yaw_delta_deg:float=60.
    minimum_table_edge_margin_m:float=.050

    def supported_station_config(self):
        turn=Rotation.from_euler('z',self.stand_yaw_deg-60.,degrees=True).as_matrix()[:2,:2]
        original=np.array([.4,-.08])+turn@np.array([-.0000871656817521,-.000020507134015])
        delta=np.r_[np.asarray(self.stand_center_xy_m)-original,0.]
        return RightWorkstationV4(station_shift_world_m=tuple(delta),minimum_table_edge_margin_m=self.minimum_table_edge_margin_m)

def relocate_supported_station_at_reset(scene_tuple,config=None):
    c=config or RightInteriorWorkstationV6()
    return relocate_workstation_at_reset(scene_tuple,c.supported_station_config())

def relocate_coater_at_reset(scene_tuple,config=None):
    """Translate complete coater subtree to requested actual dispense-site XY.

    Handles the original coater's Rz90 attachment frame explicitly. Changes
    model/spec geometry placement only at RESET; leaves robot/free-object qpos.
    """
    c=config or RightInteriorWorkstationV6();spec,m,d,children=scene_tuple
    if float(d.time)!=0.:raise ValueError('Coater relocation is RESET-only')
    mujoco.mj_forward(m,d);sid=m.site('spincoater/dispense_target').id;old=d.site_xpos[sid].copy();delta=np.r_[np.asarray(c.coating_target_xy_m)-old[:2],0.]
    b=m.body('spincoater/spincoater').id;parent=int(m.body_parentid[b]);parent_delta=d.xmat[parent].reshape(3,3).T@delta;sb=spec.body('spincoater/spincoater');frame=sb.frame
    # MjSpec retains attachment-frame rotation; compiled MjModel folds it into
    # body_pos/body_quat. Apply the same WORLD translation in both coordinates.
    if frame is None:spec_delta=parent_delta
    else:
        if frame.frame is not None:raise ValueError('Nested coater attachment frames require explicit transform audit')
        fq=np.asarray(frame.quat);spec_delta=Rotation.from_quat(fq[[1,2,3,0]]).as_matrix().T@parent_delta
    q=d.qpos.copy();mass=m.body_mass.copy();inertia=m.body_inertia.copy();m.body_pos[b]+=parent_delta;sb.pos[:]=np.asarray(sb.pos)+spec_delta;mujoco.mj_forward(m,d)
    if not np.array_equal(q,d.qpos):raise AssertionError('RESET scene operation changed qpos')
    if not np.array_equal(mass,m.body_mass) or not np.array_equal(inertia,m.body_inertia):raise AssertionError('Coater relocation altered body dynamics')
    if not np.allclose(d.site_xpos[sid,:2],c.coating_target_xy_m,atol=1e-10):raise AssertionError('Coater target mismatch')
    return scene_tuple,dict(scope='RESET-only complete coater placement, not an actual manipulation test',config=asdict(c),old_dispense_target_world_m=old.tolist(),new_dispense_target_world_m=d.site_xpos[sid].tolist(),world_translation_m=delta.tolist(),spec_body_local_translation_m=spec_delta.tolist(),robot_and_free_object_qpos_unchanged=True,mass_inertia_unchanged=True,station_origin_world_m=d.xpos[b].tolist())

class VerticalCupWithdrawalV6:
    """Predeclared vertical cup exit; no stand detour or contact-triggered phase."""
    def __init__(self,source,config=None):
        c=config or RightTransportLowConfig();self.final_goal=np.r_[source.center_xy_m,max(c.receiver_rim_z_m+c.tip_clearance_over_rim_m,source.initial_surface_z_m+c.minimum_dry_clearance_m)]
    def target(self,time,age,anchor,actual_tip):
        u=float(np.clip(age/3.,0.,1.));u=u*u*u*(10+u*(-15+6*u));return (1-u)*np.asarray(anchor)+u*self.final_goal

def adapt_vertical_aspiration_source(source,config=None):
    """Private original aspiration source adapter, original .98m anchors strict."""
    c=RightTransportLowConfig(stand_bypass_x_m=0.)
    source,env,report=_low(source,c);env['RightLowWithdrawalPolicy']=VerticalCupWithdrawalV6
    env['ik']=make_validated_aspiration_ik_v6()
    report['ik_method']='Existing validated .04rad margin IK; unchanged smooth-nullspace3mrad/call and .04rad iteration trust clipping'
    report.update(policy='v6 vertical-only cup withdrawal then original coarse transfer',no_world_x_bypass=True,contact_or_force_policy_input=False,receiver_config=asdict(config or RightInteriorWorkstationV6()))
    return source,env,report


def make_validated_aspiration_ik_v6():
    from t1_spincoat_scene import arm_indices
    from t1_workflow_ik import install_validated_ik
    import types
    solvers={}
    def ik(model,data,position,quaternion,**kwargs):
        key=id(model)
        if key not in solvers:
            runner=types.SimpleNamespace(arm_indices=arm_indices,workflow_nullspace_enabled=True);install_validated_ik(runner,float(data.time));solvers[key]=(model,runner)
        return solvers[key][1].ik(model,data,position,quaternion,**kwargs)
    return ik
