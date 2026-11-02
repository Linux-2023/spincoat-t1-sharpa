"""Canonical complete v6 RESET scene, all placement compiled into fresh BVHs.
For actual independent LEFT/RIGHT tests, not a stored-held-state continuation.
"""
from pathlib import Path
import json,os
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import ROOT
from t1_tabletop_decor_reset_v6 import apply_decor_to_spec,verify_compiled_decor
from t1_right_interior_workstation_v6 import RightInteriorWorkstationV6,relocate_coater_at_reset
from t1_reagent_workflow_adapter import augment_preparation_scene
from t1_bimanual_glass_preparation import BimanualGlassConfig,add_reagent_preparation

DEFAULT_DECOR=ROOT/'research/hardware/tabletop_layout_v6/complete_unified_06/unified_layout_candidate.json'

def _set_spec_body_world(spec,model,data,name,world_p,world_R):
    b=model.body(name).id;parent=int(model.body_parentid[b]);R=data.xmat[parent].reshape(3,3);local_p=R.T@(world_p-data.xpos[parent]);local_R=R.T@world_R
    body=spec.body(name);frame=body.frame
    if frame is not None:
        if frame.frame is not None:raise ValueError('Nested attachment frame not reviewed')
        F=Rotation.from_quat(np.asarray(frame.quat)[[1,2,3,0]]).as_matrix();local_p=F.T@(local_p-np.asarray(frame.pos));local_R=F.T@local_R
    body.pos[:]=local_p;body.quat[:]=Rotation.from_matrix(local_R).as_quat()[[3,0,1,2]]

def place_stand_and_unheld_pipe(spec,model,data,center_xy=(.68,0.),yaw_deg=60.):
    if float(data.time)!=0.:raise ValueError('Stand and unheld pipe are RESET-only')
    mujoco.mj_forward(model,data);bid=model.body('stand/pipette_stand').id;gid=model.geom('stand/stand_base').id
    old_p=data.xpos[bid].copy();old_R=data.xmat[bid].reshape(3,3).copy();new_R=Rotation.from_euler('z',yaw_deg,degrees=True).as_matrix();offset=old_R.T@(data.geom_xpos[gid]-old_p);target=data.geom_xpos[gid].copy();target[:2]=center_xy;new_p=target-new_R@offset
    pb=model.body('pipette/pipette').id;turn=new_R@old_R.T;pipe_p=new_p+turn@(data.xpos[pb]-old_p);pipe_R=turn@data.xmat[pb].reshape(3,3)
    _set_spec_body_world(spec,model,data,'stand/pipette_stand',new_p,new_R)
    _set_spec_body_world(spec,model,data,'pipette/pipette',pipe_p,pipe_R)
    j=int(model.body_jntadr[pb]);qa=int(model.jnt_qposadr[j]);assert model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE
    data.qpos[qa:qa+3]=pipe_p;data.qpos[qa+3:qa+7]=Rotation.from_matrix(pipe_R).as_quat()[[3,0,1,2]]
    return dict(center_world_xy_m=list(center_xy),stand_yaw_deg=yaw_deg,free_pipe_reset_pose_written=True,no_live_pose_writes=True)

def build_scene(out,*,cup_xy=(.60,.30),tube_xy=(.60,.50),stand_xy=(.68,0.),stand_yaw=60.,coater_xy=(.55,-.34),decor_proposal=DEFAULT_DECOR,appearance=True,settled_tube=True):
    os.environ['T1_LAB_CONTEXT']='1';os.environ['T1_CHEMIST_COAT']='0'
    from run_t1_aspiration_probe import build_aspiration_scene
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec,model,data,children=build_aspiration_scene(source_xy=cup_xy)
    decor=apply_decor_to_spec(spec,model,data,decor_proposal)
    cfg=RightInteriorWorkstationV6(cup_xy_m=tuple(cup_xy),tube_xy_m=tuple(tube_xy),stand_center_xy_m=tuple(stand_xy),coating_target_xy_m=tuple(coater_xy),standing_torso_q_rad=(0.,0.,np.radians(12.),0.))
    scene,coater=relocate_coater_at_reset((spec,model,data,children),cfg)
    stand=place_stand_and_unheld_pipe(spec,model,data,stand_xy,stand_yaw)
    old=ROOT/'research/hardware/human_workflow_v5/integrated_human_v5_04_geometry/preparation_config.json';config=json.loads(old.read_text());config.update(tube_xy_m=list(tube_xy),receiver_xy_m=list(cup_xy))
    glass=BimanualGlassConfig(**config)
    scene,handles=augment_preparation_scene((spec,model,data,children),glass,asset_builder=add_reagent_preparation)
    spec,model,data,children=scene
    ready=json.loads((ROOT/'research/hardware/task_ready_natural_01_pinky_010mm.json').read_text())
    for name,value in ready['initial_joint_positions'].items():data.qpos[model.joint(name).qposadr[0]]=value
    if appearance:
        from preview_lab_materials_v2 import appearance_assets,lighting
        appdir=(out/'appearance').resolve();appdir.mkdir(exist_ok=True);appearance_assets(spec,appdir);old_data=data;model=spec.compile();data=mujoco.MjData(model);data.qpos[:]=old_data.qpos;data.qvel[:]=old_data.qvel;data.ctrl[:]=old_data.ctrl;lighting(model,True,neutral=True)
    model.opt.noslip_iterations=10;model.vis.global_.offwidth=2048;model.vis.global_.offheight=2048;mujoco.mj_forward(model,data)
    if settled_tube:
        from settle_t1_glass_reset import initialize_naturally_settled_tube
        initialize_naturally_settled_tube(model,data,handles,out/'reagent_preparation',initial_tilt_deg=.3,tilt_axis_world=(1.,0.,0.))
    verification=verify_compiled_decor(model,data,decor)
    base=model.geom('stand/stand_base').id;target=model.site('spincoater/dispense_target').id
    assert np.allclose(data.geom_xpos[base,:2],stand_xy,atol=1e-8)
    assert np.allclose(data.site_xpos[target,:2],coater_xy,atol=1e-8)
    report=dict(scope=__doc__,cup_xy_m=list(cup_xy),tube_xy_m=list(tube_xy),stand=stand,coater=coater,decor=decor,decor_compile_verification=verification,physics_scope='Existing scene defaults; all geometry changes occur before the final compile; no collision masks, gains or success gates changed')
    (out/'canonical_scene_v6.json').write_text(json.dumps(report,indent=2));(out/'preparation_config.json').write_text(json.dumps(config,indent=2))
    return (spec,model,data,children),handles,report
