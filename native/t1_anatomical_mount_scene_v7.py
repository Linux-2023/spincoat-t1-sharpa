"""Opt-in RESET-only anatomical mirror convention; provisional, NOT hardware-calibrated."""
from pathlib import Path
import json,hashlib,numpy as np,mujoco
from t1_interior_scene_v6 import build_scene as build_v6_scene
ROOT=Path(__file__).resolve().parents[1]
PROOF=ROOT/'research/hardware/human_workflow_v7/independent_mount_symmetry_01/mirror_algebra_proof.json'
LEFT='sharpa_left/left_hand_C_MC';RIGHT='sharpa_right/right_hand_C_MC';QUAT=np.array([0.,0.,0.,1.])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_scene(out,*,cup_xy=(.60,.30),tube_xy=(.60,.50),stand_xy=(.68,0.),stand_yaw=60.,coater_xy=(.55,-.34),decor_proposal=None,appearance=True,settled_tube=True,anatomical_ready=True):
    """Build a NEW scene at RESET. Never pass an existing live model/data here."""
    args=dict(cup_xy=cup_xy,tube_xy=tube_xy,stand_xy=stand_xy,stand_yaw=stand_yaw,coater_xy=coater_xy,appearance=appearance,settled_tube=False)
    if decor_proposal is not None:args['decor_proposal']=decor_proposal
    scene,handles,base_report=build_v6_scene(out,**args);spec,old,old_data,children=scene
    lid=old.body(LEFT).id;rid=old.body(RIGHT).id
    assert old.body(int(old.body_parentid[lid])).name=='astribot_arm_left_tool_link'
    assert np.allclose(old.body_pos[lid],0.,atol=1e-12) and np.allclose(old.body_quat[lid],[1.,0.,0.,0.],atol=1e-12),'Refuse double application or unexpected source mount'
    body=spec.body(LEFT);body.quat[:]=QUAT
    model=spec.compile()
    assert (old.nq,old.nv,old.nu)==(model.nq,model.nv,model.nu)
    assert [old.joint(i).name for i in range(old.njnt)]==[model.joint(i).name for i in range(model.njnt)]
    assert [old.actuator(i).name for i in range(old.nu)]==[model.actuator(i).name for i in range(model.nu)]
    # Preserve all postcompile v6 option/visual settings. Do not copy derived model constants.
    for attr in dir(old.opt):
        if attr.startswith('_'):continue
        v=getattr(old.opt,attr)
        if callable(v):continue
        try:
            if isinstance(v,np.ndarray):getattr(model.opt,attr)[:]=v
            elif isinstance(v,(int,float,bool,np.integer,np.floating)):setattr(model.opt,attr,v)
        except (AttributeError,TypeError):pass
    # v6 appearance applies model-only presentation arrays after compile.
    for attr in ['geom_rgba','mat_rgba','mat_emission','mat_specular','mat_shininess','mat_reflectance','light_ambient','light_diffuse','light_specular']:
        if hasattr(old,attr):getattr(model,attr)[:]=getattr(old,attr)
    model.vis.global_.offwidth=old.vis.global_.offwidth;model.vis.global_.offheight=old.vis.global_.offheight
    for attr in ['ambient','diffuse','specular']:getattr(model.vis.headlight,attr)[:]=getattr(old.vis.headlight,attr)
    data=mujoco.MjData(model);data.qpos[:]=old_data.qpos;data.qvel[:]=old_data.qvel;data.ctrl[:]=old_data.ctrl;data.time=old_data.time
    if model.nmocap:data.mocap_pos[:]=old_data.mocap_pos;data.mocap_quat[:]=old_data.mocap_quat
    initial_before_override=data.qpos.copy();ready_override={}
    if anatomical_ready:
        mirror_sign=np.array([-1.,1.,-1.,1.,-1.,1.,-1.])
        for i,sgn in enumerate(mirror_sign,1):
            lj=model.joint('astribot_arm_left_joint_'+str(i));rj=model.joint('astribot_arm_right_joint_'+str(i));value=sgn*float(data.qpos[rj.qposadr[0]])
            assert model.jnt_range[lj.id,0]<=value<=model.jnt_range[lj.id,1]
            data.qpos[lj.qposadr[0]]=value;ready_override[lj.name]=value
    overridden_ids=[model.joint(n).qposadr[0] for n in ready_override]
    untouched=np.ones(model.nq,dtype=bool);untouched[overridden_ids]=False
    assert np.array_equal(data.qpos[untouched],initial_before_override[untouched])
    mujoco.mj_forward(model,data)
    assert np.allclose(model.body_quat[model.body(LEFT).id],QUAT,atol=1e-12)
    assert np.array_equal(model.body_quat[model.body(RIGHT).id],old.body_quat[rid])
    unchanged=['body_mass','body_inertia','jnt_range','jnt_axis','jnt_limited','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange']
    checks={n:bool(np.array_equal(getattr(old,n),getattr(model,n))) for n in unchanged}
    assert all(checks.values()),checks
    if settled_tube:
        from settle_t1_glass_reset import initialize_naturally_settled_tube
        initialize_naturally_settled_tube(model,data,handles,Path(out)/'reagent_preparation',initial_tilt_deg=.3,tilt_axis_world=(1.,0.,0.))
    report=dict(base_report,scope=__doc__,mount_calibrated=False,mount_convention='LEFT local tool-Z pi rotation to anatomically mirror unchanged RIGHT; not measured hardware adapter',left_parent='astribot_arm_left_tool_link',left_child=LEFT,left_tool_to_palm_quat_wxyz=QUAT.tolist(),left_tool_to_palm_pos_m=model.body_pos[model.body(LEFT).id].tolist(),right_mount_unchanged=True,model_arrays_preserved=checks,source_builder_sha256=sha(Path(__file__)),mirror_proof_sha256=sha(PROOF),anatomical_ready=bool(anatomical_ready),initial_joint_override=ready_override,all_other_initial_qpos_exactly_preserved_before_reset_settle=True,ready_override_scope='Seven LEFT arm qpos at NEW RESET only; RIGHT, torso, head, fingers and freeobjects unchanged',settled_tube_at_new_scene_reset_only=bool(settled_tube),model_recompiled_with_fresh_derived_constants=True,old_record_or_live_state_reused=False)
    (Path(out)/'anatomical_mount_scene_v7.json').write_text(json.dumps(report,indent=2))
    return (spec,model,data,children),handles,report
