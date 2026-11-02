"""Independent physical thumb aspiration probe; initial held-tool reset only.

Seeds robot+pipette state by joint name from a prior fixed-base episode at 14 s.
That one reset is explicit and DOES NOT demonstrate a continuous pick-to-aspirate
chain. Runtime button motion comes only from physical thumb contact.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import ROOT, scene, arm_indices, hand_indices, ik
from t1_mobile_stage import build_mobile_scene, MobileDockController, hold_control
from run_t1_spincoat_episode import thumb_ik, smooth
from t1_aspiration_stage import AspirationStage, SourceGeometry, Stage


def build_probe(start_xy=(0., .4), source_xy=(.4, .05), seed_held_state=True):
    s, oldmobile, mobiledata, children = build_mobile_scene(start_xy=start_xy)
    from t1_visual_relocalization import add_workspace_camera
    add_workspace_camera(s)
    cup = s.worldbody.add_body(name='aspiration_source', pos=[*source_xy, 0.])
    cup.add_geom(name='aspiration_source_bottom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0, 0, .837], size=[.022, .003, 0], rgba=[.7,.8,.85,.7])
    for i in range(32):
        a=2*np.pi*i/32
        cup.add_geom(name=f'aspiration_source_wall_{i}',
                     type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     pos=[.022*np.cos(a), .022*np.sin(a), .87],
                     size=[.0023, .03, 0], rgba=[.65,.8,.85,.35])
    cup.add_geom(name='aspiration_source_liquid_visual',
                 type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0,0,.86], size=[.0195,.0198,0],
                 contype=0, conaffinity=0, rgba=[.2,.6,.85,.5])
    m=s.compile();m.vis.global_.offwidth=max(2048,m.vis.global_.offwidth)
    m.vis.global_.offheight=max(2048,m.vis.global_.offheight)
    d=mujoco.MjData(m)
    if not seed_held_state:
        d.qpos[:]=mobiledata.qpos;d.qvel[:]=mobiledata.qvel
        mujoco.mj_forward(m,d)
        return s,m,d,children,{'reset':'mobile_natural_ready_no_held_seed'}
    _, old, _, _ = scene()
    source=np.load(ROOT/'logs/spincoat/t1/task_ready_pinky_010_full_01/episode.npz')
    frame=int(np.argmin(abs(source['time']-14.)))
    for j in range(old.njnt):
        name=old.joint(j).name;nj=m.joint(name)
        nq=7 if old.jnt_type[j]==0 else 4 if old.jnt_type[j]==1 else 1
        nv=6 if old.jnt_type[j]==0 else 3 if old.jnt_type[j]==1 else 1
        d.qpos[nj.qposadr[0]:nj.qposadr[0]+nq]=source['qpos'][frame,old.jnt_qposadr[j]:old.jnt_qposadr[j]+nq]
        d.qvel[nj.dofadr[0]:nj.dofadr[0]+nv]=source['qvel'][frame,old.jnt_dofadr[j]:old.jnt_dofadr[j]+nv]
    baseqa=m.joint('chassis_free').qposadr[0]
    d.qpos[baseqa:baseqa+2]=start_xy
    pipeqa=m.joint('pipette/free').qposadr[0]
    d.qpos[pipeqa:pipeqa+2]+=np.asarray(start_xy)
    for a in range(m.nu):
        if m.actuator(a).name.startswith('astribot_chassis_motor_'):continue
        j=int(m.actuator_trnid[a,0])
        if j>=0:
            oj=old.joint(m.joint(j).name)
            d.ctrl[a]=source['desired_qpos'][frame,int(oj.qposadr[0])]
    mujoco.mj_forward(m,d)
    return s,m,d,children,{'source_episode':'task_ready_pinky_010_full_01',
        'source_time_s':float(source['time'][frame]),'copy_mode':'joint_names',
        'reset_robot_pipe_world_xy_shift_m':list(start_xy),
        'runtime_qpos_writes':False,'continuous_grasp_to_aspiration_demonstrated':False}


def build_aspiration_scene(start_xy=(0.,.4), source_xy=(.4,.05)):
    """Build mobile+static cup with natural ready reset, without held-tool seed."""
    return build_probe(start_xy, source_xy, seed_held_state=False)[:4]


def run_aspiration_on_state(m,d,*,duration=22.,source_geometry=None,provenance=None):
    """Advance a LIVE held-tool state without resetting any joint/object/time.

    Returns (arrays, result, observer). Same m/d can immediately enter wheel
    transfer. Actuated-joint bias compensation excludes freebase and wheels.
    """
    provenance=provenance or {'runtime_qpos_writes':False,'entry':'live_state'}
    started_at=float(d.time)
    ids,qa,ad=arm_indices(m);hj,hq,hd=hand_indices(m)
    desired=d.qpos.copy()
    # Preserve grip preload from live position commands; replacing these with
    # measured finger angles would relax the grasp and let the pipette slip.
    for a in range(m.nu):
        if m.actuator(a).name.startswith('astribot_chassis_motor_'):continue
        j=int(m.actuator_trnid[a,0])
        if j>=0:desired[int(m.jnt_qposadr[j])]=d.ctrl[a]
    hand=desired[hq].copy()
    armq=desired[qa].copy();dock=MobileDockController(m,[0.,.4])
    pipe=m.body('pipette/pipette').id
    palm=m.body('sharpa_right/right_hand_C_MC').id
    tip=m.site('pipette/tip_site').id
    pad=m.geom('sharpa_right/right_thumb_elastomer').id
    knob=m.body('pipette/pipette_knob').id
    kg=next(g for g in range(m.ngeom) if m.geom_bodyid[g]==knob and m.geom_contype[g])
    button=m.joint('pipette/pipette_button').qposadr[0]
    cupxy=tuple(d.xpos[m.body('aspiration_source').id,:2])
    source=source_geometry or SourceGeometry(cupxy,.0195,.84, np.pi*.0195**2*.04,.88)
    immersed_goal=np.r_[source.center_xy_m,source.initial_surface_z_m-.005]
    withdrawal_goal=np.r_[source.center_xy_m,.98]
    observer=AspirationStage(source)
    nonbase_dofs=[]
    for a in range(m.nu):
        if m.actuator(a).name.startswith('astribot_chassis_motor_'):continue
        j=m.actuator_trnid[a,0]
        if j>=0 and not m.joint(j).name.startswith(('pipette/','spincoater/')):
            nonbase_dofs.append(int(m.jnt_dofadr[j]))
    initial_tip=d.site_xpos[tip].copy();initial_thumb=d.geom_xpos[pad].copy()
    stage_started=0.;last_stage=observer.state;anchor=initial_tip.copy()
    immersion_lower_at=None;immersion_anchor=None
    rec=[];collision_events={};failure=None;ready_at=None;last_print=-1.
    for k in range(int(duration/m.opt.timestep)):
        t=float(d.time)-started_at
        if observer.state!=last_stage:
            print('STAGE',t,observer.state.value,flush=True)
            last_stage=observer.state;stage_started=t;anchor=d.site_xpos[tip].copy()
        age=t-stage_started
        if k%20==0:
            goal=initial_tip.copy();goal[:2]=source.center_xy_m
            if observer.state==Stage.IMMERSE:
                # Complete lateral centering ABOVE the cup before descending;
                # a diagonal shortcut otherwise strikes the narrow cup rim.
                if immersion_lower_at is None and np.linalg.norm(d.site_xpos[tip,:2]-np.asarray(source.center_xy_m))<.0015:
                    immersion_lower_at=t;immersion_anchor=d.site_xpos[tip].copy()
                if immersion_lower_at is not None:
                    u=smooth((t-immersion_lower_at)/3)
                    goal=(1-u)*immersion_anchor+u*immersed_goal
            elif observer.state==Stage.RELEASE:
                goal=immersed_goal.copy()
            elif observer.state in (Stage.WITHDRAW,Stage.READY):
                goal=(1-smooth(age/3))*anchor+smooth(age/3)*withdrawal_goal
            Rh=d.xmat[palm].reshape(3,3).copy();Rp=d.xmat[pipe].reshape(3,3).copy()
            current=d.site_xpos[tip].copy();axis=Rp[:,2]
            cross=np.cross(axis,[0.,0.,1.]);angle=np.arctan2(np.linalg.norm(cross),axis[2])
            rv=cross/max(np.linalg.norm(cross),1e-12)*angle
            correction=Rotation.from_rotvec(rv*min(1.,.010/max(angle,1e-12))).as_matrix()
            desired_R=correction@Rh;rel=Rh.T@(current-d.xpos[palm])
            shift=goal-current;shift*=min(1.,.0015/max(np.linalg.norm(shift),1e-12))
            handgoal=current+shift-desired_R@rel
            quat=Rotation.from_matrix(desired_R).as_quat()[[3,0,1,2]]
            armq,error=ik(m,d,handgoal,quat,seed=armq,iterations=35)
            top=d.geom_xpos[kg]+axis*.00225
            thumb_goal=top-axis*.015
            if observer.state==Stage.PRESS:
                u=smooth(t/3);thumb_goal=(1-u)*initial_thumb+u*thumb_goal
            elif observer.state==Stage.RELEASE:
                thumb_goal=top+axis*(-.015+.029*smooth(age/3))
            elif observer.state in (Stage.WITHDRAW,Stage.READY):
                thumb_goal=top+axis*.014
            hand=thumb_ik(m,d,thumb_goal,hand,hj,hq)
            desired[qa]=armq;desired[hq]=hand
        hold_control(m,d,desired);dock.step(d)
        d.qfrc_applied[:]=0
        d.qfrc_applied[nonbase_dofs]=d.qfrc_bias[nonbase_dofs]
        mujoco.mj_step(m,d)
        status=observer.update(d.time,float(d.qpos[button]),d.site_xpos[tip],allow_dispense=False)
        contacts=np.zeros(5,dtype=int)
        for ci,c in enumerate(d.contact):
            names=[m.body(m.geom_bodyid[int(g)]).name for g in (c.geom1,c.geom2)]
            if any(n.startswith('pipette/') for n in names):
                for fi,finger in enumerate(['thumb','index','middle','ring','pinky']):
                    if any(n.startswith('sharpa_right/') and finger in n for n in names):contacts[fi]+=1
            if 'aspiration_source' in names and any(n.startswith(('pipette/','sharpa_right/')) for n in names):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                if np.linalg.norm(force[:3])>.01:
                    key='|'.join(names);ev=collision_events.setdefault(key,{'count':0,'max_force_N':0.})
                    ev['count']+=1;ev['max_force_N']=max(ev['max_force_N'],float(np.linalg.norm(force[:3])))
        if k%20==0:
            rec.append((d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.site_xpos[tip].copy(),float(d.qpos[button]),contacts.copy(),observer.remaining_volume_m3,observer.state.value))
        if t-last_print>=1:
            print('SAMPLE',round(t,2),observer.state.value,'tip',d.site_xpos[tip].round(5).tolist(),'button_mm',round(float(d.qpos[button])*1000,3),'fill_uL',round(observer.remaining_volume_m3*1e9,3),'contacts',contacts.tolist(),flush=True);last_print=t
        if status['failure']:
            failure=status['failure'];break
        if not np.isfinite(d.qpos).all():failure='nonfinite_state';break
        if observer.state==Stage.READY:
            if ready_at is None:ready_at=float(d.time)
            if d.time-ready_at>3.5 and np.linalg.norm(d.site_xpos[tip]-withdrawal_goal)<.002:break
    arrays={name:np.array([r[i] for r in rec]) for i,name in enumerate(['time','qpos','qvel','ctrl','tip','button','finger_contacts','pipette_volume_m3','aspiration_stage'])}
    arrays['joint_names']=np.array([m.joint(j).name for j in range(m.njnt)])
    arrays['final_qpos']=d.qpos.copy();arrays['final_qvel']=d.qvel.copy();arrays['final_ctrl']=d.ctrl.copy()
    withdrawal_error=float(np.linalg.norm(d.site_xpos[tip]-withdrawal_goal))
    grip_min=np.min(arrays['finger_contacts'],axis=0)
    result={'success':observer.state==Stage.READY and not failure and not collision_events and withdrawal_error<.002 and bool(np.all(grip_min[1:]>0)),
            'failure':failure,'aspiration_audit':observer.audit(),'events':observer.events,
            'source_collision_events':collision_events,'initial_state_provenance':provenance,
            'final_tip_m':d.site_xpos[tip].tolist(),'duration_s':float(d.time)-started_at,
            'withdrawal_goal_m':withdrawal_goal.tolist(),'withdrawal_position_error_m':withdrawal_error,
            'finger_contacts_min':grip_min.tolist(),
            'model_nq':m.nq,'model_nv':m.nv,'model_nu':m.nu,
            'base_gravity_compensation':False,'actuated_joint_bias_compensation':True,
            'limitations':['Kinematic stroke-volume proxy; no resolved aspiration fluid.',
                'Reset seed from held-tool episode; not continuous initial pick yet.' if 'source_episode' in provenance else 'Live-state helper applies no reset; caller is responsible for upstream continuity provenance.',
                'T1 arm/torso physical collider coverage incomplete.',
                'Cup walls/liquid visual are explicit primitive prototypes.']}
    arrays['metadata_json']=json.dumps(result)
    arrays['observer_ledger_json']=json.dumps(observer.audit())
    arrays['observer_state_json']=json.dumps(observer.__dict__,default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.value if isinstance(v,Stage) else v)
    return arrays,result,observer


def run(args):
    s,m,d,children,provenance=build_probe(source_xy=args.source_xy)
    out=ROOT/'research/hardware/aspiration_stage'/args.tag
    out.mkdir(parents=True,exist_ok=False)
    (out/'scene.xml').write_text(s.to_xml())
    mujoco.mj_saveModel(m,str(out/'model.mjb'))
    arrays,result,observer=run_aspiration_on_state(m,d,duration=args.duration,provenance=provenance)
    np.savez_compressed(out/'episode.npz',**arrays)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='physical_thumb_probe_01');ap.add_argument('--duration',type=float,default=22.);ap.add_argument('--source-xy',type=float,nargs=2,default=[.4,.05]);run(ap.parse_args())
