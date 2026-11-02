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
    servo_goal_local=None;release_goal_local=None
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
                u=smooth(t/1.5);above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if t<1.5 else top+axis*(.014-.029*smooth((t-1.5)/2.5))
            elif observer.state==Stage.RELEASE:
                if release_goal_local is None:release_goal_local=Rh.T@(d.geom_xpos[pad]-d.xpos[palm])
                thumb_goal=d.xpos[palm]+Rh@release_goal_local+axis*.024*smooth(age/3)
            elif observer.state in (Stage.WITHDRAW,Stage.READY):
                thumb_goal=top+axis*.014
            if observer.state in (Stage.PRESS,Stage.IMMERSE) and (servo_goal_local is not None or d.qpos[button]<-.0035):
                if servo_goal_local is None:servo_goal_local=Rh.T@(d.geom_xpos[pad]-d.xpos[palm])
                stroke_error=-.0052-float(d.qpos[button])
                servo_goal_local+=Rh.T@axis*float(np.clip(.08*stroke_error,-.00008,.00008))
                thumb_goal=d.xpos[palm]+Rh@servo_goal_local
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
