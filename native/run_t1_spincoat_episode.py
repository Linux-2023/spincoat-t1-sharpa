"""Scripted T1/Sharpa physical task integration, separate from VLA evaluation."""
import argparse,json,os,time,hashlib,shutil
os.environ.setdefault('MUJOCO_GL','egl')
from dataclasses import asdict
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from t1_spincoat_scene import *


def smooth(u):
    u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))


def thumb_ik(m,d,goal,seed,hj,hq):
    mask=np.array([k for k,j in enumerate(hj) if 'thumb' in m.joint(j).name])
    qids=hq[mask];ids=hj[mask]
    work=mujoco.MjData(m);work.qpos[:]=d.qpos
    pad=m.geom('sharpa_right/right_thumb_elastomer').id
    lo,hi=m.jnt_range[ids].T
    initial=np.clip(seed[mask],lo+1e-6,hi-1e-6)
    def residual(q):
        work.qpos[qids]=q;mujoco.mj_kinematics(m,work)
        return np.r_[(work.geom_xpos[pad]-goal)*100,.003*(q-initial)]
    r=least_squares(residual,initial,bounds=(lo+1e-7,hi-1e-7),max_nfev=35)
    result=seed.copy();result[mask]=r.x
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='t1_sharpa_full_01')
    parser.add_argument('--max-seconds',type=float,default=36.)
    parser.add_argument('--from-home',action='store_true')
    parser.add_argument('--motion-profile',type=Path,help='Validated home-to-grasp joint waypoints with C2 time scaling')
    parser.add_argument('--lab-context',action='store_true')
    parser.add_argument('--chemist-coat',action='store_true')
    parser.add_argument('--close-pinky-before-lift',action='store_true')
    args=parser.parse_args()
    if args.lab_context:os.environ['T1_LAB_CONTEXT']='1'
    if args.chemist_coat:os.environ['T1_CHEMIST_COAT']='1'
    profile=json.loads(args.motion_profile.read_text()) if args.motion_profile else None
    if profile is not None:args.from_home=True
    out=ROOT/'logs/spincoat/t1'/args.tag
    out.mkdir(parents=True,exist_ok=False)
    s,m,d,children=scene()
    collision_coverage={prefix:sum(bool(m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith(prefix) for g in range(m.ngeom)) for prefix in ['astribot_arm_','astribot_torso','sharpa_right/','sharpa_left/']}
    (out/'scene.xml').write_text(s.to_xml())
    snapshot=out/'source';snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    if profile is not None:(out/'motion_profile.json').write_text(json.dumps(profile,indent=2))
    fit=json.loads((ROOT/'research/hardware/grasp/grasp_fit.json').read_text())
    ids,qa,dv=arm_indices(m);hj,hq,hd=hand_indices(m)
    close=np.array([fit['joint_targets'][m.joint(j).name.split('/')[1]] for j in hj])
    for k,j in enumerate(hj):
        if 'thumb' not in m.joint(j).name and any(name in m.joint(j).name for name in ('MCP_FE','_DIP')):
            close[k]=min(close[k]+.12,m.jnt_range[j,1])
        if 'thumb' in m.joint(j).name:
            close[k]=fit['thumb_hover'][m.joint(j).name.split('/')[1]]
    openq=close*.8
    for k,j in enumerate(hj):
        if 'MCP_FE' in m.joint(j).name and 'thumb' not in m.joint(j).name:openq[k]=0.
        if 'thumb' in m.joint(j).name:openq[k]=close[k]
    pinky=np.array(['pinky' in m.joint(j).name for j in hj])
    base=PIPETTE_START+GRASP_WRIST_OFFSET
    seed=np.load(ROOT/'research/hardware/grasp/probe.npz')['qpos'][0][qa]
    q,error=ik(m,d,base,HAND_QUAT,seed=seed,iterations=400);assert error<.001,error
    home_arm=d.qpos[qa].copy();home_hand=d.qpos[hq].copy()
    prefix=float(profile['duration_s']) if profile else 13. if args.from_home else 0.
    home_plan=[]
    if profile:
        assert profile['joint_names']==[m.joint(j).name for j in ids]
        assert profile['hand_profile']['joint_names']==[m.joint(j).name for j in hj]
        frames=[(float(w['time_s']),np.asarray(w['arm_q_rad'])) for w in profile['waypoints']]
        assert np.allclose(frames[0][1],home_arm)
        assert np.allclose(profile['hand_profile']['home_q_rad'],home_hand)
        assert np.allclose(profile['hand_profile']['preshape_q_rad'],openq)
    elif args.from_home:
        previous=seed
        for point in ([.08,-.36,1.10],base+[-.06,0,.12],base+[-.06,0,0],base):
            waypoint,error=ik(m,d,np.asarray(point),HAND_QUAT,seed=previous,iterations=400)
            assert error<.001,(point,error)
            home_plan.append(waypoint);previous=waypoint
    else:
        d.qpos[qa]=q;d.qpos[hq]=openq
    mujoco.mj_forward(m,d)
    desired=d.qpos.copy();desired_hand=close.copy()
    robot_dofs=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if not m.joint(j).name.startswith(('pipette/','spincoater/'))])
    act_q=m.jnt_qposadr[m.actuator_trnid[:,0]]
    for a in range(4,m.nu):d.ctrl[a]=desired[act_q[a]]
    pipe=m.body('pipette/pipette').id;palm=m.body('sharpa_right/right_hand_C_MC').id
    tip=m.site('pipette/tip_site').id;targetid=m.site('spincoater/dispense_target').id
    knob=m.body('pipette/pipette_knob').id
    knob_geom=next(g for g in range(m.ngeom) if m.geom_bodyid[g]==knob and m.geom_contype[g])
    button_q=m.joint('pipette/pipette_button').qposadr[0]
    rotor=m.joint('spincoater/rotor');rotorq=rotor.qposadr[0];rotord=rotor.dofadr[0]
    F=fluid_module('fluid_v2');fluid=F.SpincoatFluid(F.FluidParameters())
    target=d.site_xpos[targetid].copy()
    pad=m.geom('sharpa_right/right_thumb_elastomer').id
    launched=False;landed=False;motor_at=None;release=None;landing_at=None
    valid=False;impact_error=None;contact_violation=False
    environment_contacts={}
    records=[];events=[];last_phase=None;align_anchor=None;transit_anchor=None;lower_anchor=None
    start_thumb=None;retract_at=None;retract_anchor=None
    omega=0.;start_wall=time.monotonic();last_print=-2
    station_yaw=2*np.arctan2(STATION_QUAT[3],STATION_QUAT[0])
    phases=['close','hold','lift','verticalize','translate','lower','settle','thumb_above','press','retract','spin']
    failure_reason=None
    for step in range(round((args.max_seconds+prefix)/.001)):
        t=step*.001
        task_t=t-prefix
        if t<prefix:
            if profile:
                segment=next((i for i in range(len(frames)-1) if t<frames[i+1][0]),len(frames)-2)
                phase='home' if segment==0 else profile['waypoints'][segment+1].get('name','approach_pipette')
            else:phase='home' if t<1 else 'raise_from_home' if t<4 else 'reach_pregrasp' if t<7 else 'lower_pregrasp' if t<10 else 'approach_pipette'
        elif task_t<2:phase='close'
        elif task_t<3:phase='hold'
        elif task_t<6:phase='lift'
        elif task_t<9:phase='verticalize'
        elif task_t<13:phase='translate'
        elif task_t<16:phase='lower'
        elif task_t<18:phase='settle'
        elif task_t<21:phase='thumb_above'
        else:phase='press'
        if launched:phase='retract' if motor_at is None else 'spin'
        if phase!=last_phase:
            events.append({'time_s':t,'phase':phase});print('PHASE',t,phase,flush=True)
            if phase=='verticalize':align_anchor=d.site_xpos[tip].copy()
            if phase=='translate':transit_anchor=d.site_xpos[tip].copy()
            if phase=='lower':lower_anchor=d.site_xpos[tip].copy()
            if phase=='thumb_above':start_thumb=d.geom_xpos[pad].copy()
            last_phase=phase
        if step%20==0:
            if t<prefix:
                if not profile:frames=[(0.,home_arm),(1.,home_arm),(4.,home_plan[0]),(7.,home_plan[1]),(10.,home_plan[2]),(12.,home_plan[3]),(13.,home_plan[3])]
                for (ta,pa),(tb,pb) in zip(frames[:-1],frames[1:]):
                    if ta<=t<=tb:
                        u=smooth((t-ta)/(tb-ta));q=(1-u)*pa+u*pb;break
                hand_start,hand_end=(profile['hand_profile']['start_time_s'],profile['hand_profile']['end_time_s']) if profile else (4.,7.)
                u=smooth((t-hand_start)/(hand_end-hand_start));desired_hand=(1-u)*home_hand+u*openq
                error=0.
            elif phase in ('close','hold','lift'):
                u=smooth(task_t/2);desired_hand=(1-u)*openq+u*close
                # Three fingers establish the grasp. Curl the little finger
                # only after the pipette has cleared the support fork.
                if not args.close_pinky_before_lift:
                    up=smooth((task_t-4.5)/1.5)
                    desired_hand[pinky]=(1-up)*openq[pinky]+up*close[pinky]
                goal=base.copy();goal[2]+=.08*smooth((task_t-3)/3)
                q,error=ik(m,d,goal,HAND_QUAT,seed=desired[qa],iterations=45)
            else:
                Rh=d.xmat[palm].reshape(3,3).copy();Rp=d.xmat[pipe].reshape(3,3).copy()
                current_tip=d.site_xpos[tip].copy()
                goal_tip=target+np.array([.0001,0,.012])
                if phase=='verticalize':
                    goal_tip=align_anchor.copy();goal_tip[2]+=.03*smooth((task_t-6)/3)
                elif phase=='translate':goal_tip=(1-smooth((task_t-9)/4))*transit_anchor+smooth((task_t-9)/4)*(target+[.0001,0,.08])
                elif phase=='lower':goal_tip=(1-smooth((task_t-13)/3))*lower_anchor+smooth((task_t-13)/3)*(target+[.0001,0,.012])
                elif phase in ('retract','spin'):
                    if retract_at is None:retract_at=t;retract_anchor=current_tip.copy()
                    goal_tip=retract_anchor+np.array([0,0,.10*smooth((t-retract_at)/2)])
                # Slowly correct measured pipe orientation and nozzle position.
                axis=Rp[:,2];cross=np.cross(axis,np.array([0.,0.,1.]))
                tilt=np.arctan2(np.linalg.norm(cross),axis[2])
                rotation_error=cross/max(np.linalg.norm(cross),1e-12)*tilt
                angle=np.linalg.norm(rotation_error)
                correction=Rotation.from_rotvec(rotation_error*min(1,.012/max(angle,1e-12))).as_matrix()
                desired_R=correction@Rh
                rel=Rh.T@(current_tip-d.xpos[palm])
                shift=goal_tip-current_tip
                shift*=min(1,.0025/max(np.linalg.norm(shift),1e-12))
                hand_goal=current_tip+shift-desired_R@rel
                quat=Rotation.from_matrix(desired_R).as_quat()[[3,0,1,2]]
                q,error=ik(m,d,hand_goal,quat,seed=desired[qa],iterations=35)
                if phase in ('thumb_above','press','retract','spin'):
                    axis=d.xmat[pipe].reshape(3,3)[:,2]
                    top=d.geom_xpos[knob_geom]+axis*.00225
                    thumb_goal=top+axis*.010
                    if phase=='thumb_above':
                        u=smooth((task_t-18)/3)
                        thumb_goal=(1-u)*start_thumb+u*thumb_goal
                    elif phase=='press':thumb_goal=top+axis*(.010-.025*smooth((task_t-21)/3))
                    else:thumb_goal=top+axis*.014
                    desired_hand=thumb_ik(m,d,thumb_goal,desired_hand,hj,hq)
            desired[qa]=q;desired[hq]=desired_hand
            for a in range(4,m.nu):d.ctrl[a]=desired[act_q[a]]
        d.qfrc_applied[:]=0;d.qfrc_applied[robot_dofs]=d.qfrc_bias[robot_dofs]
        if motor_at is not None:omega=20*np.pi*min(1,(t-motor_at)/.5)
        d.qvel[rotord]=omega
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all():failure_reason='nonfinite_state';break
        for contact_index,c in enumerate(d.contact):
            bnames=[m.body(m.geom_bodyid[int(g)]).name for g in (c.geom1,c.geom2)]
            obstacle=next((n for n in bnames if n.startswith(('lab_bench','stand/','spincoater/'))),None)
            moving=any(n and not n.startswith(('lab_bench','stand/','spincoater/','pipette/')) for n in bnames)
            if obstacle and (moving or (not obstacle.startswith('stand/') and any(n.startswith('pipette/') for n in bnames))):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,contact_index,force)
                magnitude=float(np.linalg.norm(force[:3]))
                if magnitude>.001:
                    contact_violation=True
                    group=obstacle.split('/')[0]
                    event=environment_contacts.setdefault(group,{'samples':0,'first_s':t,'max_force_N':0.,'max_penetration_m':0.,'bodies':bnames})
                    event['samples']+=1;event['last_s']=t
                    event['max_force_N']=max(event['max_force_N'],magnitude)
                    event['max_penetration_m']=max(event['max_penetration_m'],-float(c.dist))
        if not launched and d.qpos[button_q]<-.005:
            launched=True;release=t+.001;pos=d.site_xpos[tip].copy()
            vel6=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,tip,vel6,0);vel=vel6[3:]
            dz=pos[2]-target[2]
            flight=(vel[2]+np.sqrt(max(vel[2]**2+2*9.81*dz,0)))/9.81
            impact=pos+vel*flight;impact[2]-=.5*9.81*flight**2
            impact_error=float(np.linalg.norm(impact[:2]-target[:2]))
            valid=phase=='press' and .003<=dz<=.03 and np.linalg.norm(vel)<.05 and not contact_violation
            landing_at=release+flight
            release_state={'position_m':pos.tolist(),'velocity_m_s':vel.tolist(),'flight_s':flight,'impact_m':impact.tolist(),'error_mm':impact_error*1000,'valid':bool(valid)}
            print('RELEASE',json.dumps(release_state),flush=True)
            if not valid:failure_reason='invalid_or_premature_dispense';break
        if launched and not landed and t+.001>=landing_at:
            landed=True
            angle=d.qpos[rotorq]+station_yaw;x,y=impact[:2]-target[:2]
            fluid.deposit((np.cos(angle)*x+np.sin(angle)*y,-np.sin(angle)*x+np.cos(angle)*y))
        if launched and motor_at is None and landed and d.site_xpos[tip,2]-target[2]>.04:
            motor_at=t;events.append({'time_s':t,'phase':'motor_start'})
        if step%20==0:
            if landed:
                fluid_dt=max(0.,t+.001-landing_at-fluid.time_s)
                fluid.advance(fluid_dt,omega,rotor_angle_start=station_yaw+d.qpos[rotorq]-omega*fluid_dt)
            contacts=0;finger_contacts=np.zeros(5,dtype=int);finger_forces=np.zeros(5)
            for ci,c in enumerate(d.contact):
                names=[m.body(m.geom_bodyid[int(g)]).name for g in (c.geom1,c.geom2)]
                if any(n.startswith('pipette/') for n in names) and any(n.startswith('sharpa_right/') for n in names):
                    contacts+=1
                    hand_name=next(n for n in names if n.startswith('sharpa_right/'))
                    for fi,finger in enumerate(['thumb','index','middle','ring','pinky']):
                        if finger in hand_name:
                            force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                            finger_contacts[fi]+=1;finger_forces[fi]+=np.linalg.norm(force[:3])
            records.append({'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time':t,'phase':phase,
                'tip':d.site_xpos[tip].copy(),'palm':d.xpos[palm].copy(),'button':float(d.qpos[button_q]),
                'grip_contacts':contacts,'finger_contacts':finger_contacts,'finger_forces_N':finger_forces,'film':fluid.h.copy(),'fluid_time':fluid.time_s,
                'desired_qpos':desired.copy(),'ik_error_m':error})
            if t-last_print>=1.99:
                print('STATE',round(t,2),phase,'tip',d.site_xpos[tip].round(4).tolist(),'button_mm',round(d.qpos[button_q]*1000,2),'contacts',contacts,'IK_mm',round(error*1000,2),flush=True);last_print=t
        if motor_at is not None and t-motor_at>=4.2:break
        if d.xpos[pipe,2]<.76:failure_reason='pipette_dropped';break
    success=bool(valid and impact_error<=.0007 and landed and motor_at is not None and t-motor_at>=4. and not contact_violation and failure_reason is None)
    metadata={'task':'T1-Sharpa-SpinCoat-Integration','fluid_model':'v2','film_parameters':asdict(fluid.p),
        'step_dt':.02,'simulation_dt':.001,'controller':'privileged-state Cartesian IK + named finger joints, gravity compensation',
        'fixed_base':True,'mount_calibrated':False,'no_weld':True,'no_object_pose_writes_after_reset':True,
        'starts_from_home':args.from_home,'home_approach_duration_s':prefix,
        'motion_profile':profile,
        'pinky_closes_before_lift':args.close_pinky_before_lift,'grasp_wrist_offset_m':GRASP_WRIST_OFFSET.tolist(),
        'lab_context_enabled':os.environ.get('T1_LAB_CONTEXT')=='1','chemist_coat_enabled':os.environ.get('T1_CHEMIST_COAT')=='1',
        'bench_dimensions_m':[2.,.75,BENCH_HEIGHT],'bench_origin':BENCH_ORIGIN.tolist(),
        'pipette_start':PIPETTE_START.tolist(),'station_origin':STATION.tolist(),'station_quaternion_wxyz':STATION_QUAT.tolist(),
        'hand_PD_velocity_sign_corrected':True,'button_limit_time_constant_s':.002,
        'hand_PD':'MCP/CMC kp20, other kp10, kd0.6; illustrative torque caps 1.5/0.5 Nm, uncalibrated',
        'policy_uses_tactile':False,'dispense_mode':'static','spin_rpm':600,'release':release_state if launched else None,
        'physical_collision_geom_counts':collision_coverage,
        'full_t1_arm_collision_coverage':collision_coverage['astribot_arm_']>0,
        'success_scope':'Scripted task checks against enabled collision geometries; inspect collision coverage separately before claiming whole-robot clearance.',
        'target':target.tolist(),'events':events,'success':success,'failure_reason':failure_reason,
        'robot_station_contact_violation':contact_violation,'environment_contact_events':environment_contacts,'wall_seconds':time.monotonic()-start_wall,
        'fluid_final_metrics':fluid.metrics()}
    arrays={k:np.array([r[k] for r in records]) for k in records[0]}
    arrays['metadata_json']=np.array(json.dumps(metadata))
    arrays['drop_film_time_s']=arrays['fluid_time']
    if fluid.parcels:
        for key in fluid.parcels[0]:arrays['fluid_parcel_'+key]=np.array([p[key] for p in fluid.parcels])
    np.savez_compressed(out/'episode.npz',**arrays)
    (out/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('EPISODE_RESULT',json.dumps(metadata),flush=True)


if __name__=='__main__':main()
