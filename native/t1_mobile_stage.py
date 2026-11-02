"""Independent physical wheel-driven T1 mobile stage. No base pose teleporting."""
import os,json,importlib.util,argparse
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco
from t1_spincoat_scene import scene,ROOT

def add_mobile_freejoint(s):
    """Mutation hook for independently built task specs; do before compile."""
    roots=[b for b in s.bodies if b.parent is not None and b.name=='chassis_base']
    if not roots:
        raise RuntimeError('Official chassis_base root body missing')
    roots[0].add_freejoint(name='chassis_free')
    return s

def build_mobile_scene(start_xy=(0.,0.)):
    s,old,od,children=scene()
    add_mobile_freejoint(s)
    m=s.compile();d=mujoco.MjData(m)
    # Named-joint reset copies the delivered fixed-base natural ready pose.
    src=np.load(ROOT/'inputs/ready_pose.npz')['qpos'][0]
    for j in range(old.njnt):
        nj=m.joint(old.joint(j).name);n=7 if old.jnt_type[j]==0 else 4 if old.jnt_type[j]==1 else 1
        d.qpos[nj.qposadr[0]:nj.qposadr[0]+n]=src[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
    root=m.joint('chassis_free').qposadr[0]
    d.qpos[root:root+2]=start_xy  # One reset, not a runtime motion update.
    mujoco.mj_forward(m,d)
    return s,m,d,children

def official_driver(m):
    path=ROOT/'inputs/chassis_kinematics.py'
    spec=importlib.util.spec_from_file_location('mobile_official_chassis',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.MujocoOmniChassis(m,.078,.216,.214,max_wheel_speed=10.)

def hold_control(m,d,qref):
    for a in range(m.nu):
        name=m.actuator(a).name
        if name.startswith('astribot_chassis_motor_'):continue
        j=m.actuator_trnid[a,0]
        if j>=0:d.ctrl[a]=qref[m.jnt_qposadr[j]]

class MobileDockController:
    """GT odometry outer loop + official physical wheel map; no state assignment."""
    def __init__(self,m,target_xy,speed_limit=.09):
        self.driver=official_driver(m);j=m.joint('chassis_free')
        self.qa=int(j.qposadr[0]);self.dv=int(j.dofadr[0]);self.target=np.asarray(target_xy)
        self.speed_limit=speed_limit;self.stable_duration=0.;self.stopped=False;self.integral=np.zeros(2)
    def step(self,d,target_xy=None,enable_gate=True):
        if target_xy is not None:self.target=np.asarray(target_xy)
        a,v=self.qa,self.dv;yaw=self.driver._yaw_from_quat(*d.qpos[a+3:a+7])
        err=self.target-d.qpos[a:a+2]
        dt=float(d.time-self.last_time) if hasattr(self,'last_time') else 0.
        if enable_gate:self.integral=np.clip(self.integral+err*dt,-.005,.005)
        cmd=np.clip(2*err+.8*self.integral-.5*d.qvel[v:v+2],-self.speed_limit,self.speed_limit)
        c,sn=np.cos(yaw),np.sin(yaw);body=np.array([[c,sn],[-sn,c]])@cmd
        wz=np.clip(-2*yaw-.4*d.qvel[v+5],-.2,.2)
        self.driver.apply(d,-np.r_[body,wz])
        # Engineering gate; measurements are GT simulator odometry, not vision.
        stable=enable_gate and np.linalg.norm(err)<.001 and np.linalg.norm(d.qvel[v:v+2])<.001 and abs(d.qvel[v+5])<.002
        self.stable_duration=self.stable_duration+float(d.time-self.last_time) if stable and hasattr(self,'last_time') else 0.
        self.last_time=float(d.time);self.stopped=self.stable_duration>=.5
        return self.stopped

def advance_mobile_stage(m,d,target_xy,duration_s=18.,route_s=6.,hold_callback=None,log_stride=20,speed_limit=.09,stop_when_docked=False):
    """Continue supplied LIVE model/data without resetting robot or held object.

    hold_callback(m,d,qref) may preserve a task-specific five-finger grip/controller.
    Returns raw logs and GT audit. This interface is not a held-pipette validation.
    """
    qref=d.qpos.copy();ctrlref=d.ctrl.copy()
    nonwheel=[a for a in range(m.nu) if not m.actuator(a).name.startswith('astribot_chassis_motor_')]
    def preserve_grip_commands(m,d,qref):
        # Preserve contact preload: retargeting fingers to their measured qpos
        # would erase the commanded closure error and unload the grasp.
        d.ctrl[nonwheel]=ctrlref[nonwheel]
    hold=hold_callback or preserve_grip_commands
    dock=MobileDockController(m,target_xy,speed_limit=speed_limit);start=d.qpos[dock.qa:dock.qa+2].copy();t0=float(d.time)
    qlog=[];vlog=[];tlog=[];glog=[];clog=[];relative=[];tilts=[];first_stop=None
    try:pb=m.body('pipette/pipette').id;hb=m.body('sharpa_right/right_hand_C_MC').id
    except KeyError:pb=hb=None
    mujoco.mj_forward(m,d)
    def object_relative():
        return d.xmat[hb].reshape(3,3).T@(d.xpos[pb]-d.xpos[hb])
    rel0=object_relative().copy() if pb is not None else None
    for k in range(int(duration_s/m.opt.timestep)):
        elapsed=float(d.time-t0);u=np.clip((elapsed-1)/route_s,0,1);h=u**3*(10-15*u+6*u*u)
        hold(m,d,qref)
        stopped=dock.step(d,start+h*(np.asarray(target_xy)-start),enable_gate=elapsed>=1+route_s)
        if stopped and first_stop is None:first_stop=float(d.time)
        mujoco.mj_step(m,d)
        if k%log_stride==0 or (stop_when_docked and stopped):
            qlog.append(d.qpos.copy());vlog.append(d.qvel.copy());tlog.append(float(d.time));glog.append(stopped);clog.append(d.ctrl.copy())
            if pb is not None:
                relative.append(object_relative());tilts.append(float(np.rad2deg(np.arccos(np.clip(d.xmat[pb].reshape(3,3)[2,2],-1,1)))))
        if stop_when_docked and stopped:break
    qa,dv=dock.qa,dock.dv;ql=np.asarray(qlog);vl=np.asarray(vlog)
    logs={'qpos':ql,'qvel':vl,'time':np.asarray(tlog),'ctrl':np.asarray(clog),'stopped_gate':np.asarray(glog),'base_pose':ql[:,qa:qa+7],'base_twist':vl[:,dv:dv+6]}
    audit={'physics_dt_s':float(m.opt.timestep),'log_dt_s':log_stride*float(m.opt.timestep),'first_stopped_absolute_time_s':first_stop,'stopped_gate_final':dock.stopped,'endpoint_error_xy_mm':float(np.linalg.norm(d.qpos[qa:qa+2]-target_xy)*1000),'final_base_speed_m_s':float(np.linalg.norm(d.qvel[dv:dv+2])),'object_state_reset':False,'runtime_base_qpos_assignments':0,'held_object_validation':'Not certified by interface existence; inspect actual contact/slip and full rollout.'}
    audit.update(event_driven_handoff=bool(stop_when_docked),actual_stage_duration_s=float(d.time-t0),duration_cap_s=float(duration_s))
    if pb is not None:
        logs.update(pipette_relative_translation_m=np.asarray(relative),pipette_tilt_deg=np.asarray(tilts))
        audit.update(max_pipette_relative_translation_change_mm=float(np.max(np.linalg.norm(np.asarray(relative)-rel0,axis=1))*1000),max_pipette_tilt_deg=max(tilts))
    return logs,audit,dock

def run(args):
    s,m,d,children=build_mobile_scene(args.start_xy);drive=official_driver(m);qref=d.qpos.copy()
    dock=MobileDockController(m,np.asarray(args.start_xy)+[0,args.distance])
    j=m.joint('chassis_free');qa=j.qposadr[0];dv=j.dofadr[0]
    print('model',m.nq,m.nv,m.nu,'root',m.body(m.jnt_bodyid[j.id]).name,flush=True)
    out=ROOT/'research/hardware/mobile_stage'/args.tag;out.mkdir(parents=True,exist_ok=True)
    initial=d.qpos[qa:qa+7].copy();qs=[];vs=[];ts=[];cs=[];ctl=[];gates=[];max_tilt=0;wheel_contacts=set();robot_env=[];first_stopped=None
    # An explicit world-error controller feeds the OFFICIAL wheel velocity map.
    # Official wheel direction sign is tested rather than assumed to be world positive.
    for k in range(int(args.duration/m.opt.timestep)):
        t=d.time;hold_control(m,d,qref)
        progress=np.clip((t-2)/6,0,1);blend=progress**3*(10-15*progress+6*progress**2)
        target=initial[:2]+[0,args.distance*blend]
        stopped=dock.step(d,target,enable_gate=t>=8)
        if stopped and first_stopped is None:first_stopped=float(t)
        mujoco.mj_step(m,d)
        tilt=float(np.arccos(np.clip(mujoco_quat_up(d.qpos[qa+3:qa+7])[2],-1,1)));max_tilt=max(max_tilt,tilt)
        if k%20==0:
            qs.append(d.qpos.copy());vs.append(d.qvel.copy());ts.append(d.time);ctl.append(d.ctrl.copy());gates.append(stopped);nwheel=0
            for ci in range(d.ncon):
                con=d.contact[ci];names=[m.geom(int(g)).name or '' for g in (con.geom1,con.geom2)]
                if any('roller' in n or 'wheel' in n for n in names) and 'floor' in names:
                    nwheel+=1;wheel_contacts.update(names)
            cs.append(nwheel)
        if not np.isfinite(d.qpos).all() or tilt>.5:raise RuntimeError('Unstable chassis')
    arr=np.asarray(qs);vv=np.asarray(vs);tt=np.asarray(ts);tail=tt>args.duration-2
    result={'tag':args.tag,'method':'freejoint dynamics + official four omni velocity actuators; base qpos set only at reset','distance_requested_m':args.distance,'sign':args.sign,'initial_root_qpos':initial.tolist(),'final_root_qpos':d.qpos[qa:qa+7].tolist(),'endpoint_error_xy_mm':float(np.linalg.norm(d.qpos[qa:qa+2]-(initial[:2]+[0,args.distance]))*1000),'final_base_speed_m_s':float(np.linalg.norm(d.qvel[dv:dv+2])),'max_base_tilt_deg':float(np.rad2deg(max_tilt)),'last2s_xy_drift_mm':float(np.max(np.linalg.norm(arr[tail,qa:qa+2]-arr[-1,qa:qa+2],axis=1))*1000),'wheel_floor_contact_sample_min':int(min(cs)),'wheel_floor_contact_sample_max':int(max(cs)),'wheel_contact_geom_names':sorted(wheel_contacts),'model_nq':m.nq,'model_nv':m.nv,'model_nu':m.nu,'base_qpos_index':int(qa),'base_dof_index':int(dv),'limitations':['Empty hands; pipette remains on stand.','No base-mounted visual relocalization or precision dispensing yet.','Original T1 arm/torso physical collider coverage remains incomplete.']}
    result.update({'first_stopped_gate_time_s':first_stopped,'stopped_gate_final':dock.stopped,'physics_dt_s':float(m.opt.timestep),'log_dt_s':.020,'stopped_gate_definition':'XY error<1mm, planar speed<1mm/s, yaw rate<.002rad/s continuously .5s after route complete; GT odometry only'})
    result['success']=result['endpoint_error_xy_mm']<5 and result['final_base_speed_m_s']<.005 and result['max_base_tilt_deg']<5 and result['wheel_floor_contact_sample_max']>0 and dock.stopped
    np.savez_compressed(out/'episode.npz',qpos=arr,qvel=vv,time=tt,ctrl=np.asarray(ctl),wheel_floor_contacts=cs,stopped_gate=gates,base_pose=arr[:,qa:qa+7],base_twist=vv[:,dv:dv+6])
    (out/'result.json').write_text(json.dumps(result,indent=2));mujoco.mj_saveModel(m,str(out/'model.mjb'));print(json.dumps(result,indent=2),flush=True)

def mujoco_quat_up(q):
    mat=np.zeros(9);mujoco.mju_quat2Mat(mat,q);return mat.reshape(3,3)[:,2]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='empty_lateral_01');ap.add_argument('--distance',type=float,default=.3);ap.add_argument('--duration',type=float,default=16);ap.add_argument('--sign',type=float,default=-1);ap.add_argument('--start-xy',type=float,nargs=2,default=[0.,0.]);run(ap.parse_args())
