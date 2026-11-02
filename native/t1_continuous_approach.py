"""Continuous forward/down approach; station60 and grasp/finger targets unchanged."""
import json,types,inspect
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from scipy.interpolate import make_interp_spline
from t1_spincoat_scene import ROOT,arm_indices,ik
from run_t1_integrated_workflow import *
from probe_t1_station_orientation_chain import station_prepare as baseline_station_prepare
_CURVES={}
APPROACH_SIDE_BIAS=-.12
def continuous_prefix_sample(profile,t):
 key=id(profile)
 if key not in _CURVES:
  q=np.array(profile['continuous_approach']['arm_q_rad']);times=np.array(profile['continuous_approach']['time_s']);zero=np.zeros(q.shape[1]);_CURVES[key]=(profile,times,q,make_interp_spline(times,q,k=5,bc_type=([(1,zero),(2,zero)],[(1,zero),(2,zero)])))
 _,times,q,curve=_CURVES[key]
 return q[0].copy() if t<=times[0] else q[-1].copy() if t>=times[-1] else curve(t)

def build_continuous_profile(m,d,path,out):
 profile=json.loads(path.read_text());ids,qa,ad=arm_indices(m);bqa=int(m.joint('chassis_free').qposadr[0]);palm=m.body('sharpa_right/right_hand_C_MC').id;work=mujoco.MjData(m);work.qpos[:]=d.qpos;first=np.array(profile['waypoints'][0]['arm_q_rad']);last=np.array(profile['waypoints'][-1]['arm_q_rad']);work.qpos[qa]=first;work.qpos[bqa]=-.08;mujoco.mj_kinematics(m,work);start=work.xpos[palm].copy();R0=work.xmat[palm].reshape(3,3).copy();start[0]-=.02;first,initial_err=ik(m,work,start,Rotation.from_matrix(R0).as_quat()[[3,0,1,2]],seed=first,iterations=180);work.qpos[qa]=first;mujoco.mj_kinematics(m,work);start=work.xpos[palm].copy();work.qpos[qa]=last;work.qpos[bqa]=0.;mujoco.mj_kinematics(m,work);end=work.xpos[palm].copy();R1=work.xmat[palm].reshape(3,3).copy();mid=(start+end)/2+np.array([0,APPROACH_SIDE_BIAS,.015]);slerp=Slerp([0,1],Rotation.from_matrix([R0,R1]));times=np.linspace(.7,5.8,129);qs=[];audit=[];previous=first.copy()
 for t in times:
  u=np.clip((t-.7)/5.1,0,1);s=u*u*u*(10+u*(-15+6*u));point=(1-s)**2*start+2*s*(1-s)*mid+s*s*end;yu=np.clip((t-.7)/1.4,0,1) if t<=2.1 else np.clip((t-2.1)/3.7,0,1);ys=yu*yu*yu*(10+yu*(-15+6*yu));point[1]=start[1]+ys*(-.186-start[1]) if t<=2.1 else -.186+ys*(end[1]+.186);ru=np.clip((t-.7)/1.2,0,1);rs=ru*ru*ru*(10+ru*(-15+6*ru));R=slerp(float(rs)).as_matrix();quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];v=np.clip((t-.7)/2.,0,1);work.qpos[bqa]=-.08*(1-v*v*v*(10+v*(-15+6*v)));q,err=ik(m,work,point,quat,seed=previous,iterations=150);previous=q;work.qpos[qa]=q;mujoco.mj_kinematics(m,work);er=np.linalg.norm(Rotation.from_matrix(work.xmat[palm].reshape(3,3)@R.T).as_rotvec());qs.append(q);audit.append({'time_s':float(t),'target_palm_m':point.tolist(),'position_error_mm':err*1000,'orientation_error_deg':float(np.rad2deg(er))})
 if max(x['position_error_mm'] for x in audit)>.2 or max(x['orientation_error_deg'] for x in audit)>.1:raise RuntimeError('Continuous approach IK failed: '+str(max(x['position_error_mm'] for x in audit)))
 d.qpos[qa]=first;mujoco.mj_forward(m,d);profile['continuous_approach']={'initial_ready_retreat_x_m':.02,'time_s':times.tolist(),'arm_q_rad':np.asarray(qs).tolist(),'cartesian_start_m':start.tolist(),'cartesian_control_m':mid.tolist(),'cartesian_end_m':end.tolist(),'method':'Continuous monotone forward/down Bezier with a smooth lateral clearance excursion, early wrist turn, IK and globalC4 quintic joint spline; no interior full stop.'};profile['waypoints']=[{'name':'ready_hold','time_s':0.,'arm_q_rad':first.tolist()}]+[{'name':'continuous_forward_down','time_s':float(t),'arm_q_rad':q.tolist()} for t,q in zip(times,qs)];profile['interpolator']='Continuous global quintic joint spline; no interior waypoint stop';profile['approach_ik']=audit;path.write_text(json.dumps(profile,indent=2));qsample=np.array([continuous_prefix_sample(profile,float(t)) for t in np.linspace(.7,5.8,5101)]);lo=m.jnt_range[ids,0];hi=m.jnt_range[ids,1];report={'start_m':start.tolist(),'control_m':mid.tolist(),'end_m':end.tolist(),'duration_s':5.8,'motion_start_s':.7,'formal_close_time_s':5.8,'min_joint_margin_rad':float(np.minimum(qsample-lo,hi-qsample).min()),'max_joint_speed_rad_s':float(np.max(abs(np.gradient(qsample,.001,axis=0)))),'initial_ready_retreat_x_m':.02,'finger_targets_or_hand_timing_changed':False,'scope':'Kinematic plan, pending actual fresh grasp physics/whole-mesh verification'};(out/'continuous_approach_plan.json').write_text(json.dumps(report,indent=2));return profile

def prepare_continuous_approach(m,d,out,plan,*,time_scale=1.25):
 baseline_station_prepare.angle=60.;baseline_station_prepare.closing_scale=1.;source=inspect.getsource(baseline_station_prepare);needle="glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob.update(";source=source.replace(needle,"glob=dict(t1_support_grasp.prepare_support_grasp.__globals__);glob['adapted_runner']=adapted_runner;glob.update(");globals_copy=dict(baseline_station_prepare.__globals__);globals_copy['adapted_runner']=adapted_runner;exec(compile(source,str(out/'continuous_station_prepare.py'),'exec'),globals_copy);(out/'continuous_station_prepare.py').write_text(source);function=globals_copy['station_prepare'];function.angle=60.;function.closing_scale=1.;result=function(m,d,out,plan,time_scale=time_scale);build_continuous_profile(m,d,result[1],out);return result

prepare_smooth_station_grasp=prepare_continuous_approach







def adapted_runner(kind,fast_transfer=False,controlled_stop=False,rapid_dispense=False,left_gate=False):
    source=(ROOT/'scripts/run_t1_spincoat_episode.py').read_text()
    source=source.replace("robot_dofs=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if not m.joint(j).name.startswith(('pipette/','spincoater/'))])",
      "robot_dofs=np.array(sorted({int(m.jnt_dofadr[m.actuator_trnid[a,0]]) for a in range(m.nu) if not m.actuator(a).name.startswith('astribot_chassis_motor_') and m.actuator_trnid[a,0]>=0}))")
    source=source.replace("seed=np.load(ROOT/'research/hardware/grasp/probe.npz')['qpos'][0][qa]", "seed=d.qpos[qa].copy()")
    if kind=='grasp':
        source=source.replace('base=PIPETTE_START+GRASP_WRIST_OFFSET','base=PIPETTE_START+GRASP_WRIST_OFFSET+np.array([0,.4,0])')
    else:
        if fast_transfer:
            source=source.replace('task_t=t-prefix', "task_t=t-prefix\n        e=max(0.,task_t-9.)\n        task_t=float(np.interp(e,[0,2.,3.5,4.,4.2],[9.,13.,16.,18.,21.])) if e<=4.2 else 21.+e-4.2")
        if rapid_dispense:
            source=source.replace('e,[0,2.,3.5,4.,4.2],[9.,13.,16.,18.,21.])) if e<=4.2 else 21.+e-4.2','e,[0,1.2,2.4],[9.,13.,21.])) if e<=2.4 else 21.+e-2.4')
            # Map descent directly to task16 at2.4s; omit fixed settle/thumb holds.
            source=source.replace('[9.,13.,21.]','[9.,13.,16.]')
            source=source.replace('else 21.+e-2.4','else 21.+e-2.4')
        source=source.replace("        if phase!=last_phase:","""        workflow_press_enabled[0]=False
        if phase=='press':
            workflow_press_enabled[0]=bool(workflow_base_ready())
            if workflow_press_enabled[0]:
                if workflow_press_start[0] is None:workflow_press_start[0]=d.geom_xpos[pad].copy()
                workflow_press_clock[0]+=.001
                task_t=21.+workflow_press_clock[0]
            else:phase='wait_for_base_stop'
        if phase!=last_phase:""")
        source=source.replace('q,error=ik(m,d,base,HAND_QUAT,seed=seed,iterations=400);assert error<.001,error','q=d.qpos[qa].copy();error=0.')
        source=source.replace('d.qpos[qa]=q;d.qpos[hq]=openq','pass  # live stage; never reset robot or object')
        source=source.replace('desired=d.qpos.copy();desired_hand=close.copy()', 'desired=d.qpos.copy();desired_hand=current_hand_controls(m,d,hj)')
        source=source.replace('t=step*.001\n','t=step*.001+9.\n')
        source=source.replace('target=d.site_xpos[targetid].copy()','audit_target=d.site_xpos[targetid].copy();target=visual_target.copy()')
        source=source.replace('impact[:2]-target[:2]','impact[:2]-audit_target[:2]')
        source=source.replace('if not launched and d.qpos[button_q]<-.005:', 'if not launched and d.qpos[button_q]<-.005 and ledger.delivered_m3>=.95*ledger.p.capacity_m3:')
        source=source.replace('F.SpincoatFluid(F.FluidParameters())','F.SpincoatFluid(F.FluidParameters(volume_m3=ledger.remaining_volume_m3))')
        source=source.replace("if not valid:failure_reason='invalid_or_premature_dispense';break", "if not valid:failure_reason='invalid_or_premature_dispense';break\n            fluid=F.SpincoatFluid(F.FluidParameters(volume_m3=ledger.delivered_m3))")
        source=source.replace("if phase in ('thumb_above','press','retract','spin'):","if phase in ('press','retract','spin'):")
        source=source.replace("elif phase=='press':thumb_goal=top+axis*(.010-.025*smooth((task_t-21)/3))", "elif phase=='press':\n                        u=smooth((task_t-21)/3);thumb_goal=(1-u)*workflow_press_start[0]+u*(top-axis*.015)")
    if rapid_dispense:
        source=source.replace("else:phase='wait_for_base_stop'","else:phase='wait_for_motion_ready'")
        source=source.replace('u=smooth((task_t-21)/3);thumb_goal=', 'u=smooth((task_t-21)/1.8);thumb_goal=')
    if controlled_stop:
        source=source.replace('if motor_at is not None:omega=20*np.pi*min(1,(t-motor_at)/.5)', 'if motor_at is not None:omega=20*np.pi*min(1,(t-motor_at)/.5)*(1-smooth((t-motor_at-4.2)/2.))')
        source=source.replace('if motor_at is not None and t-motor_at>=4.2:break','if motor_at is not None and t-motor_at>=6.7:break')
    source=source.replace("'fixed_base':True", "'fixed_base':False")
    source=source.replace("'starts_from_home':args.from_home,'home_approach_duration_s':prefix", "'starts_from_home':False,'home_approach_duration_s':0.,'task_ready_approach_duration_s':prefix")
    source=source.replace("'pipette_start':PIPETTE_START.tolist()", "'pipette_start':(PIPETTE_START+np.array([0,.4,0])).tolist()")
    source=source.replace("'target':target.tolist(),'events':events", "'target':target.tolist(),'controller_clock_offset_s':workflow_clock_offset_s,'time_basis':'stage_logical_seconds','events':events")
    if kind=='grasp' and left_gate:
        source=source.replace("        if phase!=last_phase:", "        if phase=='close' and last_phase!='close' and not workflow_left_gate():failure_reason='left_base_not_stopped_before_close';break\n        if phase!=last_phase:")
    if kind=='grasp':source=source.replace('                for (ta,pa),(tb,pb) in zip(frames[:-1],frames[1:]):\n                    if ta<=t<=tb:\n                        u=smooth((t-ta)/(tb-ta));q=(1-u)*pa+u*pb;break','                q=continuous_prefix_sample(profile,t)')
    module=types.ModuleType('front_workflow_'+kind);module.__file__=str(ROOT/'scripts/run_t1_spincoat_episode.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    module.GRASP_WRIST_OFFSET=np.array([-.025,.09,.12])
    module.workflow_press_clock=[0.]
    module.workflow_press_enabled=[False]
    module.workflow_press_start=[None]
    module.workflow_base_ready=lambda:False
    module.workflow_clock_offset_s=0.
    module.continuous_prefix_sample=continuous_prefix_sample
    return module
