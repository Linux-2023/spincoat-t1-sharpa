"""One fresh supported-contact / seat-then-close experiment. No geometry edits."""
import os,json,types,argparse
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from t1_spincoat_scene import ROOT
from run_t1_integrated_workflow import run_child
from run_t1_aspiration_probe import run_aspiration_on_state
from t1_support_grasp import make_support_plan,prepare_support_grasp
import t1_coarse_stop

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--time-scale',type=float,default=1.25);a=p.parse_args();out=ROOT/'research/hardware/grasp_azimuth'/a.tag;out.mkdir(parents=True,exist_ok=False);plan=make_support_plan(out/'plan');(out/'contact_plan.json').write_text(json.dumps(plan,indent=2))
 if not plan['planning_success']:(out/'result.json').write_text(json.dumps({'success':False,'failure':'static_support_fit_rejected','plan':plan},indent=2));print('STATIC_REJECT',flush=True);return
 ref=ROOT/'logs/spincoat/t1/integrated_direct07_01';m=mujoco.MjModel.from_binary_path(str(ref/'model.mjb'));d=mujoco.MjData(m);z=np.load(ref/'grasp.npz');d.qpos[:]=z['qpos'][0];wheel=[i for i in range(m.nu) if m.actuator(i).name.startswith('astribot_chassis_motor_')];m.actuator_gainprm[wheel,0]=30.;m.actuator_biasprm[wheel,2]=-30.;m.opt.noslip_iterations=10;t1_coarse_stop.ZERO_SPEED_GAIN_SCALE=32.
 runner,path,callback,getaudit,timing=prepare_support_grasp(m,d,out,plan,time_scale=a.time_scale);mujoco.mj_saveModel(m,str(out/'initial_model.mjb'));modelid,dataid=id(m),id(d);handoffs=[]
 def handoff(stage):handoffs.append({'stage':stage,'time_s':float(d.time),'model_identity':id(m),'data_identity':id(d)});np.savez_compressed(out/('handoff_'+stage+'.npz'),qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
 handoff('initial');native=mujoco.mj_step;records=[];digitnames=['thumb','index','middle','ring','pinky']
 def monitor(model,data,*args,**kw):
  result=native(model,data,*args,**kw);counts=np.zeros(5,int);positive=np.zeros(5,int);normal=np.zeros(5);magnitude=np.zeros(5);segments=np.zeros(5,int);rack_count=0;rack_force=0.
  for ci,c in enumerate(data.contact):
   names=[model.body(model.geom_bodyid[int(g)]).name for g in [c.geom1,c.geom2]]
   if not any(n.startswith('pipette/') for n in names):continue
   force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force);fn=abs(float(force[0]));fm=float(np.linalg.norm(force[:3]))
   if any(n.startswith('stand/') for n in names):rack_count+=1;rack_force+=fn
   for j,name in enumerate(digitnames):
    hand=next((n for n in names if n.startswith('sharpa_right/right_'+name+'_')),None)
    if hand:
     counts[j]+=1;positive[j]+=int(fm>.01);normal[j]+=fn;magnitude[j]+=fm;segments[j]|=2 if hand.endswith('_PP') else 4 if hand.endswith('_MP') else 8 if hand.endswith('_DP') else 1
  records.append((float(data.time),float(data.qpos[model.joint('pipette/pipette_button').qposadr[0]]),counts,positive,normal,magnitude,segments,rack_count,rack_force));return result
 mujoco.mj_step=monitor
 try:
  raw,gr=run_child(runner,types.SimpleNamespace(to_xml=lambda:'<!-- support staged initial model is authoritative -->'),m,d,[],a.tag+'_grasp',['--motion-profile',str(path),'--close-pinky-before-lift','--lab-context','--max-seconds',str(11.5*a.time_scale)],callback);np.savez_compressed(out/'episode.npz',**raw);mujoco.mj_saveModel(m,str(out/'model.mjb'));ga=getaudit();(out/'grasp_contact_audit.json').write_text(json.dumps(ga,indent=2));handoff('grasp_end');result={'success':False,'scope':'Fresh same-live local support contact fit and staged seating/closure/lift, optional aspiration; no full transfer/coating test','prefix_clear':ga['prefix_clear'],'grasp_failure':gr['failure_reason'],'station_contact_violation':gr['robot_station_contact_violation'],'timing':timing,'handoffs':handoffs,'force_logging':'Read-only1kHz contact force/count observations; not control input or tactile policy','plan_contract':plan['contract']};(out/'result.json').write_text(json.dumps(result,indent=2))
  if ga['prefix_clear'] and not gr['failure_reason'] and not gr['robot_station_contact_violation']:
   glob=dict(run_aspiration_on_state.__globals__);glob['MobileDockController']=lambda model,target:getaudit.dock;hold=glob['hold_control'];started=float(d.time);index=next(i for i in range(m.nu) if 'right_index_DIP' in m.joint(int(m.actuator_trnid[i,0])).name)
   def preload(model,data,desired):
    hold(model,data,desired);u=np.clip((data.time-started)/.5,0,1);u=u**3*(10+u*(-15+6*u));data.ctrl[index]=np.clip(data.ctrl[index]+.015*u,*model.actuator_ctrlrange[index])
   glob['hold_control']=preload;source=(ROOT/'research/hardware/direct_grasp/direct_oblique_07/executed_aspiration_adapter.py').read_text().replace('stroke_error=-.0051-float(d.qpos[button])','stroke_error=-.0052-float(d.qpos[button])');(out/'executed_aspiration_adapter.py').write_text(source);exec(compile(source,str(out/'executed_aspiration_adapter.py'),'exec'),glob);arrays,ar,ledger=glob['run_aspiration_on_state'](m,d,provenance={'entry':'fresh supported-contact grasp and staged seating/closure/lift; same live model/data'});np.savez_compressed(out/'aspiration.npz',**arrays);result['live_aspiration']=ar;handoff('aspiration_end');result['success']=ar['success']
 finally:
  mujoco.mj_step=native
  if records:
   names=['time','button_qpos_m','finger_contact_count','finger_positive_wrench_count','finger_normal_force_N','finger_wrench_magnitude_N','finger_contact_segment_bits','tool_rack_contact_count','tool_rack_normal_force_N'];arrays={name:np.array([r[i] for r in records]) for i,name in enumerate(names)};np.savez_compressed(out/'fullrate_contact_observer.npz',**arrays,digit_names=np.array(digitnames));np.savez_compressed(out/'actual_button_1khz.npz',time=arrays['time'],button_qpos_m=arrays['button_qpos_m'])
  dockaudit=t1_coarse_stop.save_coarse_control(getaudit.dock,out/'left_coarse_control.npz');np.savez_compressed(out/'checkpoint.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time);mujoco.mj_saveModel(m,str(out/'final_model.mjb'))
 if id(m)!=modelid or id(d)!=dataid:raise RuntimeError('live state identity changed')
 with np.load(out/'left_coarse_control.npz') as z:
  mask=z['time']>=5.8;speed=np.hypot(z['base_vx'][mask],z['base_vy'][mask]);yaw=abs(z['yaw_rate'][mask]);bad=(speed>=.001)|(yaw>=.002)|(z['workspace_error_m'][mask]>.01);fine={'samples':len(speed),'violations':int(bad.sum()),'speed_max_mm_s':float(speed.max()*1000),'yaw_rate_max_rad_s':float(yaw.max())}
 result['base_fine']=fine;result['success']=bool(result['success'] and not fine['violations']);result['actual_button_peak_mm']=float(-arrays['button_qpos_m'].min()*1000);result['final_time_s']=float(d.time);result['handoffs']=handoffs;(out/'result.json').write_text(json.dumps(result,indent=2));print('SUPPORT_STAGED_RESULT',json.dumps({k:v for k,v in result.items() if k!='live_aspiration'}),flush=True)
if __name__=='__main__':main()
