"""Opt-in open-collar withdrawal along the actual stand +Y axis; original hand targets/physics."""
import inspect
import numpy as np

def _c2(u):
 u=float(np.clip(u,0,1));return u*u*u*(10+u*(-15+6*u))

def install_slot_exit(module,m,d,*,distance_m=.012,start_task_s=3.,end_task_s=4.5,time_scale=1.25):
 if not 0<=distance_m<=.025 or not 1.5<=start_task_s<end_task_s<=4.5:raise ValueError('Invalid fixed collar exit parameters')
 old=module.ik;closure=inspect.getclosurevars(old);native=closure.nonlocals['original_ik'];env=old.__globals__;clock=env['station_clock_origin'];station_point=env['station_point'];station_quat=env['station_quat'];vert=float(env['station_verticalize_start']);axis=d.xmat[m.body('stand/pipette_stand').id].reshape(3,3)[:,1].copy();trace=[]
 def shifted_ik(model,data,position,quat,**kw):
  elapsed=round((float(data.time)-clock[0])/model.opt.timestep)*model.opt.timestep;task=(elapsed-5.8)/time_scale
  if elapsed<vert and task>=start_task_s and kw.get('iterations')!=400:
   distance=distance_m*_c2((task-start_task_s)/(end_task_s-start_task_s));point=station_point(np.asarray(position))+axis*distance;rotation=station_quat(quat);result=native(model,data,point,rotation,**kw);trace.append((float(data.time),task,distance,point.copy(),float(result[1])));return result
  return old(model,data,position,quat,**kw)
 module.ik=shifted_ik;module.right_slot_exit_trace=trace;module.right_slot_exit_config=dict(scope=__doc__,distance_m=distance_m,start_task_s=start_task_s,end_task_s=end_task_s,time_scale=time_scale,actual_open_axis_world=axis.tolist(),force_or_tactile_feedback_used=False,new_finger_targets=False)
 return module

def adapt_neutral_source(source,*,distance_m=.012,start_task_s=3.,end_task_s=4.5):
 anchor='        runner,path,callback,audit,clock=prepared\n'
 if source.count(anchor)!=1:raise RuntimeError('Exact original prepared-runner anchor required once')
 insertion='        from t1_right_slot_exit_v5 import install_slot_exit\n        install_slot_exit(runner,m,d,distance_m='+repr(distance_m)+',start_task_s='+repr(start_task_s)+',end_task_s='+repr(end_task_s)+",time_scale=clock['time_scale'])\n"
 return source.replace(anchor,anchor+insertion)

def adapt_launch_source(source,**kw):
 anchor=" layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))\n source=Path(baseline.__file__).read_text()\n"
 if source.count(anchor)!=1:raise RuntimeError('Exact outer neutral source-read anchor required once')
 return source.replace(anchor,anchor+' from t1_right_slot_exit_v5 import adapt_neutral_source as adapt_right_slot_exit_source\n source=adapt_right_slot_exit_source(source,**'+repr(kw)+')\n')
