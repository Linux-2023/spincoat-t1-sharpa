"""Privileged geometric pre-seat reanchor to the actual free pipette posture.
Single GT6D sample before seating, no force/tactile adaptation, no pose writes.
"""
import inspect
import numpy as np
from scipy.spatial.transform import Rotation

def install_seat_alignment(module,m,d,*,nominal_position=None,nominal_rotation=None,time_scale=1.25):
 pb0=m.body('pipette/pipette').id
 if nominal_position is None:nominal_position=d.xpos[pb0].copy()
 if nominal_rotation is None:nominal_rotation=d.xmat[pb0].reshape(3,3).copy()
 old=module.ik;env=old.__globals__;origin=env['station_clock_origin'];verticalize=float(env['station_verticalize_start']);cells=dict(zip(old.__code__.co_freevars,old.__closure__));cell=cells['original_ik'];native=cell.cell_contents;pb=m.body('pipette/pipette').id;P0=np.array(nominal_position);R0=np.array(nominal_rotation);reference={};trace=[]
 def aligned_native(model,data,point,quat,**kw):
  elapsed=round((float(data.time)-origin[0])/model.opt.timestep)*model.opt.timestep
  if elapsed<5.8 or elapsed>=verticalize:return native(model,data,point,quat,**kw)
  if not reference:
   reference.update(position=data.xpos[pb].copy(),rotation=data.xmat[pb].reshape(3,3).copy());reference['rotation_vector']=Rotation.from_matrix(reference['rotation']@R0.T).as_rotvec()
  u=float(np.clip((elapsed-5.8)/(1.5*time_scale),0,1));u=u*u*u*(10+u*(-15+6*u));turn=Rotation.from_rotvec(u*reference['rotation_vector']).as_matrix();p=P0+u*(reference['position']-P0)+turn@(np.asarray(point)-P0);R=turn@Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix();q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];result=native(model,data,p,q,**kw);trace.append((float(data.time),u,p.copy(),float(result[1])));return result
 cell.cell_contents=aligned_native;module.right_seat_alignment_reference=reference;module.right_seat_alignment_trace=trace;module.right_seat_alignment_config=dict(scope=__doc__,nominal_position_m=P0.tolist(),nominal_rotation=R0.tolist(),time_scale=time_scale)
 return module


def adapt_neutral_source(source,*,distance_m=.012):
 anchor='        runner,path,callback,audit,clock=prepared\n'
 if source.count(anchor)!=1:raise RuntimeError('Exact original prepared-runner anchor required once')
 insert="        from t1_right_seat_alignment_v5 import install_seat_alignment\n        from t1_right_slot_exit_v5 import install_slot_exit\n        install_seat_alignment(runner,m,d,time_scale=clock['time_scale'])\n        install_slot_exit(runner,m,d,distance_m="+repr(distance_m)+",start_task_s=3.,end_task_s=4.5,time_scale=clock['time_scale'])\n"
 return source.replace(anchor,anchor+insert)

def adapt_launch_source(source,*,distance_m=.012):
 anchor=" layout_config=RightWorkstationV4(station_shift_world_m=(a.v4_station_shift_x_m,a.v4_station_shift_y_m,0.))\n source=Path(baseline.__file__).read_text()\n"
 if source.count(anchor)!=1:raise RuntimeError('Exact outer neutral source-read anchor required once')
 insert=' from t1_right_seat_alignment_v5 import adapt_neutral_source as adapt_right_seat_exit_source\n source=adapt_right_seat_exit_source(source,distance_m='+repr(distance_m)+')\n'
 return source.replace(anchor,anchor+insert)
