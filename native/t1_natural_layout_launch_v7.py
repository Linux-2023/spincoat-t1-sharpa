"""V7 RESET scene selects explicit provisional anatomical LEFT mount convention."""
from pathlib import Path
import json
import numpy as np,mujoco
import t1_interior_layout_launch_v6 as original

def adapt_neutral_scene_source(source,layout_path):
 layout=json.loads(Path(layout_path).read_text())
 if layout.get('left_hand_mount_convention')!='RIGHT_anatomical_mirror_Rz180_provisional':raise ValueError('Explicit provisional anatomical LEFT mount convention required')
 source=original.adapt_neutral_scene_source(source,layout_path)
 station_yaw=float(layout['right_runtime_config']['stand_yaw_deg'])
 if station_yaw not in (60.,75.):raise ValueError('Unsupported v7 canonical stand yaw')
 anchor=',stand_yaw=60.,coater_xy='
 if source.count(anchor)!=1:raise RuntimeError('Exactly one canonical stand yaw anchor required')
 source=source.replace(anchor,',stand_yaw='+repr(station_yaw)+',coater_xy=',1)
 old='from t1_interior_scene_v6 import build_scene as build_canonical_v6_scene'
 if source.count(old)!=1:raise RuntimeError('Exactly one canonical scene factory required')
 source=source.replace(old,'from t1_anatomical_mount_scene_v7 import build_scene as build_canonical_v6_scene')
 anchor="    exec(compile(engine_text,engine.__file__,'exec'),engine.__dict__)"
 if source.count(anchor)!=1:raise RuntimeError('Exactly one finalized engine compilation required')
 return source.replace(anchor,"    from t1_natural_layout_launch_v7 import adapt_final_reset_ready_engine\n    engine_text=adapt_final_reset_ready_engine(engine_text)\n"+anchor)

def adapt_launch_source(source,layout_path):
 source=original.adapt_launch_source(source,layout_path)
 old='from t1_interior_layout_launch_v6 import adapt_neutral_scene_source'
 if source.count(old)!=1:raise RuntimeError('Exactly one neutral layout adapter import required')
 return source.replace(old,'from t1_natural_layout_launch_v7 import adapt_neutral_scene_source')


def apply_final_anatomical_ready_reset(m,d,out):
 if abs(float(d.time))>1e-12:raise RuntimeError('Anatomical ready may be assigned only at fresh RESET')
 before=d.qpos.copy();changed=[];values={}
 for i,sign in enumerate([-1.,1.,-1.,1.,-1.,1.,-1.],1):
  left=m.joint('astribot_arm_left_joint_'+str(i));right=m.joint('astribot_arm_right_joint_'+str(i));value=sign*float(d.qpos[right.qposadr[0]])
  if not m.jnt_range[left.id,0]<=value<=m.jnt_range[left.id,1]:raise ValueError('Anatomical ready joint limit violation')
  d.qpos[left.qposadr[0]]=value;changed.append(int(left.qposadr[0]));values[left.name]=value
 mask=np.ones(m.nq,bool);mask[changed]=False
 assert np.array_equal(before[mask],d.qpos[mask])
 (Path(out)/'anatomical_ready_final_reset_v7.json').write_text(json.dumps(dict(scope='Seven LEFT arm joints only, after legacy task-ready RESET overlay and before any simulation step',time_s=float(d.time),joint_positions=values,all_other_qpos_exact=True,ctrl_unchanged=True,hardware_mount_calibrated=False),indent=2))


def assert_final_anatomical_ready_reset(m,d,out):
 if abs(float(d.time))>1e-12:raise RuntimeError('Initial handoff must be at RESET')
 actual=[];expected=[]
 for i,sign in enumerate([-1.,1.,-1.,1.,-1.,1.,-1.],1):
  actual.append(float(d.qpos[m.joint('astribot_arm_left_joint_'+str(i)).qposadr[0]]));expected.append(sign*float(d.qpos[m.joint('astribot_arm_right_joint_'+str(i)).qposadr[0]]))
 if not np.array_equal(actual,expected):raise AssertionError('A later RESET initializer overwrote anatomical LEFT ready')
 quat=m.body_quat[m.body('sharpa_left/left_hand_C_MC').id]
 if min(np.linalg.norm(quat-np.array([0.,0.,0.,1.])),np.linalg.norm(quat+np.array([0.,0.,0.,1.])))>1e-10:raise AssertionError('Wrong LEFT mounting convention')
 (Path(out)/'anatomical_ready_initial_handoff_assert_v7.json').write_text(json.dumps(dict(time_s=float(d.time),left_q_rad=actual,expected_mirror_q_rad=expected,pass_=True),indent=2))


def adapt_final_reset_ready_engine(source):
 anchor="    for name,value in config['initial_joint_positions'].items():d.qpos[m.joint(name).qposadr[0]]=value"
 if source.count(anchor)!=1:raise RuntimeError('Unique legacy RESET overlay required')
 source=source.replace(anchor,anchor+"\n    from t1_natural_layout_launch_v7 import apply_final_anatomical_ready_reset,assert_final_anatomical_ready_reset\n    apply_final_anatomical_ready_reset(m,d,out)")
 anchor="    handoff('initial')"
 if source.count(anchor)!=1:raise RuntimeError('Unique initial physical handoff required')
 source=source.replace(anchor,"    assert_final_anatomical_ready_reset(m,d,out)\n"+anchor)
 compile(source,'<v7 final RESET readiness>','exec');return source
