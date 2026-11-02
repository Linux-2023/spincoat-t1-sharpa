"""Opt-in empty-hand retreat and crossdigit-aware relaxed-ready interpolation."""
import numpy as np

def smooth(u):
 u=np.clip(u,0.,1.);return u*u*u*(10+u*(-15+6*u))

def safe_ready_hand_target(m,hand_joint_ids,start,goal,progress):
 start=np.asarray(start);goal=np.asarray(goal);u=float(progress);value=start+(goal-start)*smooth(u)
 spread={'left_index_MCP_AA':-.06,'left_middle_MCP_AA':0.,'left_ring_MCP_AA':.04}
 for i,j in enumerate(hand_joint_ids):
  name=m.joint(int(j)).name.split('/')[-1]
  if name not in spread:continue
  mid=spread[name]
  value[i]=start[i]+(mid-start[i])*smooth(u/.2)if u<.2 else mid+(goal[i]-mid)*smooth((u-.8)/.2)if u>.8 else mid
 return value

def adapt_final_stage_source(source):
 old="   if goalhand is not None:\n    handcmd=(1-u)*h0+u*np.asarray(goalhand)"
 if source.count(old)!=1:raise ValueError('Empty-ready hand interpolation anchor must occur once')
 new=old+"\n    if name=='return_to_original_ready_joints' and HUMAN_POUR_PROFILE.get('v7_empty_ready_clearance',False):\n     from t1_empty_hand_ready_v7 import safe_ready_hand_target\n     handcmd=safe_ready_hand_target(m,hj,h0,goalhand,(k+1)*m.opt.timestep/duration)"
 source=source.replace(old,new,1)
 lines=source.splitlines();matches=[line for line in lines if line.startswith('  if goalarm is not None and ')and 'cartesian_transport_phase(name)'in line]
 if len(matches)!=1:raise ValueError('Empty retreat Cartesian phase anchor must occur once')
 source=source.replace(matches[0],matches[0].replace('cartesian_transport_phase(name)',"(cartesian_transport_phase(name) or name=='empty_hand_clearance_before_ready')"),1)
 anchor="phase('human_pour_idle_restore_extend',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)));phase('return_to_original_ready_joints'"
 if source.count(anchor)!=1:raise ValueError('Post-restore original-ready anchor must occur once')
 new="phase('human_pour_idle_restore_extend',float(V6_RUNTIME_PROFILE.get('idle_segment_s',1.2)))\n  if HUMAN_POUR_PROFILE.get('v7_empty_ready_clearance',False):\n   empty_P=d.xpos[palm].copy()+np.array([-.10,0.,0.]);empty_R=d.xmat[palm].reshape(3,3).copy();phase('empty_hand_clearance_before_ready',1.5,goal(empty_P,empty_R))\n  phase('return_to_original_ready_joints'"
 source=source.replace(anchor,new,1);compile(source,'<safe_empty_ready_v7>','exec');return source

def adapt_workflow_launch_source(source,profile=None):
 from t1_actual_frame_return_v7 import adapt_workflow_launch_source as previous
 source=previous(source,profile=profile)
 anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
 if source.count(anchor)!=1:raise ValueError('Final stage snapshot anchor must occur once')
 replacement="from t1_empty_hand_ready_v7 import adapt_final_stage_source as adapt_empty_ready\\n        source=adapt_empty_ready(source)\\n        "+anchor
 result=source.replace(anchor,replacement,1);compile(result,'<v7_empty_ready_outer>','exec');return result
