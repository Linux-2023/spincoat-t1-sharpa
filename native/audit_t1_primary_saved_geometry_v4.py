"""Read-only native hand/tube geometry review of saved states. No mj_step."""
import argparse,json,hashlib
from pathlib import Path
import mujoco,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--states',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();mp=Path(a.model);sp=Path(a.states);op=Path(a.out);m=mujoco.MjModel.from_binary_path(str(mp));z=np.load(sp);d=mujoco.MjData(m);names=[m.geom(i).name for i in range(m.ngeom)];tube=set(i for i,n in enumerate(names) if n.startswith('prep/reagent_tube_'));walls=set(i for i,n in enumerate(names) if n.startswith('prep/reagent_tube_wall_'));hand=set(i for i,n in enumerate(names) if n.startswith('sharpa_left/'));finger={i:next((f for f in ['thumb','index','middle','ring','pinky'] if n.startswith('sharpa_left/left_'+f+'_')),None) for i,n in enumerate(names)};pads=set(i for i,n in enumerate(names) if n.startswith('sharpa_left/') and 'elastomer' in n);tb=m.body('prep/reagent_tube').id;events=[];maxdepth=0.;maxwall=0.;maxdot=-1.;mindot=1.
for k,t in enumerate(z['time']):
 d.qpos[:]=z['qpos'][k];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);axis=d.xmat[tb].reshape(3,3)[:,2];center=d.xpos[tb];phase=str(z['phase'][k]) if 'phase' in z else ''
 for c in d.contact:
  aa,b=map(int,c.geom);fa,fb=finger[aa],finger[b]
  if c.dist>=0:continue
  if fa and fb and fa!=fb:events.append(dict(time=float(t),phase=phase,kind='crossdigit',pair=[names[aa],names[b]],depth_mm=-float(c.dist)*1000))
  if aa in tube or b in tube:
   h=b if aa in tube else aa;tg=aa if aa in tube else b
   if h not in hand:continue
   dep=-float(c.dist)*1000;maxdepth=max(maxdepth,dep)
   if h not in pads:events.append(dict(time=float(t),phase=phase,kind='nonpad_tube',pair=[names[aa],names[b]],depth_mm=dep))
   if tg in walls:
    maxwall=max(maxwall,dep);rad=c.pos-center;rad-=axis*(rad@axis)
    if np.linalg.norm(rad)>1e-10:
     rad/=np.linalg.norm(rad);n=c.frame.reshape(3,3)[0]*(-1 if aa in tube else 1);dot=float(n@rad);maxdot=max(maxdot,dot);mindot=min(mindot,dot)
     if dot>0:events.append(dict(time=float(t),phase=phase,kind='wall_normal_outward_flip',pair=[names[aa],names[b]],depth_mm=dep,normal_radial_dot=dot))
    if dep>1.1:events.append(dict(time=float(t),phase=phase,kind='wall_penetration_exceeds_full_thickness',pair=[names[aa],names[b]],depth_mm=dep))
report=dict(scope='STATIC native geometry review of this model and saved qpos only. No physics continuation, control writes, or modification of original report/video. These are geometric contact distances, not measured contact force. Saved-pose coverage cannot exclude between-frame events.',model=str(mp),states=str(sp),saved_pose_count=len(z['time']),time_range_s=[float(z['time'][0]),float(z['time'][-1])],source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [mp,sp]},crossdigit_entries=sum(e['kind']=='crossdigit' for e in events),nonpad_tube_entries=sum(e['kind']=='nonpad_tube' for e in events),wall_innerface_normal_flip_entries=sum(e['kind']=='wall_normal_outward_flip' for e in events),wall_full_thickness_exceed_entries=sum(e['kind']=='wall_penetration_exceeds_full_thickness' for e in events),max_hand_tube_penetration_mm=maxdepth,max_wall_penetration_mm=maxwall,inward_wall_normal_radial_dot_range=[mindot,maxdot],events=events,geometry_reject=bool(events));op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['events','source_sha256']},indent=2));print('first events',events[:3])
