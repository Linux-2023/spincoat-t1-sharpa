"""Strict replay assembly of one continuous actual v6 runtime. No dynamics or interpolation."""
from pathlib import Path
import argparse,json,hashlib,numpy as np,mujoco
ORDER=('reagent_preparation','v6_right_grasp_dock','grasp','v6_right_aspiration_dock','v6_right_direct_cup','aspiration','mobile','view_clearance','coating','finish')
ENDS={'reagent_preparation':'reagent_preparation_end','v6_right_grasp_dock':'v6_right_grasp_dock_end','grasp':'grasp_end','v6_right_aspiration_dock':'v6_right_aspiration_dock_end','v6_right_direct_cup':'v6_right_direct_cup_end','aspiration':'aspiration_end','mobile':'mobile_end','view_clearance':'view_clearance_end','finish':'coat_end'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def normalized_stage(name,a,m):
 for k in ('time','qpos','qvel'):
  if k not in a:raise ValueError(name+' missing actual '+k)
 t=np.asarray(a['time'],float);n=len(t)
 if not n or t.ndim!=1 or not np.all(np.isfinite(t)):raise ValueError(name+' invalid time')
 for k,dim in [('qpos',m.nq),('qvel',m.nv)]+([('ctrl',m.nu)] if 'ctrl' in a else []):
  if a[k].shape!=(n,dim) or not np.all(np.isfinite(a[k])):raise ValueError(name+' invalid '+k)
 if np.any(np.diff(t)<-1e-8):raise ValueError(name+' decreasing physical clock')
 duplicate=np.flatnonzero(np.diff(t)<=1e-8)
 for i in duplicate:
  for k in ('qpos','qvel','ctrl'):
   if k in a and not np.array_equal(a[k][i],a[k][i+1]):raise ValueError(name+' unequal states at duplicate time: '+k)
 keep=np.r_[True,np.diff(t)>1e-8]
 a={k:v[keep] if v.ndim and v.shape[0]==n else v for k,v in a.items()};t=a['time'];delta=np.diff(t);cadence=float(np.median(delta)) if len(delta) else float(m.opt.timestep)
 maxgap=float(delta.max()) if len(delta) else 0.
 if len(delta) and maxgap>.020+float(m.opt.timestep)+1e-8:raise ValueError(name+' unrecorded internal gap '+str(maxgap)+'s; never fill by interpolation')
 if name.startswith('v6_right_'):a['phase']=np.full(len(t),name)
 elif name=='aspiration':a['phase']=np.array(['asp_'+str(p) for p in a['aspiration_stage']])
 elif name=='mobile':a['phase']=np.array(['base_stopped' if x else 'base_transfer' for x in a['stopped_gate']])
 elif name=='grasp':
  meta=json.loads(str(a.get('metadata_json','{}')));profile=meta.get('motion_profile');origin=float(meta.get('grasp_clock_origin_s',0.));labels=[]
  for time,phase in zip(t,a['phase']):
   logical=float(time)-origin;label=str(phase)
   if profile and logical<float(profile['duration_s']):
    knots=[w['time_s'] for w in profile['waypoints']];segment=min(max(int(np.searchsorted(knots,logical,side='right'))-1,0),len(knots)-2);label='optimized_ready' if logical<.7 else profile['waypoints'][segment+1].get('name','approach')
   labels.append(label)
  a['phase']=np.array(labels)
 elif 'phase' not in a:raise ValueError(name+' missing actual phase labels')
 if len(a['phase'])!=len(t):raise ValueError(name+' phase length mismatch')
 return a,dict(recording_cadence_s=cadence,max_internal_gap_s=maxgap,identical_duplicate_samples_removed=len(duplicate))
def assemble(directory,timing_correction=None):
 directory=Path(directory).resolve();paths={n:directory/(n+'.npz') for n in ORDER};needed=list(paths.values())+[directory/'model.mjb',directory/'result.json',directory/'interior_workflow_v6_provenance.json']
 missing=[str(p.name) for p in needed if not p.exists()]
 if missing:raise FileNotFoundError('Incomplete new v6 actual rollout; no assembly: '+', '.join(missing))
 if any(p.resolve().parent!=directory for p in needed):raise ValueError('Source outside the one runtime directory')
 result=json.loads((directory/'result.json').read_text());handoffs=result.get('live_handoffs',[])
 if not handoffs:raise ValueError('Missing one-live-runtime handoff identity evidence')
 identities={(v['model_identity'],v['data_identity']) for v in handoffs}
 if len(identities)!=1:raise ValueError('Mixed model/data runtime identities')
 if not result.get('no_runtime_qpos_writes'):raise ValueError('Missing original no-runtime-pose-write declaration')
 if np.any(np.diff([v['physical_time_s'] for v in handoffs])<-1e-8):raise ValueError('Decreasing handoff clocks')
 coating=result.get('stages',{}).get('coating')
 if not coating or coating.get('time_basis')!='physical_data_time_s':raise ValueError('Actual physical-clock coating metadata required')
 m=mujoco.MjModel.from_binary_path(str(directory/'model.mjb'));stages=[];manifest=[];boundaries=[];previous=None;correction=None;corrected={}
 if timing_correction is not None:
  cp=Path(timing_correction).resolve();correction=json.loads(cp.read_text())
  if correction['model_sha256']!=sha(directory/'model.mjb') or correction['arrays_sha256']!=sha(correction['arrays_path']):raise ValueError('Timestamp-only provenance mismatch')
  for row in correction['stages']:
   if sha(directory/row['raw_path'])!=row['raw_sha256']:raise ValueError('Timestamp source raw changed')
  corrected=read(correction['arrays_path']);correction=dict(manifest_path=str(cp),manifest_sha256=sha(cp),arrays_path=correction['arrays_path'],arrays_sha256=correction['arrays_sha256'])
 for name in ORDER:
  raw=read(paths[name])
  if name in corrected:raw['time']=corrected[name]
  a,review=normalized_stage(name,raw,m);t=a['time'];entry=dict(stage=name,path=paths[name].name,sha256=sha(paths[name]),start_s=float(t[0]),end_s=float(t[-1]),frames=len(t),**review)
  if previous:
   pn,pa,pr=previous;gap=float(t[0]-pa['time'][-1]);limit=max(review['recording_cadence_s'],pr['recording_cadence_s'],.020)+float(m.opt.timestep)+1e-8
   if gap<-1e-8 or gap>limit:raise ValueError('Unrecorded or overlapping boundary '+pn+' -> '+name+': '+str(gap)+'s')
   if abs(gap)<=1e-8:
    for key in ('qpos','qvel','ctrl'):
     if key in pa and key in a and not np.array_equal(pa[key][-1],a[key][0]):raise ValueError('Mixed endpoint states at equal boundary clock '+key)
   boundaries.append(dict(before=pn,after=name,gap_s=gap,maximum_original_recording_interval_s=limit,no_synthetic_states_inserted=True))
  checkpoint=directory/('handoff_'+ENDS.get(name,'__none__')+'.npz')
  if checkpoint.exists():
   c=read(checkpoint);dt=float(c['time'])-float(t[-1]);limit=max(review['recording_cadence_s'],.020)+float(m.opt.timestep)+1e-8
   if dt<-1e-8 or dt>limit:raise ValueError(name+' checkpoint clock mismatch')
   exact=abs(dt)<=1e-8
   if exact:
    for key in ('qpos','qvel','ctrl'):
     if key in a and not np.array_equal(a[key][-1],c[key]):raise ValueError(name+' actual end disagrees with same-clock live checkpoint: '+key)
   entry['end_checkpoint']=dict(path=checkpoint.name,sha256=sha(checkpoint),time_gap_s=dt,exact_state_checked=exact)
  elif name in ENDS:raise ValueError('Missing original end handoff for '+name)
  stages.append((name,a));manifest.append(entry);previous=(name,a,review)
 shape=next((a['film'].shape[1:] for _,a in stages if 'film' in a),(64,64));chunks={k:[] for k in ('time','qpos','qvel','phase','ctrl','ctrl_available','film','fluid_time','drop_film_time_s','source_stage')}
 for name,a in stages:
  n=len(a['time'])
  for k in ('time','qpos','qvel','phase'):chunks[k].append(a[k])
  chunks['ctrl'].append(a.get('ctrl',np.full((n,m.nu),np.nan)));chunks['ctrl_available'].append(np.full(n,'ctrl' in a));chunks['source_stage'].append(np.full(n,name))
  film=a.get('film',np.zeros((n,)+shape))
  if film.shape!=(n,)+shape:raise ValueError('Inconsistent recorded film resolution')
  chunks['film'].append(film);ft=a.get('fluid_time',np.zeros(n));chunks['fluid_time'].append(ft);chunks['drop_film_time_s'].append(ft)
 arrays={k:np.concatenate(v) for k,v in chunks.items()};keep=np.r_[True,np.diff(arrays['time'])>1e-8];arrays={k:v[keep] for k,v in arrays.items()}
 for _,a in reversed(stages):
  keys=[k for k in a if k.startswith('fluid_parcel_')]
  if keys:arrays.update({k:a[k] for k in keys});break
 meta=dict(coating);events=list(coating.get('events',[]))+list(result['stages'].get('finish',{}).get('events',[]));meta.update(events=list({(float(e['time_s']),str(e['phase'])):e for e in events}.values()),fixed_base=False,lab_context_enabled=True,chemist_coat_enabled=False,motion_profile=None,workflow=result,workflow_success=result['success'],integrated_optimized_approach=True,controller='Scripted IK / physical coarse wheels / GT geometry + RGB localization; not pi0.5',no_object_pose_writes_after_reset=True,substrate_pickup=False,workflow_version='V6',strict_base_status='unconfirmed',strict_hand_status='unconfirmed')
 arrays['metadata_json']=np.array(json.dumps(meta));report=dict(scope=__doc__,record_time_correction=correction,stage_order=list(ORDER),model_sha256=sha(directory/'model.mjb'),result_sha256=sha(directory/'result.json'),provenance_sha256=sha(directory/'interior_workflow_v6_provenance.json'),assembler_sha256=sha(__file__),one_live_model_data_identity=list(next(iter(identities))),stages=manifest,boundaries=boundaries,unrecorded_gaps_hidden=False,states_interpolated=False,ctrl_missing_policy='NaN with ctrl_available=False; never invent recorded controls',labels='New actual stage names assigned only where original v6 raws have no phase; original physical timestamps unchanged')
 for name,writer in [('episode.npz',lambda f:np.savez_compressed(f,**arrays)),('stage_recording_manifest_v6.json',None)]:
  target=directory/name
  if target.exists():raise FileExistsError('Refuse overwriting existing replay artefact: '+str(target))
 with (directory/'episode.npz').open('wb') as f:np.savez_compressed(f,**arrays)
 report['episode_sha256']=sha(directory/'episode.npz');(directory/'stage_recording_manifest_v6.json').write_text(json.dumps(report,indent=2));return arrays,meta,report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--physical-time-artifact',type=Path);a=p.parse_args();_,_,r=assemble(a.directory,a.physical_time_artifact);print(json.dumps(r,indent=2))
