from export_actions import load_render_stream
from native_mesh_display_lists import isolate_mesh_display_lists
"""Two external DISPLAY cameras replay saved physical-clock states; no policy inputs.

Saved model lighting/materials and the recorded episode are never overwritten.
Right camera changes use observed physical phase boundaries, not task-clock guesses.
"""
import os,argparse,json,hashlib,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import RegularGridInterpolator
from t1_spincoat_scene import fluid_module
from t1_gravity_liquid_render_v2 import GravityLiquidReplay,draw_fluid_with_palette
from t1_liquid_palette import apply_liquid_palette,LIQUID_RGBA

SIZE=960;HEAD=120
def smooth(x):
 x=float(np.clip(x,0,1));return x*x*x*(10+x*(-15+6*x))
def category(phase):
 if phase in ('v6_right_grasp_dock','v6_right_aspiration_dock'):return 'transfer'
 if phase=='v6_right_direct_cup':return 'aspiration'
 if phase.startswith('asp_'):return 'aspiration'
 if any(token in phase.lower() for token in ('view_clearance','vision_clearance','visibility_clearance','camera_clearance')) or phase.lower().startswith('preobserve_'):return 'view_clearance'
 if phase in ('spin','spin_down','rotor_stopped','finish_hold','stop_spin','final_hold'):return 'film'
 if phase in ('translate','lower','settle','thumb_above','press','retract','wait_for_motion_ready','wait_for_base_stop'):return 'precision'
 if phase in ('base_transfer','base_stopped'):return 'transfer'
 return 'pickup'
def camera_spec(kind,d,ids):
 palm=d.xpos[ids['palm']];wrist=d.xpos[ids['wrist']];tip=d.site_xpos[ids['tip']];target=d.site_xpos[ids['target']]
 if kind=='prep_body':return dict(lookat=np.array([.10,.40,.92]),distance=1.85,azimuth=50.,elevation=-12.)
 if kind.startswith('prep_'):
  palm=d.xpos[ids.get('prep_palm',ids['palm'])];wrist=d.xpos[ids.get('prep_wrist',ids['wrist'])];left_prep=ids.get('prep_side','right')=='left'
  tube=d.xpos[ids['prep_tube']];mouth=d.site_xpos[ids['prep_mouth']];receiver=ids['prep_receiver']
  rack=d.site_xpos[ids['prep_rack']] if 'prep_rack' in ids else tube
  # Initial left hand can be far above/medial to the rack; keep both in frame.
  frame_radius=max(np.linalg.norm(palm-(.4*tube+.6*rack))+.14,np.linalg.norm(tube-rack)+.12)
  pickup_distance=max(.78,min(1.05,2.1*float(frame_radius)))
  if kind=='prep_pour':return dict(lookat=.50*mouth+.35*receiver+.15*palm,distance=.52,azimuth=235.,elevation=-20.)
  if kind=='prep_return':return dict(lookat=.40*tube+.20*palm+.10*wrist+.30*rack,distance=pickup_distance,azimuth=235. if left_prep else 145.,elevation=-29.)
  return dict(lookat=.35*tube+.25*palm+.15*wrist+.25*rack,distance=pickup_distance,azimuth=235. if left_prep else 145.,elevation=-29.)
 if kind=='film':return dict(lookat=target+np.array([0,0,.008]),distance=.31,azimuth=125.,elevation=-58.)
 if kind=='precision':return dict(lookat=.55*tip+.45*target+np.array([0,0,.035]),distance=.48,azimuth=135.,elevation=-33.)
 if kind in ('transfer','view_clearance'):return dict(lookat=.50*palm+.30*tip+.20*wrist+np.array([0,0,.025]),distance=.94,azimuth=100.,elevation=-15.)
 if kind=='aspiration':return dict(lookat=.45*palm+.40*tip+.15*wrist+np.array([0,0,.015]),distance=.76,azimuth=115.,elevation=-22.)
 return dict(lookat=.45*palm+.40*tip+.15*wrist+np.array([0,0,.025]),distance=.80,azimuth=145.,elevation=-18.)
class PreparationReplay:
 """Recorded liquid proxy replay, no dynamics or mixing CFD inference."""
 def __init__(self,source,model):
  folder=source/'reagent_preparation';folder=folder if folder.exists() else source;self.path=source/'reagent_preparation.npz'
  with np.load(self.path,allow_pickle=False) as z:self.raw={k:z[k].copy() for k in z.files}
  self.times=self.raw['time'];assert len(self.times)>0 and np.all(np.diff(self.times)>=0),'Prep time must not decrease'
  self.duplicate_samples_removed=int(np.sum(np.diff(self.times)==0))
  if self.duplicate_samples_removed:
   duplicate=np.flatnonzero(np.diff(self.times)==0)
   for key in ['qpos','qvel','ctrl']:
    if key in self.raw:assert all(np.array_equal(self.raw[key][i],self.raw[key][i+1]) for i in duplicate),'Unequal physical states at duplicate prep time'
   keep=np.r_[np.diff(self.times)>0,True];count=len(self.times);self.raw={key:value[keep] if value.ndim and value.shape[0]==count else value for key,value in self.raw.items()};self.times=self.raw['time']
  config=json.loads((folder/'preparation_config.json').read_text());self.handles=json.loads((folder/'scene_handles.json').read_text())
  self.manipulation_side=self.handles.get('manipulation_side',self.handles.get('side','right'))
  if self.manipulation_side not in ('left','right'):raise ValueError('Invalid preparation manipulation_side')
  self.asset_type=self.handles.get('preparation_asset_type','legacy_50ml_centrifuge')
  self.ledger_module=self.handles.get('ledger_module','t1_reagent_preparation')
  if self.asset_type=='bimanual_traditional_glass_testtube' or self.ledger_module=='t1_bimanual_glass_preparation':
   from t1_bimanual_glass_preparation import BimanualGlassConfig as PreparationConfig,BimanualGlassLedger as PreparationLedger
   self.asset_type='bimanual_traditional_glass_testtube';self.ledger_module='t1_bimanual_glass_preparation'
  elif self.asset_type=='traditional_glass_testtube' or self.ledger_module=='t1_glass_testtube_preparation':
   from t1_glass_testtube_preparation import GlassPreparationConfig as PreparationConfig,GlassPreparationLedger as PreparationLedger
   self.asset_type='traditional_glass_testtube';self.ledger_module='t1_glass_testtube_preparation'
  elif self.ledger_module=='t1_reagent_preparation' and self.asset_type=='legacy_50ml_centrifuge':
   from t1_reagent_preparation import PreparationConfig,PreparationLedger
  else:raise ValueError('Unsupported explicit preparation asset/ledger: '+self.asset_type+'/'+self.ledger_module)
  self.config=PreparationConfig(**config);self.ledger=PreparationLedger(self.config)
  # These display-only positions change every frame; compiled SAMEFRAME_BODY
  # shortcuts otherwise leave world pour markers at their original origin.
  visual_names=[self.handles['tube_visual'],self.handles['receiver_visual']]+[self.config.prefix+'pour_visual_'+str(i) for i in range(12)]
  if self.ledger_module in ('t1_glass_testtube_preparation','t1_bimanual_glass_preparation'):visual_names.extend([self.config.prefix+'tube_bottom_liquid_'+str(i) for i in range(12)])
  for name in visual_names:model.geom_sameframe[model.geom(name).id]=0
  self.tube=int(model.body(self.handles['tube_body']).id);self.mouth=int(model.site(self.handles['tube_mouth_site']).id)
  self.source_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
 def category_at(self,time):
  if time<float(self.times[0])-1e-8 or time>float(self.times[-1])+1e-8:return None
  j=int(np.clip(np.searchsorted(self.times,time,side='right')-1,0,len(self.times)-1));phase=str(self.raw['phase'][j])
  if phase in ('crouch_before_pickup','crouch_before_pour','settle_crouched','stand_after_preparation','stand_after_pour','settle_standing'):return 'prep_body'
  if phase=='human_swirl_ellipse':return 'prep_mix'
  if phase=='coarse_carry_to_receiver':return 'prep_transport'
  if phase=='coarse_return_free_corridor':return 'prep_return'
  if any(token in phase for token in ('human_pour_approach','human_pour_tilt','human_pour_drain_wait','human_pour_return','human_pour_retreat','human_pour_idle_','tilt_tube','pour_hold','carry_above_receiver')):return 'prep_pour'
  if any(token in phase for token in ('return_above_holder','lower_into_holder','release_tube','withdraw_empty','return_to_original','settle_original','prep_end')):return 'prep_return'
  if any(token in phase for token in ('return_clearance_waypoint','raise_upright_clear_of_rack_before_return')):return 'prep_return'
  return 'prep_pickup'
 def update(self,model,data,time):
  j=int(np.clip(np.searchsorted(self.times,time,side='right')-1,0,len(self.times)-1));k=min(j+1,len(self.times)-1);dt=float(self.times[k]-self.times[j]);u=float(np.clip((time-self.times[j])/dt,0,1)) if dt else 0.
  for key in ['tube_volume_m3','receiver_volume_m3']:
   setattr(self.ledger,key,float((1-u)*self.raw[key][j]+u*self.raw[key][k]))
  mouth=(1-u)*self.raw['tube_mouth_world_m'][j]+u*self.raw['tube_mouth_world_m'][k]
  flow=bool(float(self.raw['flow_volume_m3'][j])>0 and self.times[0]<=time<=self.times[-1])
  c=self.config;inside=bool(np.linalg.norm(mouth[:2]-c.receiver_xy_m)<c.receiver_inner_radius_m-.003 and c.receiver_bottom_z_m+.006<mouth[2]<c.receiver_bottom_z_m+c.proxy_max_pour_height_above_bottom_m)
  self.ledger.update_visual(model,data,self.handles,mouth,flow,inside);mujoco.mj_forward(model,data)
 def manifest(self):
  return dict(identical_duplicate_timestamp_samples_removed_for_display=self.duplicate_samples_removed,manipulation_side=self.manipulation_side,preparation_asset_type=self.asset_type,ledger_module=self.ledger_module,config_class=type(self.config).__name__,ledger_class=type(self.ledger).__name__,source=str(self.path),sha256=self.source_sha256,physical_time_range_s=[float(self.times[0]),float(self.times[-1])],volume_sampling='Linear interpolation of recorded tube/receiver volumes; endpoint volumes held outside prep interval. Flow uses actual recorded positive step volume only during prep.',display_only_geom_transform_cache='SAMEFRAME shortcuts disabled only for animated prep liquid visuals in replay model; saved model unchanged',scope='PreparationLedger.update_visual called every display frame. Liquid volume/stream proxy, not mixing or resolved liquid CFD; no changes to policy cameras, lights, material arrays or source episode.')

def right_camera(time,index,types,changes,times,d,ids):
 kind=types[index];spec=camera_spec(kind,d,ids);change_index=int(np.searchsorted(changes,index,side='right')-1)
 start=int(changes[max(change_index,0)])
 if start>0:
  before=camera_spec(types[start-1],d,ids);u=smooth((time-float(times[start]))/.85)
  for key in spec:spec[key]=(1-u)*before[key]+u*spec[key]
 cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
 for key,value in spec.items():
  if key=='lookat':cam.lookat[:]=value
  else:setattr(cam,key,float(value))
 return cam,kind,spec
_original_camera_spec=camera_spec
_original_right_camera=right_camera
def camera_spec(kind,d,ids):
 spec=_original_camera_spec(kind,d,ids)
 if not kind.startswith('prep_'):
  spec['azimuth']=50.
  if kind=='film':spec.update(azimuth=90.,distance=.34,elevation=-58.)
  elif kind=='precision':spec.update(azimuth=90.,distance=.52,elevation=-45.)
  elif kind in ('transfer','view_clearance'):spec.update(distance=.80,elevation=-26.)
  elif kind=='aspiration':spec.update(distance=.68,elevation=-28.)
  elif kind=='pickup':spec.update(distance=.72,elevation=-25.)
 return spec

def right_camera(time,index,types,changes,times,d,ids):
 # Cut at the LEFT->RIGHT handoff: interpolating across the lab wall is unsafe.
 kind=types[index]
 ci=int(np.searchsorted(changes,index,side='right')-1)
 start=int(changes[max(ci,0)])
 if not kind.startswith('prep_') and start>0 and str(types[start-1]).startswith('prep_'):
  spec=camera_spec(kind,d,ids)
  cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
  for key,value in spec.items():
   if key=='lookat':cam.lookat[:]=value
   else:setattr(cam,key,float(value))
  return cam,kind,spec
 return _original_right_camera(time,index,types,changes,times,d,ids)

def wrist_angle(d,ids,use_prep=False):
 prefix='prep_' if use_prep and 'prep_palm' in ids else ''
 a=d.xpos[ids[prefix+'wrist']]-d.xpos[ids[prefix+'forearm']];b=d.xpos[ids[prefix+'middle']]-d.xpos[ids[prefix+'palm']]
 return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
def base_status(meta,source,override):
 if override!='auto':return override,dict(source='explicit display label',override=override)
 workflow=meta.get('workflow',{})
 if (source/'result.json').exists():workflow=json.loads((source/'result.json').read_text())
 evidence=[]
 def visit(value,path=''):
  if isinstance(value,dict):
   for key,item in value.items():
    here=path+'/'+key
    if key in ('violations','violation_samples','violations_1ms') and isinstance(item,(int,float)) and any(x in path.lower() for x in ('base','fine','readiness')):evidence.append((here,int(item)))
    visit(item,here)
  elif isinstance(value,list):
   for i,item in enumerate(value):visit(item,path+'/'+str(i))
 visit(workflow)
 return ('not-passed' if any(v>0 for _,v in evidence) else 'passed' if evidence else 'unconfirmed'),dict(numeric_evidence=evidence)
def preview_samples(arrays,meta,requested):
 end=float(arrays['time'][-1]);samples=[('initial',0.)]
 phases=arrays['phase'].astype(str);times=arrays['time']
 for name,mask in [('grip_hold',phases=='hold'),('aspiration',np.char.startswith(phases,'asp_')&np.array(['release' in p or 'immerse' in p for p in phases])),('dispense',phases=='press'),('spin',phases=='spin')]:
  indices=np.flatnonzero(mask)
  if len(indices):samples.append((name,float(times[indices[len(indices)//2]])))
 release=meta.get('release');release_at=next((float(e['time_s']) for e in meta.get('events',[]) if e['phase']=='retract'),None)
 if release and release_at is not None:samples.append(('actual_release',min(end,release_at)))
 samples.append(('final_stop',end))
 if requested:samples=[('requested_'+str(i),min(end,max(0.,t))) for i,t in enumerate(requested)]
 return samples

def main():
 p=argparse.ArgumentParser();p.add_argument('--hand-geometry-label',choices=['passed','not-passed','unconfirmed'],default='unconfirmed');p.add_argument('--gravity-record',type=Path);p.add_argument('--offline-liquid-diagnostic',action='store_true');p.add_argument('--left-actual-preview',action='store_true');p.add_argument('directory',type=Path);p.add_argument('--outdir',type=Path)
 p.add_argument('--preview-only',action='store_true');p.add_argument('--preview-times',type=float,nargs='+')
 p.add_argument('--fps',type=int,default=24);p.add_argument('--output-name',default='T1_human_workflow_v7_full.mp4')
 p.add_argument('--strict-base-label',choices=['auto','not-passed','passed','unconfirmed'],default='auto')
 p.add_argument('--provisional',action='store_true',help='Visible composition-test watermark: reference episode is not new B.')
 a=p.parse_args();
 if a.strict_base_label not in ('auto','unconfirmed',V6_AUDIT_LABELS['strict_base_status']):raise ValueError('Conflicting base display status')
 if a.hand_geometry_label not in ('unconfirmed',V6_AUDIT_LABELS['strict_hand_status']):raise ValueError('Conflicting hand display status')
 a.strict_base_label=V6_AUDIT_LABELS['strict_base_status'];a.hand_geometry_label=V6_AUDIT_LABELS['strict_hand_status']
 source=a.directory.resolve();out=(a.outdir or source/'multiview').resolve();out.mkdir(parents=True,exist_ok=True)
 if Path(a.output_name).name!=a.output_name:raise ValueError('Output name must be a filename.')
 source_hash=hashlib.sha256((source/'episode.npz').read_bytes()).hexdigest()
 with np.load(source/'episode.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 times=arrays['time'];assert np.all(np.diff(times)>0),'Episode physical timestamps must be strictly increasing'
 meta=json.loads(str(arrays['metadata_json']));m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));d=mujoco.MjData(m)
 prep=PreparationReplay(source,m) if (source/'reagent_preparation.npz').exists() else None
 if prep is None:raise RuntimeError('Human workflow requires recorded preparation states')
 if a.left_actual_preview and not a.preview_only:raise RuntimeError('LEFT ACTUAL PREVIEW is restricted to still previews')
 gravity_path=a.gravity_record or (source/'reagent_preparation'/'gravity_liquid_record.json')
 if not gravity_path.exists() and not a.gravity_record:gravity_path=source/'gravity_liquid_record.json'
 if a.gravity_record and not a.offline_liquid_diagnostic:raise RuntimeError('External liquid record requires explicit --offline-liquid-diagnostic watermark')
 if not gravity_path.exists():raise FileNotFoundError('Missing actual gravity liquid record: '+str(gravity_path))
 gravity_replay=GravityLiquidReplay(gravity_path,prep.config)
 gravity_hash=hashlib.sha256(gravity_path.read_bytes()).hexdigest()
 model_hash_before=hashlib.sha256((source/'model.mjb').read_bytes()).hexdigest()
 liquid_palette_geoms=apply_liquid_palette(m)

 preserved={key:getattr(m,key).copy() for key in ['light_pos','light_dir','light_diffuse','light_ambient','light_specular','mat_rgba','mat_shininess','mat_specular']}
 m.vis.global_.offwidth=max(SIZE,m.vis.global_.offwidth);m.vis.global_.offheight=max(SIZE,m.vis.global_.offheight);m.vis.global_.fovy=55
 options=mujoco.MjvOption();options.geomgroup[:]=[1,1,1,0,0,0]
 ids={key:int(m.body(name).id) for key,name in [('palm','sharpa_right/right_hand_C_MC'),('wrist','astribot_arm_right_link_6'),('forearm','astribot_arm_right_link_5'),('middle','sharpa_right/right_middle_MCP_VL')]}
 ids.update(tip=int(m.site('pipette/tip_site').id),target=int(m.site('spincoater/dispense_target').id));rotor_dof=int(m.joint('spincoater/rotor').dofadr[0])
 if prep:
  ids.update(prep_tube=prep.tube,prep_mouth=prep.mouth,prep_receiver=np.r_[np.asarray(prep.config.receiver_xy_m),prep.config.receiver_bottom_z_m+.03])
  side=prep.manipulation_side;ids['prep_side']=side
  ids.update({key:int(m.body(name).id) for key,name in [('prep_palm',f'sharpa_{side}/{side}_hand_C_MC'),('prep_wrist',f'astribot_arm_{side}_link_6'),('prep_forearm',f'astribot_arm_{side}_link_5'),('prep_middle',f'sharpa_{side}/{side}_middle_MCP_VL')]})
  if prep.handles.get('rack_display_center_site'):ids['prep_rack']=int(m.site(prep.handles['rack_display_center_site']).id)
 mesh=m.mesh('spincoater/liquid_surface').id;geom=m.geom('spincoater/liquid_visual').id;first,count=int(m.mesh_vertadr[mesh]),int(m.mesh_vertnum[mesh]);matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,m.mesh_quat[mesh]);matrix=matrix.reshape(3,3)
 mesh_local=m.mesh_vert[first:first+count]@matrix.T+m.mesh_pos[mesh];top=mesh_local[:,2]>-.00005;draw_fluid=draw_fluid_with_palette(fluid_module('fluid_render').draw_fluid_v2)
 released=next((float(e['time_s']) for e in meta.get('events',[]) if e['phase']=='retract'),float('inf'))
 types=np.array([(prep.category_at(float(t)) if prep else None) or category(str(x)) for t,x in zip(times,arrays['phase'])]);changes=np.r_[0,np.flatnonzero(types[1:]!=types[:-1])+1]
 left=mujoco.MjvCamera();mujoco.mjv_defaultCamera(left);left.lookat[:]=[.25,.15,.95];left.distance=2.75;left.azimuth=50.;left.elevation=-20.
 if prep and prep.manipulation_side=='left':left.lookat[:]=[.25,.20,.98];left.distance=3.10
 label,base_evidence=base_status(meta,source,a.strict_base_label);aligned=load_render_stream(out);end=aligned['terminal_time_s'];ticks=len(aligned['frame_index'])
 samples=preview_samples(arrays,meta,a.preview_times)
 if prep and not a.preview_times:
  for name,token in [('prep_grip','close_on_tube'),('prep_shake','human_swirl_ellipse'),('prep_pour','human_pour_tilt'),('prep_return','lower_into_holder')]:
   selected_prep=np.flatnonzero(np.char.find(prep.raw['phase'].astype(str),token)>=0)
   if len(selected_prep):samples.append((name,float(prep.times[selected_prep[len(selected_prep)//2]])))
 preview_tick_names={min(ticks-1,max(0,int(round(t*a.fps)))):name for name,t in samples}
 selected=sorted(set(preview_tick_names)) if a.preview_only else range(ticks)
 fontpath='/home/agilex/spincoat-t1-sharpa/research/hardware/v34_layout_optimization/speed_optimization/runs/v34_speed_r08_layout_002/_sandbox/_system/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,25);small=ImageFont.truetype(fontpath,21)
 output=out/a.output_name;stream=None;preview_records=[]
 if not a.preview_only:
  stream=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s:v','1920x1080','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(output)],stdin=subprocess.PIPE)
 try:
  with mujoco.Renderer(m,height=SIZE,width=SIZE,max_geom=20000) as renderer:
   native_mesh_layout=isolate_mesh_display_lists(m,renderer)
   for tick in selected:
    time=float(aligned['source_time_s'][tick]);j=int(np.clip(np.searchsorted(times,time,side='right')-1,0,len(times)-1));k=min(j+1,len(times)-1);dt=float(times[k]-times[j]);blend=float(np.clip((time-times[j])/dt,0,1)) if dt>0 else 0.
    d.qpos[:]=aligned['qpos'][tick];d.qvel[:]=aligned['qvel'][tick];d.ctrl[:]=aligned['action'][tick];mujoco.mj_forward(m,d)
    if prep:prep.update(m,d,time)
    gravity_replay.update(m,d,time,renderer)
    target=d.site_xpos[ids['target']].copy();h=(1-blend)*arrays['film'][j]+blend*arrays['film'][k];axis=np.linspace(-.018+.018/h.shape[0],.018-.018/h.shape[0],h.shape[0]);heights=RegularGridInterpolator((axis,axis),h,bounds_error=False,fill_value=0)(mesh_local[top][:,[1,0]])
    local=mesh_local.copy();local[top,2]=heights-.00002*np.clip(1-heights/1e-6,0,1);m.mesh_vert[first:first+count]=(local-m.mesh_pos[mesh])@matrix;m.geom_rgba[geom,3]=.85 if h.max()>1e-9 else 0;mujoco.mjr_uploadMesh(m,renderer._mjr_context,mesh)
    right,kind,spec=right_camera(time,j,types,changes,times,d,ids);frame=Image.new('RGB',(1920,1080),(17,22,29))
    for panel,cam in enumerate([left,right]):
     renderer.update_scene(d,camera=cam,scene_option=options);draw_fluid(renderer.scene,arrays,j,target,blend);gravity_replay.draw(renderer.scene)
     release=meta.get('release');age=time-released
     if release and 0<=age<=release['flight_s']:
      position=np.asarray(release['position_m'])+age*np.asarray(release['velocity_m_s']);position[2]-=.5*9.81*age*age;g=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.ones(3)*(3*30e-9/(4*np.pi))**(1/3),position,np.eye(3).ravel(),np.asarray(LIQUID_RGBA,dtype=np.float32));renderer.scene.ngeom+=1
     frame.paste(Image.fromarray(renderer.render().copy()),(panel*SIZE,HEAD))
    draw=ImageDraw.Draw(frame);phase=str(arrays['phase'][j]);rpm=float(d.qvel[rotor_dof]*30/np.pi);wrist=wrist_angle(d,ids,kind.startswith('prep_'));display_side=ids.get('prep_side','right') if kind.startswith('prep_') else 'right'
    title=V6_AUDIT_LABELS.get('presentation_title_en','T1 + Sharpa | Human workflow V7 | Diagnostic replay') if not a.provisional else 'COMPOSITION TEST: reference support04 episode, NOT new B rollout'
    if a.offline_liquid_diagnostic:title='OFFLINE LIQUID DIAGNOSTIC | saved robot poses | NOT new actual rollout'
    if a.left_actual_preview:title='LEFT ACTUAL PREVIEW | recorded left preparation only | RIGHT/coating not shown'
    draw.text((18,5),title+f' | t={time:.2f}s',font=font,fill='white')
    draw.text((18,36),f'Phase: {phase} | Operation view: {kind} | Actual {display_side} forearm-palm angle {wrist:.1f} deg | Rotor {rpm:.0f} rpm',font=small,fill=(195,218,240))
    strict={'not-passed':'NOT PASSED','passed':'passed in recorded checks','unconfirmed':'NOT CONFIRMED'}[label]
    draw.text((18,63),'Original precision base gates: '+strict+' | LEFT native geometry: '+a.hand_geometry_label.upper()+' | Scripted IK | No substrate pickup',font=small,fill=(245,187,125) if label!='passed' or a.hand_geometry_label!='passed' else (172,215,187))
    draw.text((18,89),'DISPLAY ONLY / not policy | '+V6_AUDIT_LABELS.get('residual_tool_slip_text','Tool slip: not audited')+' | Reduced liquid, uncalibrated',font=small,fill=(165,190,215))
    for x,caption in [(10,'GLOBAL / fixed external view'),(970,'OPERATION / '+kind+' / external display')]:
     draw.rectangle((x,HEAD+8,x+550,HEAD+39),fill=(18,25,33));draw.text((x+8,HEAD+10),caption,font=small,fill='white')
    draw.line((SIZE,HEAD,SIZE,1080),fill=(120,135,150),width=2)
    if stream:stream.stdin.write(np.asarray(frame).tobytes())
    if tick in preview_tick_names:
     name=f'multiview_preview_{tick:05d}_{preview_tick_names[tick]}.png';frame.save(out/name);preview_records.append(dict(file=name,time_s=time,phase=phase,actual_wrist_angle_deg=wrist,manipulation_side=display_side,rpm=rpm,right_camera={key:np.asarray(value).tolist() for key,value in spec.items()}))
    if tick%96==0:print('MULTIVIEW_RENDER',tick,ticks,flush=True)
 finally:
  if stream:
   stream.stdin.close()
   if stream.wait()!=0:raise RuntimeError('ffmpeg failed')
 assert source_hash==hashlib.sha256((source/'episode.npz').read_bytes()).hexdigest(),'Source episode changed during replay'
 if prep:assert prep.source_sha256==hashlib.sha256(prep.path.read_bytes()).hexdigest(),'Source preparation record changed during replay'
 assert all(np.array_equal(value,getattr(m,key)) for key,value in preserved.items()),'Renderer changed lighting/material arrays'
 assert gravity_hash==hashlib.sha256(gravity_path.read_bytes()).hexdigest(),'Liquid record changed during replay'
 assert model_hash_before==hashlib.sha256((source/'model.mjb').read_bytes()).hexdigest(),'Saved source model changed during replay'
 manifest=dict(native_mesh_list_isolation=native_mesh_layout,aligned_action_manifest=aligned["manifest"],source_directory=str(source),source_episode_sha256=source_hash,source_model_sha256=hashlib.sha256((source/'model.mjb').read_bytes()).hexdigest(),physical_time_range_s=[float(times[0]),end],interpolation='Exact native pre-step robot states shared with exported actions; only recorded liquid and film are interpolated for display',frame_wh=[1920,1080],fps=a.fps,output=output.name if stream else None,provisional_reference_not_B=a.provisional,strict_base_status=label,strict_base_evidence=base_evidence,left_camera=dict(lookat=np.asarray(left.lookat).tolist(),distance=float(left.distance),azimuth=float(left.azimuth),elevation=float(left.elevation),fovy=55),right_camera_categories=list(dict.fromkeys(types.tolist())),preparation_replay=prep.manifest() if prep else None,camera_transition_duration_s=.85,lighting_and_material_arrays_unchanged=True,policy_camera_configuration_unchanged=True,scope='Two EXTERNAL DISPLAY cameras, not policy inputs. Replay only; no new dynamics, pose resets or source episode edits. Reduced fluid rendering reused from frozen fluid renderer.',previews=preview_records)
 manifest['human_workflow_v7']=dict(v7_display_wrapper_sha256=V7_DISPLAY_WRAPPER_SHA,v6_camera_fix_source_sha256=V6_CAMERA_FIX_SHA,operation_camera_fix='RIGHT pickup/aspiration/transfer front-side50 degrees; precision/film lateral90 degrees; LEFT-to-RIGHT cut avoids wall interpolation',assembly_manifest_sha256=V6_ASSEMBLY_HASH,actual_audit_labels=V6_AUDIT_LABELS,all_three_new_v6_segments_included=True,left_operation_display_azimuth_deg=235.,hand_geometry_display_status=a.hand_geometry_label,inherited_v2_renderer_adapter_sha256=V2_ADAPTER_HASH,gravity_record=str(gravity_path),gravity_record_sha256=gravity_hash,liquid_sampling='Nearest previous actual recorded50Hz hydrostatic plane and ballistic parcels; no alternative liquid integration',liquid_model=gravity_replay.raw['model'],liquid_rgba=LIQUID_RGBA,liquid_palette_geom_names=liquid_palette_geoms,fluid_overlay_palette='Only liquid geoms appended by frozen fluid_render.draw_fluid_v2',blue_target_and_substrate_preserved=True,offline_counterfactual_liquid_diagnostic=a.offline_liquid_diagnostic,left_actual_preview=a.left_actual_preview,adapter_source_sha256=ADAPTER_HASH,base_renderer_source_sha256=BASE_HASH)
 (out/('multiview_preview_manifest.json' if a.preview_only else 'multiview_render_manifest.json')).write_text(json.dumps(manifest,indent=2));print('MULTIVIEW_READY',str(output if stream else out),flush=True)

if __name__=='__main__':main()
