"""Independent prep scene build, preview and no-grasp tube-settling test."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['T1_LAB_CONTEXT']='1';os.environ['T1_CHEMIST_COAT']='0'
from pathlib import Path
import argparse,json,numpy as np,mujoco
from PIL import Image,ImageDraw
from run_t1_aspiration_probe import build_aspiration_scene
from t1_spincoat_scene import ROOT
from t1_reagent_preparation import PreparationConfig,PreparationLedger,add_reagent_preparation

def copy_named_state(old,source,m,d):
 for j in range(old.njnt):
  n=old.joint(j).name;new=m.joint(n);nq=7 if old.jnt_type[j]==0 else 4 if old.jnt_type[j]==1 else 1;nv=6 if old.jnt_type[j]==0 else 3 if old.jnt_type[j]==1 else 1;d.qpos[new.qposadr[0]:new.qposadr[0]+nq]=source['qpos'][old.jnt_qposadr[j]:old.jnt_qposadr[j]+nq];d.qvel[new.dofadr[0]:new.dofadr[0]+nv]=source['qvel'][old.jnt_dofadr[j]:old.jnt_dofadr[j]+nv]
 for a in range(old.nu):d.ctrl[m.actuator(old.actuator(a).name).id]=source['ctrl'][a]

def build_preparation_scene(out,config=None,reference=None):
 c=config or PreparationConfig();ref=Path(reference) if reference else ROOT/'logs/spincoat/t1/integrated_neutralB01_03';s,old,d0,children=build_aspiration_scene();frozen=mujoco.MjModel.from_binary_path(str(ref/'model.mjb'))
 for name in ['stand/pipette_stand']:
  b=frozen.body(name);s.body(name).pos[:]=frozen.body_pos[b.id];s.body(name).quat[:]=frozen.body_quat[b.id]
 h=add_reagent_preparation(s,c);children.append(h['child_spec']);from preview_lab_materials_v2 import appearance_assets,lighting
 app=Path(out)/'appearance';app.mkdir(parents=True,exist_ok=True);appearance_assets(s,app);m=s.compile();lighting(m,True,neutral=True);m.opt.noslip_iterations=10;m.vis.global_.offwidth=2048;m.vis.global_.offheight=2048;d=mujoco.MjData(m);state=np.load(ref/'handoff_initial.npz');copy_named_state(frozen,state,m,d);d.qvel[:]=0;ledger=PreparationLedger(c);mujoco.mj_forward(m,d);ledger.observe(m,d,h);mujoco.mj_forward(m,d)
 return s,m,d,children,h,ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',default='scene_01');a=p.parse_args();out=ROOT/'research/hardware/reagent_preparation'/a.tag;out.mkdir(parents=True,exist_ok=False);s,m,d,children,h,ledger=build_preparation_scene(out);np.savez_compressed(out/'initial_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time);mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
 with mujoco.Renderer(m,height=900,width=1200) as renderer:
  opts=mujoco.MjvOption();opts.geomgroup[:]=[1,1,1,0,0,0];panels=[]
  for label,lookat,distance,azimuth,elevation in [('Display overview: new prep station; robot unchanged',[.60,0,1.0],2.75,50,-20),('Open 50mL tube + low holder / empty receiver',[.50,.16,.88],.60,145,-32)]:
   cam=mujoco.MjvCamera();cam.lookat[:]=lookat;cam.distance=distance;cam.azimuth=azimuth;cam.elevation=elevation;renderer.update_scene(d,camera=cam,scene_option=opts);im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im);dr.rectangle([0,0,1200,45],fill='#182737');dr.text((15,15),label,fill='white');panels.append(im)
  image=Image.new('RGB',(2400,900));image.paste(panels[0],(0,0));image.paste(panels[1],(1200,0));image.save(out/'scene_preview.png')
 # Only static asset stability, with robot held by its existing actuator controls.
 start=d.xpos[m.body(h['tube_body']).id].copy();rows=[];wheel=[a for a in range(m.nu) if m.actuator(a).name.startswith('astribot_chassis_motor_')];robotdofs=sorted({int(m.jnt_dofadr[int(m.actuator_trnid[a,0])]) for a in range(m.nu) if a not in wheel and m.actuator_trnid[a,0]>=0 and m.joint(int(m.actuator_trnid[a,0])).name.startswith(('astribot_','sharpa_'))})
 for k in range(1500):
  d.qfrc_applied[:]=0;d.qfrc_applied[robotdofs]=d.qfrc_bias[robotdofs];d.ctrl[wheel]=0;mujoco.mj_step(m,d)
  if k%20==0:rows.append((float(d.time),d.qpos.copy()))
 end=d.xpos[m.body(h['tube_body']).id].copy();axis=d.xmat[m.body(h['tube_body']).id].reshape(3,3)[:,2];report=dict(handles={k:v for k,v in h.items() if k!='child_spec'},nq=m.nq,nv=m.nv,reset_state_copy='existing joints by name; prep freejoint from new asset reset; no held seed',tube_start_world_m=start.tolist(),tube_after1p5s_world_m=end.tolist(),tube_drift_mm=float(np.linalg.norm(end-start)*1000),tube_tilt_deg=float(np.degrees(np.arccos(np.clip(axis[2],-1,1)))),ledger=ledger.audit(),preview_camera_scope='External display cameras only; workflow_workspace onboard policy camera unchanged.',grasp_or_pour_demonstrated=False,source_asset='/home/agilex/DexBioLab/assets/mjcf/object/centrifuge_50ml.xml',limitations=['Liquid visuals and ledger are proxy; no mixing homogeneity or CFD claim.','Tube mass is a fixed filled surrogate during pouring, not variable-mass mechanics.','Natural settling test is not robot pick/pour validation.']);(out/'scene_report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'settling.npz',time=np.array([r[0] for r in rows]),qpos=np.array([r[1] for r in rows]));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

