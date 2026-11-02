"""Isolated appearance study. Saved motion, collision shapes and baseline stay fixed."""
import os,json,argparse,subprocess,types,sys
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
os.environ['T1_LAB_CONTEXT']='1'
os.environ['T1_CHEMIST_COAT']='0'
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from run_t1_aspiration_probe import build_aspiration_scene

def appearance_assets(s,out):
    rng=np.random.default_rng(18026);n=512
    grain=rng.normal(0,1,(n,n))
    brushed=np.repeat(rng.normal(0,1,(n,1)),n,axis=1)
    xx,yy=np.meshgrid(np.arange(n),np.arange(n))
    grout=((xx<2)|(yy<2)).astype(float)
    fields={
        'steel':(np.array([178,184,186]),1.5*grain+2.0*brushed),
        'floor':(np.array([157,167,167]),1.8*grain-15*grout),
        'wall':(np.array([223,225,219]),.65*grain),
        'epoxy':(np.array([72,83,82]),.9*grain)}
    for name,(base,noise) in fields.items():
        file=out/(name+'.png');Image.fromarray(np.clip(base+noise[:,:,None],0,255).astype('uint8')).save(file)
        s.add_texture(name='appearance_v2_'+name,type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(file))
        s.add_material(name='appearance_v2_'+name,textures=['','appearance_v2_'+name]+['']*8,
            rgba=[1,1,1,1],texuniform=True,texrepeat=[2,2] if name=='floor' else [1,1],
            specular=.32 if name=='steel' else .04,shininess=.28 if name=='steel' else .08)
    for g in s.geoms:
        name=g.name or ''
        if name=='floor':g.material='appearance_v2_floor';g.rgba[:]=[1,1,1,1]
        elif name=='lab_bench_visual_worktop' or name.startswith('lab_context_prep_counter'):
            g.material='appearance_v2_steel';g.rgba[:]=[1,1,1,1]
        elif name.startswith(('lab_context_room_','lab_context_storage_back','lab_context_storage_side')):
            g.material='appearance_v2_wall';g.rgba[:]=[1,1,1,1]
        elif name=='lab_bench_visual_cabinet':g.rgba[:]=[.66,.72,.71,1]
    for name,spec,shine in [('lab_satin_steel',.32,.28),('lab_white_polymer',.12,.14),('lab_black_rubber',.04,.06)]:
        mat=s.material(name);mat.specular=spec;mat.shininess=shine

def lighting(m,new,neutral=False):
    # Presentation must not shrink the framebuffer requested by task cameras.
    m.vis.global_.offwidth=max(1600,int(m.vis.global_.offwidth));m.vis.global_.offheight=max(1000,int(m.vis.global_.offheight));m.vis.global_.fovy=55
    m.vis.quality.shadowsize=4096;m.light_castshadow[:]=False
    if not new:
        m.light_diffuse[:]*=1.3/max(1.3,float(m.light_diffuse.sum(axis=0).max()))
        m.light_ambient[:]*=.4;m.vis.headlight.ambient[:]=.27;m.vis.headlight.diffuse[:]=.1
    else:
        # Neutral ceiling/hood illumination plus a gentle warm task light.
        # MuJoCo rasterizer: no claim of area lights, PBR or global illumination.
        for i in range(m.nlight):
            name=m.light(i).name or ''
            m.light_ambient[i]=.025
            if 'ceiling' in name:m.light_diffuse[i]=[.23,.24,.25]
            elif name=='lab_hood_light':m.light_diffuse[i]=[.26,.25,.23]
            elif name=='lab_front_key':m.light_diffuse[i]=[.24,.25,.26]
            elif name=='lab_front_fill':m.light_diffuse[i]=[.15,.16,.17]
            else:m.light_diffuse[i]=[.08,.08,.08]
            m.light_specular[i]=[.06,.06,.06]
        m.light_diffuse[:]*=1.35;m.light_cutoff[:]=85;m.light_attenuation[:]=[1,0,0]
        m.vis.headlight.ambient[:]=.30;m.vis.headlight.diffuse[:]=.12
        if neutral:
            m.light_diffuse[:]=np.mean(m.light_diffuse,axis=1)[:,None]
            m.light_ambient[:]=np.mean(m.light_ambient,axis=1)[:,None]
            m.light_specular[:]=.025
            m.vis.headlight.ambient[:]=.28;m.vis.headlight.diffuse[:]=.08
    m.light_castshadow[m.light('lab_front_key').id]=True

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--render-task',action='store_true');p.add_argument('--neutral',action='store_true');a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    old=mujoco.MjModel.from_binary_path(str(a.directory/'model.mjb'))
    s,_,_,children=build_aspiration_scene();s.body('stand/pipette_stand').pos[1]+=.4
    appearance_assets(s,a.out);new=s.compile()
    checks={k:bool(np.array_equal(getattr(old,k),getattr(new,k))) for k in
        ['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','jnt_range','jnt_qposadr','actuator_trnid']}
    checks['joint_names']=all(old.joint(i).name==new.joint(i).name for i in range(old.njnt))
    checks['dimensions']=(old.nq,old.nv,old.nu)==(new.nq,new.nv,new.nu)
    if not all(checks.values()):raise RuntimeError(checks)
    with np.load(a.directory/'episode.npz',allow_pickle=False) as z:q=z['qpos'][0].copy()
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0]
    views=[('ROOM',[.50,0,1.1],3.8,55,-17),('WORKSTATION',[.63,0,.97],1.95,30,-24),('OPERATOR',[.15,.10,1.10],1.65,30,-12)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    images=[];metrics=[]
    for mode,m in [('BEFORE',old),('ENV V2 / NEUTRAL LIGHT' if a.neutral else 'MATERIALS + LIGHT V2',new)]:
        lighting(m,m is new,neutral=a.neutral);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
        with mujoco.Renderer(m,width=960,height=640,max_geom=20000) as r:
            for label,look,dist,az,el in views:
                cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=look;cam.distance=dist;cam.azimuth=az;cam.elevation=el
                r.update_scene(d,camera=cam,scene_option=opt);pix=r.render().copy();im=Image.fromarray(pix)
                ImageDraw.Draw(im).text((15,15),mode+' | '+label,font=font,fill='orange')
                im.save(a.out/(('before' if m is old else 'after')+'_'+label.lower()+'.png'));images.append(im)
                metrics.append(dict(mode=mode,view=label,white_clip_fraction=float(np.mean(np.all(pix>=250,axis=2))),mean_rgb=pix.mean(axis=(0,1)).tolist()))
    sheet=Image.new('RGB',(2880,1280))
    for i,im in enumerate(images):sheet.paste(im,((i%3)*960,(i//3)*640))
    sheet.save(a.out/'materials_lighting_comparison.jpg',quality=94)
    d=mujoco.MjData(new);d.qpos[:]=q;mujoco.mj_forward(new,d)
    version='v3_neutral' if a.neutral else 'v2'
    video=a.out/f'lab_materials_lighting_{version}.mp4'
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x640','-r','24','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(video)],stdin=subprocess.PIPE)
    with mujoco.Renderer(new,width=960,height=640,max_geom=20000) as r:
        for k in range(12*24):
            segment=k//96;u=(k%96)/95;label,look,dist,az,el=views[segment]
            cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=look;cam.distance=dist;cam.azimuth=az+12*(u-.5);cam.elevation=el
            r.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(r.render().copy())
            ImageDraw.Draw(im).text((15,15),'MATERIAL / LIGHTING STUDY | '+label+' | static pose',font=font,fill='orange')
            proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close();assert proc.wait()==0
    (a.out/'report.json').write_text(json.dumps(dict(scope='Appearance-only preview, including static saved-pose views and optional saved-trajectory replay. Not a new task rollout. No PBR or global illumination.',lighting_profile=version,robot_source_materials_preserved=True,saved_task_replay_requested=a.render_task,source_episode=str(a.directory/'episode.npz'),known_source_motion_issue='Preclose hand-pipette contacts remain in this saved trajectory; appearance replay does not repair them.',physical_model_equality=checks,metrics=metrics,baseline_modified=False),indent=2))
    print(video,flush=True)
    if a.render_task:
        from t1_spincoat_scene import ROOT
        source=(ROOT/'scripts/render_t1_spincoat_episode.py').read_text()
        source=source.replace("out=args.episode.parent",'out=Path('+repr(str(a.out))+')')
        source=source.replace("j=min(int(t/.02),len(a['time'])-1)","j=int(np.clip(np.searchsorted(a['time'],t,side='right')-1,0,len(a['time'])-1))")
        source=source.replace("out/'t1_sharpa_spincoat.mp4'","out/"+repr(f't1_front_workflow_materials_{version}.mp4'))
        source=source.replace("    targetid=m.site", "    presentation_lighting(m,True)\n    targetid=m.site")
        source=source.replace("cam.lookat[:]=[.43,-.40,1.00];cam.distance=.65;cam.azimuth=140;cam.elevation=-22", "cam.lookat[:]=d.site_xpos[m.site('pipette/tip_site').id]+[0,0,.08];cam.distance=.70;cam.azimuth=140;cam.elevation=-22")
        source=source.replace('fixed base, provisional mounts; uncalibrated reduced fluid v2',f'saved wheel-driven rollout | appearance {version} | uncalibrated fluid')
        module=types.ModuleType('appearance_task_replay');module.__file__=str(ROOT/'scripts/render_t1_spincoat_episode.py')
        exec(compile(source,module.__file__,'exec'),module.__dict__)
        module.scene=lambda:(None,new,mujoco.MjData(new),[]);module.presentation_lighting=lambda model,enabled:lighting(model,enabled,neutral=a.neutral)
        previous=sys.argv;sys.argv=['render',str(a.directory/'episode.npz'),'--fps','24']
        try:module.main()
        finally:sys.argv=previous
if __name__=='__main__':main()


