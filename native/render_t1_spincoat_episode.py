"""Render recorded T1 physics with the frozen v2 liquid replay renderer."""
import os,argparse,json,subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco,numpy as np
from scipy.interpolate import RegularGridInterpolator
from PIL import Image,ImageDraw,ImageFont
from t1_spincoat_scene import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('episode',type=Path)
    parser.add_argument('--fps',type=int,default=30);args=parser.parse_args()
    out=args.episode.parent
    with np.load(args.episode,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    meta=json.loads(str(a['metadata_json']))
    profile=meta.get('motion_profile')
    is_task_ready=bool(profile and np.linalg.norm(profile['waypoints'][0]['arm_q_rad'])>.1)
    approach_duration=float(profile['duration_s']) if profile else float(meta.get('home_approach_duration_s',0))
    os.environ['T1_LAB_CONTEXT']='1' if meta.get('lab_context_enabled') else '0'
    os.environ['T1_CHEMIST_COAT']='1' if meta.get('chemist_coat_enabled') else '0'
    s,m,d,children=scene()
    m.vis.headlight.ambient[:]=[.22]*3;m.vis.headlight.diffuse[:]=[.28]*3
    if meta.get('lab_context_enabled'):
        m.vis.headlight.ambient[:]=[.20]*3;m.vis.headlight.diffuse[:]=[.20]*3
        m.vis.quality.shadowsize=4096
        m.vis.global_.fovy=55
        m.light_castshadow[:]=False
        if 'lab_front_key' in [m.light(i).name for i in range(m.nlight)]:m.light_castshadow[m.light('lab_front_key').id]=True
    if meta.get('lab_context_enabled'):
        # Fixed-function lighting adds every source: normalize their combined
        # energy so extra room fill does not clip white robot/bench surfaces.
        m.light_diffuse[:]*=1.3/max(1.3,float(np.max(m.light_diffuse.sum(axis=0))))
        m.light_ambient[:]*=.2
    else:m.light_diffuse[:]*=.75
    targetid=m.site('spincoater/dispense_target').id;rotor=m.joint('spincoater/rotor')
    mesh=m.mesh('spincoater/liquid_surface').id;geom=m.geom('spincoater/liquid_visual').id
    start,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
    matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,m.mesh_quat[mesh]);matrix=matrix.reshape(3,3)
    mesh_local=m.mesh_vert[start:start+count]@matrix.T+m.mesh_pos[mesh]
    top=mesh_local[:,2]>-.00005
    draw_fluid=fluid_module('fluid_render').draw_fluid_v2
    options=mujoco.MjvOption();options.geomgroup[:]=[1,1,1,0,0,0]
    width,height,head=768,768,90
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    output=out/'t1_sharpa_spincoat.mp4'
    video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{width*2}x{height+head}',
        '-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(output)],stdin=subprocess.PIPE)
    ticks=round(a['time'][-1]*args.fps)+1
    try:
        with mujoco.Renderer(m,height=height,width=width,max_geom=20000) as renderer:
            for tick in range(ticks):
                t=tick/args.fps
                j=min(int(t/.02),len(a['time'])-1);k=min(j+1,len(a['time'])-1);blend=np.clip((t-a['time'][j])/.02,0,1)
                d.qpos[:]=a['qpos'][j];d.qvel[:]=(1-blend)*a['qvel'][j]+blend*a['qvel'][k]
                vel=np.zeros(m.nv);mujoco.mj_differentiatePos(m,vel,1,a['qpos'][j],a['qpos'][k]);mujoco.mj_integratePos(m,d.qpos,vel,float(blend))
                mujoco.mj_forward(m,d)
                target=d.site_xpos[targetid].copy()
                h=(1-blend)*a['film'][j]+blend*a['film'][k]
                axis=np.linspace(-.018+.018/h.shape[0],.018-.018/h.shape[0],h.shape[0])
                heights=RegularGridInterpolator((axis,axis),h,bounds_error=False,fill_value=0)(mesh_local[top][:,[1,0]])
                local=mesh_local.copy();local[top,2]=heights-.00002*np.clip(1-heights/1e-6,0,1)
                m.mesh_vert[start:start+count]=(local-m.mesh_pos[mesh])@matrix
                m.geom_rgba[geom,3]=.85 if h.max()>1e-9 else 0
                mujoco.mjr_uploadMesh(m,renderer._mjr_context,mesh)
                phase=str(a['phase'][j]);panes=[]
                for side in range(2):
                    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
                    if side==0:
                        cam.lookat[:]=[.38,-.10,.80];cam.distance=2.65;cam.azimuth=135;cam.elevation=-20
                        if meta.get('lab_context_enabled'):cam.lookat[:]=[.60,0,1.0];cam.distance=3.4;cam.azimuth=50
                    elif phase in ('retract','spin'):
                        cam.lookat[:]=target+[0,0,.025];cam.distance=.30;cam.azimuth=230;cam.elevation=-55
                    else:
                        cam.lookat[:]=[.43,-.40,1.00];cam.distance=.65;cam.azimuth=140;cam.elevation=-22
                        if is_task_ready and t<approach_duration+1.:
                            u=np.clip(t-approach_duration,0,1);u=u*u*(3-2*u)
                            cam.lookat[:]=(1-u)*np.array([.05,-.06,1.20])+u*np.array([.43,-.40,1.00])
                            cam.distance=(1-u)*1.1+u*.65;cam.azimuth=(1-u)*100+u*140;cam.elevation=(1-u)*-5+u*-22
                    renderer.update_scene(d,camera=cam,scene_option=options)
                    if meta.get('chemist_coat_enabled'):
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    draw_fluid(renderer.scene,a,j,target,float(blend))
                    release=meta.get('release')
                    # Point-drop ballistic visualization is the same impact
                    # abstraction as the frozen task, not resolved liquid impact.
                    if release:
                        released=next((e['time_s'] for e in meta['events'] if e['phase']=='retract'),None)
                        if released is not None:
                            age=t-released
                            if 0<=age<=release['flight_s']:
                                pos=np.array(release['position_m'])+age*np.array(release['velocity_m_s']);pos[2]-=.5*9.81*age**2
                                g=renderer.scene.geoms[renderer.scene.ngeom]
                                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.ones(3)*(3*30e-9/(4*np.pi))**(1/3),pos,np.eye(3).ravel(),np.array([.03,.5,.95,1],dtype=np.float32));renderer.scene.ngeom+=1
                    panes.append(renderer.render().copy())
                image=Image.new('RGB',(width*2,height+head),(17,22,29));image.paste(Image.fromarray(np.hstack(panes)),(0,head))
                draw=ImageDraw.Draw(image)
                caption_phase=phase
                if profile and t<approach_duration:
                    knots=[w['time_s'] for w in profile['waypoints']]
                    segment=min(max(int(np.searchsorted(knots,t,side='right'))-1,0),len(knots)-2)
                    if is_task_ready:caption_phase='task_ready' if segment==0 else profile['waypoints'][segment+1].get('name','approach_pipette')
                    else:caption_phase=['home','move_to_task_ready','reach_pregrasp','lower_pregrasp','approach_pipette','grasp_settle'][min(segment,5)]
                draw.text((14,8),f'T1 + Sharpa | {t:05.2f}s | {caption_phase} | measured physics replay',font=font,fill='white')
                label='Lab bench 2000 x 750 x 800 mm | grip / Cartesian IK / thumb press / retract / spin'
                draw.text((14,35),label,font=small,fill=(180,205,230))
                release=meta.get('release');rpm=d.qvel[rotor.dofadr[0]]*60/(2*np.pi)
                text=f'Rotor {rpm:.0f} rpm | '
                released_at=next((e['time_s'] for e in meta['events'] if e['phase']=='retract'),float('inf'))
                text+=f"Impact {release['error_mm']:.3f} mm (task limit 0.7 mm) | " if release and t>=released_at else 'Impact pending | '
                text+='fixed base, provisional mounts; uncalibrated reduced fluid v2'
                draw.text((14,60),text,font=small,fill=(160,190,210))
                video.stdin.write(np.asarray(image).tobytes())
                if tick in (0,ticks//2,ticks-1):image.save(out/f'preview_{tick:04d}.png')
                if tick%90==0:print('RENDER',tick,ticks,flush=True)
    finally:
        video.stdin.close()
        if video.wait()!=0:raise RuntimeError('ffmpeg failed')
    print('VIDEO_READY',output,flush=True)


if __name__=='__main__':main()
