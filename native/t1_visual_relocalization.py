"""RGB-only circular substrate localization with calibrated ray/plane geometry.

Engineering baseline: known blue substrate appearance, known planar height,
provisional calibrated camera. No simulator site/segmentation/depth in detector.
"""
import os,json,argparse
from pathlib import Path
import numpy as np
from scipy import ndimage

def camera_rotation(position,lookat):
    back=np.asarray(position)-lookat;back/=np.linalg.norm(back)
    right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
    return np.column_stack([right,np.cross(back,right),back])

CAMERA_POSITION_BASE=np.array([.18,-.34,1.205])
CAMERA_ROTATION_BASE=camera_rotation(CAMERA_POSITION_BASE,np.array([.43,-.34,.772]))

def add_workspace_camera(spec):
    """Compile-time provisional camera mount. Not official head extrinsics."""
    from scipy.spatial.transform import Rotation
    return spec.body('chassis_base').add_camera(name='workflow_workspace',pos=CAMERA_POSITION_BASE,
        quat=Rotation.from_matrix(CAMERA_ROTATION_BASE).as_quat()[[3,0,1,2]],fovy=55)

def capture_relocalization(model,data,base_position_estimate,base_rotation_estimate,plane_world_z=.867,out=None):
    """Detect in RGB then transform with supplied odometry; no target GT reads."""
    import mujoco
    from PIL import Image
    bp=np.asarray(base_position_estimate);br=np.asarray(base_rotation_estimate)
    captures=[]
    for resolution in [512,2048]:
        with mujoco.Renderer(model,height=resolution,width=resolution) as renderer:
            options=mujoco.MjvOption();options.geomgroup[:]=[1,1,1,0,0,0]
            renderer.update_scene(data,camera='workflow_workspace',scene_option=options)
            rgb=renderer.render().copy()
        result=localize_substrate(rgb,CAMERA_POSITION_BASE,CAMERA_ROTATION_BASE,55,plane_world_z-bp[2])
        captures.append(dict(resolution=resolution,result=result))
        if out is not None:
            Path(out).mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(Path(out)/f'relocalize_{resolution}.png')
    best=captures[-1]['result']
    if best['valid']:
        best=dict(best,position_world_m=(bp+br@np.asarray(best['position_base_m'])).tolist())
        quality=best['boundary_rms_m']<.0005 and abs(best['radius_m']-.018)<.0015 and best['pixels']>=1000
        best['image_quality_gate']=bool(quality)
        if not quality:best.update(valid=False,reason='RGB boundary fit or known-radius consistency failed')
    return dict(measurement=best,captures=captures,scope='known blue planar substrate, provisional calibrated camera, supplied base odometry; small base tilt approximation')

def localize_substrate(rgb,camera_position,camera_rotation_matrix,fovy_deg,plane_z,radius=.018):
    pixels=rgb.astype(float);r,g,b=np.moveaxis(pixels,-1,0)
    mask=(b>r*1.5)&(g>r*1.35)&(b>g*1.13)&(b<g*1.8)&(b>22)&(r<160)
    mask=ndimage.binary_closing(mask,iterations=2)
    labels,n=ndimage.label(mask);height,width=mask.shape
    f=.5*height/np.tan(np.deg2rad(fovy_deg)/2)
    cp=np.asarray(camera_position);rot=np.asarray(camera_rotation_matrix)
    results=[]
    for i in range(1,n+1):
        component=labels==i
        if component.sum()<40:continue
        # Only exterior boundary: the pale rotation marker may make a hole.
        component=ndimage.binary_fill_holes(component)
        edge=component&~ndimage.binary_erosion(component)
        y,x=np.where(edge)
        rays=np.column_stack([(x+.5-width/2)/f,-(y+.5-height/2)/f,-np.ones(len(x))])@rot.T
        scale=(plane_z-cp[2])/rays[:,2]
        if np.any(scale<=0):continue
        points=cp+rays*scale[:,None];xy=points[:,:2]
        design=np.column_stack([2*xy,np.ones(len(xy))])
        coef=np.linalg.lstsq(design,(xy*xy).sum(axis=1),rcond=None)[0]
        center=coef[:2];fit_radius=np.sqrt(max(0,coef[2]+center@center))
        residual=float(np.sqrt(np.mean((np.linalg.norm(xy-center,axis=1)-fit_radius)**2)))
        if not .010<fit_radius<.026 or residual>.002:continue
        score=abs(fit_radius-radius)+residual*3
        results.append(dict(position_base_m=[*center,float(plane_z)],radius_m=float(fit_radius),boundary_rms_m=residual,score=score,pixels=int(component.sum()),pixel_center=[float(x.mean()),float(y.mean())]))
    if not results:return {'valid':False,'reason':'no circular blue substrate candidate'}
    results.sort(key=lambda v:v['score'])
    if len(results)>1 and results[1]['score']-results[0]['score']<.0003:return {'valid':False,'reason':'ambiguous circular candidates','candidates':results}
    return dict(valid=True,**results[0])

def main():
    os.environ['MUJOCO_GL']='egl';os.environ['T1_LAB_CONTEXT']='1'
    import mujoco
    from scipy.spatial.transform import Rotation
    from PIL import Image,ImageDraw
    from t1_spincoat_scene import scene,ROOT
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # Pose is a provisional base-mounted workspace camera, not a calibrated
    # Astribot head camera. Detector receives only this calibration and RGB.
    cp=np.array([.18,-.34,1.30]);look=np.array([.43,-.34,.867]);rot=camera_rotation(cp,look)
    rows=[]
    for dx,dy,yaw_deg in [(0,0,0),(.006,-.008,0),(-.008,.010,1.5)]:
        s,_,_,children=scene()
        yaw=Rotation.from_euler('z',yaw_deg,degrees=True).as_matrix();base=np.array([dx,dy,0.])
        camera=s.worldbody.add_camera(name='workflow_visual_probe',pos=base+yaw@cp,quat=Rotation.from_matrix(yaw@rot).as_quat()[[3,0,1,2]],fovy=55)
        m=s.compile();d=mujoco.MjData(m);m.vis.global_.offwidth=2048;m.vis.global_.offheight=2048
        # Reset to a measured, retracted preparation pose, not a simulated move.
        cfg=json.loads((ROOT/'research/hardware/task_ready_natural_01.json').read_text())
        for name,value in cfg['initial_joint_positions'].items():d.qpos[m.joint(name).qposadr[0]]=value
        m.geom_rgba[m.geom('spincoater/liquid_visual').id,3]=0
        mujoco.mj_forward(m,d)
        for resolution in [512,1024,2048]:
            with mujoco.Renderer(m,height=resolution,width=resolution) as renderer:
                options=mujoco.MjvOption();options.geomgroup[:]=[1,1,1,0,0,0]
                renderer.update_scene(d,camera='workflow_visual_probe',scene_option=options);rgb=renderer.render().copy()
            result=localize_substrate(rgb,cp,rot,55,.867)
            # Privileged target is accessed ONLY here, after detection, for audit.
            truth=yaw.T@(d.site_xpos[m.site('spincoater/dispense_target').id]-base)
            err=None if not result['valid'] else float(np.linalg.norm(np.asarray(result['position_base_m'])[:2]-truth[:2])*1000)
            row=dict(base_camera_perturbation=[dx,dy,yaw_deg],resolution=resolution,measurement=result,audit_error_mm=err)
            rows.append(row);img=Image.fromarray(rgb);ImageDraw.Draw(img).text((10,10),f'{resolution}px | error {err} mm',fill='orange')
            img.save(a.out/f'camera_{dx}_{dy}_{yaw_deg}_{resolution}.png')
    report={'scope':'static RGB localization probes only; no docking controller or full workflow validation','detector_inputs':'RGB, camera calibration relative to base, known substrate plane height and radius','assumptions':['known blue substrate appearance','unoccluded circular boundary','provisional base-mounted workspace camera, not official camera calibration','planar support height .867m'],'cases':rows}
    (a.out/'visual_relocalization_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
