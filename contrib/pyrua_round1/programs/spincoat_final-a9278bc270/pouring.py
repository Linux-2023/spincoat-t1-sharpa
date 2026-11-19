"""Re-measured implementation of s04 E006's counted dynamic pour (step871)."""
import numpy as np
from scipy.spatial.transform import Rotation
from pyrua.codebox.outcome import Outcome
from pour_geometry import cameras,circle_fit,cylinder_axis,nearest,refine_mouth
from edge_tube_pose import track
from low_pour_plan import plan


def failure(reason,data=None):return Outcome(False,reason,data or {},0)


def palm(cb,o=None):
    if o is None:o=cb.robo.observe(cameras=[])
    q=dict(zip(cb.arm_names,o.arm_q));q.update(zip(cb.dex_names,o.dex_q));cb.robot.fk(q)
    p,R=cb.robot.pose(cb.arms['left'].palm)
    return p.copy(),R.copy()


def measure(cb,ctx):
    r=cb.robo;point=np.array(cb.state()['right']['grasp_point'])
    out=cb.move_to('right',point+[0,.15,.05])
    if not out and not (out.reason=='off_target' and out.data.get('error',1)<.02):return out,None
    cb.hold(8);o=r.observe(cameras=['head','right_wrist','left_wrist']);cams=cameras(cb,o,r.calibration())
    choices=[]
    for mask in r.segment('head',['beaker'],min_score=.15):
        box=np.array(mask['box']);uv=np.array([(box[0]+box[2])/2,box[1]+.18*(box[3]-box[1]),1.]);cam=cams['head']
        ray=cam['R']@np.linalg.solve(cam['K'],uv)
        if abs(ray[2])<1e-7:continue
        seed=cam['C']+ray*((cb.table_z+.058-cam['C'][2])/ray[2])
        if seed[1]<ctx['rim'][1]-.12:choices.append((mask['score'],mask,seed))
    if not choices:return failure('no_beaker'),None
    _,mh,seed=max(choices,key=lambda item:item[0]);mr=nearest(r.segment('right_wrist',['beaker'],min_score=.1),cams['right_wrist'],seed-[0,0,.03])
    if mr is None:return failure('no_wrist_beaker'),None
    beaker,bi=circle_fit([cams['head'],cams['right_wrist']],[mh['mask'],mr['mask']],seed,[0,0,1],.020,(cb.table_z+.035,cb.table_z+.095))
    print('BEAKER_FIT',beaker.tolist(),bi,flush=True)
    if bi['rms_pixels']>2.5:return failure('beaker_fit_uncertain',bi),None
    grip=np.array(cb.state()['left']['grasp_point']);masks=[]
    for name in ['head','left_wrist']:
        m=nearest(r.segment(name,['orange liquid','red liquid'],min_score=.08),cams[name],grip-[0,0,.09])
        if m is None:return failure('no_tube_liquid',{'camera':name}),None
        masks.append(m['mask'])
    axis,ai=cylinder_axis([cams['head'],cams['left_wrist']],masks)
    if axis is None or axis[2]<.65:return failure('tube_axis_uncertain',ai),None
    cap,ci=circle_fit([cams['head'],cams['left_wrist']],masks,grip-[0,0,.065],axis,.011,(grip[2]-.14,grip[2]-.02))
    mouth,mi,_=refine_mouth(o.images['head'],cams['head'],cap,axis)
    print('MOUTH_FIT',mouth.tolist(),ai,ci,mi,flush=True)
    if ci['rms_pixels']>2.5 or mi['contrast']<.015:return failure('mouth_fit_uncertain',mi),None
    p,R=palm(cb,o)
    scene={'beaker':beaker.tolist(),'beaker_room':list(cb.to_room(beaker)),'mouth_local':(R.T@(mouth-p)).tolist(),'axis_local':(R.T@axis).tolist(),'mouth_refine':mi,'palm_R':R.tolist()}
    cb.mark('beaker_rim',beaker);cb.mark('held_tube_mouth',mouth)
    return Outcome(True,'pour_scene_measured',{},0),scene


def carry(cb,s):
    p,R=palm(cb);beaker=np.asarray(cb.to_root(s['beaker_room']));axis=R@np.asarray(s['axis_local'])
    end_axis=Rotation.from_rotvec([-2.05,0,0]).apply(axis);direction=end_axis[:2]/np.linalg.norm(end_axis[:2]);distance=.06*np.sqrt(.12/.22)
    mouth=beaker+np.r_[-distance*direction,.22];goal=mouth-R@np.asarray(s['mouth_local'])
    out=cb.move_to('left',goal,frame='base',path='line',reconfigure=True,plan=True)
    if not out:return out
    out=cb.move_to('left',goal,frame='base',path='line',reconfigure=True)
    if not out and not (out.reason=='off_target' and out.data.get('error',1)<.02):return out
    cb.hold(15);p,R=palm(cb);error=float(np.linalg.norm(p+R@np.asarray(s['mouth_local'])-mouth))
    print('CARRY_TRACKING_ERROR',error,flush=True)
    return Outcome(error<.008,'carried' if error<.008 else 'carry_tracking_error',{'error':error},0)


def update_pose(cb,s,temporal):
    r=cb.robo;o=r.observe(cameras=['head','left_wrist']);cams=cameras(cb,o,r.calibration());p,R=palm(cb,o)
    mouth=p+R@np.array(s['mouth_local']);axis=R@np.array(s['axis_local'])
    candidate=nearest(r.segment('head',['glass tube'],min_score=.1),cams['head'],mouth-.06*axis)
    if candidate is not None:
        fill=nearest(r.segment('head',['orange liquid'],min_score=.08),cams['head'],mouth-.11*axis)
        if fill is not None:
            candidate=dict(candidate);candidate['mask']=candidate['mask']|fill['mask']
    found,info,_=track(o.images,cams,mouth,axis,s['mouth_refine']['radius'],masks={'head':candidate['mask']} if candidate is not None else None,temporal=temporal)
    print('TUBE_POSE_TRACK',found,info,flush=True)
    if found is None:return failure('tube_pose_uncertain',info)
    s['mouth_local']=(R.T@(found['mouth']-p)).tolist();s['axis_local']=(R.T@found['axis']).tolist()
    return Outcome(True,'tube_pose_measured',info,0)


def execute(cb,s,planned,names):
    if planned is None:return failure('pour_plan_refused')
    for phase in planned['phases']:
        if phase['name'] not in names:continue
        rows=phase['rows'];print('POUR_PHASE',phase['name'],len(rows),phase['settle'],flush=True)
        block=4 if phase['name']=='pour' else len(rows)
        for start in range(0,len(rows),block):
            moved=cb.robo.step(arm=rows[start:start+block]);cb.sync()
            if moved.episode_ended:return failure('episode_ended')
        cb.hold(phase['settle']);p,R=palm(cb)
        error=float(np.linalg.norm(p+R@np.asarray(s['mouth_local'])-phase['mouth']));turn=float(Rotation.from_matrix(phase['rotation'].T@R).magnitude())
        print('POUR_TRACKING',phase['name'],error,turn,flush=True)
        if phase['name']!='pour' and (error>.008 or turn>.03):return failure('pour_tracking_error',{'position':error,'rotation':turn})
    return Outcome(True,'pour_motion_complete',{},0)


def pour(cb,ctx):
    if cb.steps_left()<900:return failure('insufficient_steps_for_pour')
    measured,scene=measure(cb,ctx)
    if not measured:return measured
    ctx['pour_scene']=scene
    out=carry(cb,scene)
    if not out:return out
    out=execute(cb,scene,plan(cb,scene,height=.11),{'aim','pretilt'})
    if not out:return out
    out=update_pose(cb,scene,False)
    if not out:return out
    out=execute(cb,scene,plan(cb,scene,height=.11),{'aim','lower'})
    if not out:return out
    out=update_pose(cb,scene,True)
    if not out:return out
    out=execute(cb,scene,plan(cb,scene,height=.11,jet_profile=True),{'aim','pour'})
    ctx['pour_motion_done']=bool(out)
    return out
