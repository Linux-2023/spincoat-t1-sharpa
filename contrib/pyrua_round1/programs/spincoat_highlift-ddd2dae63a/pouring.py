"""Measured RGB geometry and s06's scored low, free-roll pour.

The original 80 mm / -30 mm pour scored on E005. The integrated
geometry variant uses -28 mm and is evaluated separately in the report.
Targets and arm commands are regenerated from current-layout measurements.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pyrua.codebox.kin import Frame,Goal
from pyrua.codebox.outcome import Outcome
from pour_geometry import cameras,project,circle_fit,cylinder_axis,nearest,refine_mouth


def palm(cb):
    o=cb.robo.observe(cameras=[])
    q=dict(zip(cb.arm_names,o.arm_q));q.update(zip(cb.dex_names,o.dex_q));q.update(cb.spec['embodiment'].get('held_joints',{}));cb.robot.fk(q)
    p,R=cb.robot.pose(cb.arms['left'].palm)
    return p.copy(),R.copy()


def measure_scene(cb,ctx):
    o=cb.robo.observe(cameras=['head','left_wrist','right_wrist']);cams=cameras(cb,o,cb.robo.calibration())
    head=cb.robo.segment('head',['beaker'],min_score=.15);choices=[]
    rim=np.asarray(cb.to_root(ctx['rim_room']))
    for m in head:
        box=np.array(m['box']);uv=np.array([(box[0]+box[2])/2,box[1]+.18*(box[3]-box[1]),1.]);cam=cams['head'];ray=cam['R']@np.linalg.solve(cam['K'],uv);z=cb.table_z+.058
        if abs(ray[2])<1e-5:continue
        p=cam['C']+ray*((z-cam['C'][2])/ray[2])
        if p[1]<rim[1]-.12:choices.append((m['score'],m,p))
    if not choices:return Outcome(False,'beaker_not_visible')
    _,mh,seed=max(choices,key=lambda item:item[0]);right=cb.robo.segment('right_wrist',['beaker'],min_score=.1);mr=nearest(right,cams['right_wrist'],seed-[0,0,.03])
    if mr is None:return Outcome(False,'beaker_not_visible_in_wrist')
    beaker,bi=circle_fit([cams['head'],cams['right_wrist']],[mh['mask'],mr['mask']],seed,[0,0,1],.020,(cb.table_z+.035,cb.table_z+.095))
    from rim_refinement import refine_circle
    beaker,refinement=refine_circle([o.images['head'],o.images['right_wrist']],[cams['head'],cams['right_wrist']],beaker)
    bi['refinement']=refinement
    grip=np.array(cb.state()['left']['grasp_point']);lm=[]
    for name in ['head','left_wrist']:
        candidates=cb.robo.segment(name,['orange liquid','red liquid'],min_score=.08);m=nearest(candidates,cams[name],grip-[0,0,.09])
        if m is None:return Outcome(False,'held_liquid_not_visible',{'camera':name})
        lm.append(m)
    axis,ai=cylinder_axis([cams['head'],cams['left_wrist']],[m['mask'] for m in lm])
    if axis is None or axis[2]<.65:return Outcome(False,'held_tube_axis_uncertain',ai)
    cap,ci=circle_fit([cams['head'],cams['left_wrist']],[m['mask'] for m in lm],grip-[0,0,.065],axis,.011,(grip[2]-.14,grip[2]-.02))
    mouth,mi,_=refine_mouth(o.images['head'],cams['head'],cap,axis)
    data={'beaker':beaker.tolist(),'axis':axis.tolist(),'cap':cap.tolist(),'mouth':mouth.tolist(),'beaker_fit':bi,'cap_fit':ci,'mouth_fit':mi,'axis_fit':ai}
    print('pour_geometry',data,flush=True)
    if bi['rms_pixels']>2.5 or ci['rms_pixels']>2.5 or mi['contrast']<.015:return Outcome(False,'pour_geometry_uncertain',data)
    if np.linalg.norm(mouth-grip)>.09:return Outcome(False,'mouth_outside_grasp_neighborhood',data)
    P,R=palm(cb);data['mouth_local']=(R.T@(mouth-P)).tolist();data['axis_local']=(R.T@axis).tolist();data['beaker_room']=cb.to_room(beaker);data['high_palm']=P.tolist();data['high_R']=R.tolist()
    data['high_palm_room']=cb.to_room(P)
    ctx['pour_head_image']=o.images['head'].copy();ctx['pour_head_camera']=cams['head'];ctx['pour_view_base']=o.base
    ctx['pour_scene']=data
    return Outcome(True,'pour_geometry_measured',data)


def plan_phases(cb,scene):
    arm=cb.arms['left'];off=np.array(scene['mouth_local']);ax=np.array(scene['axis_local']);op=np.cross(ax,[0,1,0]);op/=np.linalg.norm(op)
    full=Frame(np.zeros(3),[1,0,0],[0,1,0]);tool=Frame(off,ax,op)
    original=dict(zip(cb.arm_names,cb.q_arm));original.update(zip(cb.dex_names,cb.q_dex));arm.base(original);q=np.array(cb.cmd_arm[7:]);arm.at(q);P,R=cb.robot.pose(arm.palm);P=P.copy();R=R.copy();M=P+R@off;A=R@ax
    B=np.asarray(cb.to_root(scene['beaker_room']));hand=cb.robot.subtree(arm.palm);box=np.array([[B[0]-.03,B[1]-.03],[B[0]+.03,B[1]-.03],[B[0]+.03,B[1]+.03],[B[0]-.03,B[1]+.03]])
    rx=lambda t:Rotation.from_rotvec([t,0,0]).as_matrix();turn=rx(-1.13);Mlow=M-[0,0,.14];Alow=turn@A
    descriptors=[('tilt_safe',full,[(M+rx(t)@(P-M),rx(t)@R[:,0],rx(t)@R[:,1]) for t in np.linspace(0,-1.13,25)[1:]]),
                 ('lower',tool,[(M-[0,0,z],Alow,None) for z in np.linspace(0,.14,30)[1:]]),
                 ('shift',tool,[(Mlow+[0,y,0],Alow,None) for y in np.linspace(0,.012,11)[1:]]),
                 ('finish_turn',tool,[(Mlow+[0,.012,0],rx(t)@Alow,None) for t in list(-np.arange(1,27)*.046)+[-1.22]])]
    phases={};minz=10.
    try:
        for name,frame,goals in descriptors:
            rows=[]
            for i,(p,a,opening) in enumerate(goals):
                before=q.copy();q,ok,pe,ae=arm.solve(frame,Goal(p,a,opening),q,steps=160);arm.at(q)
                h,link=arm.lowest(over=cb.table_area);bh,_=cb.robot.lowest(hand,over=box,margin=.005)
                if not ok or h<cb.table_z+.015 or bh<B[2]+.01 or np.max(np.abs(q-before))>.25:
                    return Outcome(False,'pour_plan_refused',{'phase':name,'waypoint':i,'ik':bool(ok),'point_error':pe,'axis_error':ae,'lowest_z':h,'beaker_lowest':bh})
                n=max(2,int(np.ceil(np.max(np.abs(q-before)/arm.speed)*cb.hz)));rows.extend(np.linspace(before,q,n+1)[1:].tolist());minz=min(minz,h)
            phases[name]=rows
    finally:arm.base(original)
    return Outcome(True,'pour_planned',{'phases':phases,'minimum_z':minz})


def send_phase(cb,rows,hold):
    o=cb.robo.observe(cameras=[]);cmd=np.tile(o.arm_cmd,(len(rows),1));cmd[:,7:]=rows
    delta=np.abs(np.diff(np.vstack([o.arm_cmd,cmd]),axis=0))*cb.hz
    limits=np.array([3.5,3.5,3.5,5,10,5.5,5.5]*2)
    if np.any(delta>limits+1e-6):return Outcome(False,'pour_transition_rate')
    cb.robo.step(arm=cmd,log=True);cb.sync()
    if cb.ended:return Outcome(False,'episode_ended')
    return cb.hold(hold) if hold else Outcome(True,'motion_executed')


def pour(cb,ctx):
    if cb.ended:return Outcome(False,'episode_ended')
    if not ctx.get('high_lift_executed'):return Outcome(False,'tube_not_prepared')
    if not ctx.get('pour_view_ready'):
        p=np.array(cb.state()['right']['grasp_point']);out=cb.move_to('right',p+[0,.15,.05])
        if not out.ok and out.reason!='off_target':return out
        ctx['pour_view_ready']=True
    if 'pour_scene' not in ctx:
        out=measure_scene(cb,ctx)
        if not out.ok:return out
    scene=ctx['pour_scene']
    if not ctx.get('carried_to_pour'):
        B=np.array(cb.to_root(scene['beaker_room']));P,R=palm(cb);desired=B+[0,-.04,.22];target=desired-R@np.array(scene['mouth_local'])
        out=cb.move_to('left',target,frame='base',path='line',reconfigure=True,plan=True)
        if not out.ok:return out
        out=cb.move_to('left',target,frame='base',path='line',reconfigure=True)
        if not out.ok and out.reason!='off_target':return out
        out=cb.hold(15)
        if not out.ok:return out
        for correction in range(2):
            P,R=palm(cb);error=desired-(P+R@np.array(scene['mouth_local']))
            if np.linalg.norm(error)<.003:break
            target+=error;out=cb.move_to('left',target,frame='base',path='line',reconfigure=True)
            if not out.ok and out.reason!='off_target':return out
            out=cb.hold(15)
            if not out.ok:return out
        if np.linalg.norm(error)>.008:return Outcome(False,'carried_endpoint_uncertain',{'error':float(np.linalg.norm(error))})
        ctx['carried_to_pour']=True
    if 'pour_plan' not in ctx:
        out=plan_phases(cb,scene)
        if not out.ok:return out
        ctx['pour_start_arm']=list(cb.cmd_arm)
        ctx['pour_plan']=out.data['phases'];ctx['pour_done_phases']=[]
        print('pour_plan_rows',{k:len(v) for k,v in out.data['phases'].items()},flush=True)
    for name,rows in ctx['pour_plan'].items():
        if name in ctx['pour_done_phases']:continue
        out=send_phase(cb,rows,120 if name=='finish_turn' else 15)
        if not out.ok:return out
        ctx['pour_done_phases'].append(name)
    ctx['pour_motions_done']=True
    return Outcome(True,'pour_motions_done',{'touch':cb.touch('left').data})
