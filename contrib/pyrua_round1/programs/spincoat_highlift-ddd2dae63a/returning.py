"""Return the empty tube to its camera-measured hole, with thumb support.

E005: the identical insertion failed without thumb IP .2 and counted return
with it. The hand then opened explicitly and retreated vertically.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter,sobel,distance_transform_edt,map_coordinates,binary_propagation
from scipy.optimize import least_squares
from pyrua.codebox.kin import Frame,Goal
from pyrua.codebox.outcome import Outcome
from pour_geometry import cameras,project
from pouring import palm,send_phase


def base_transform(base):
    return Rotation.from_rotvec([0,0,base['yaw']]).as_matrix(),np.array([base['x'],base['y'],base['z']])


def edges(image):
    gray=np.rint(np.asarray(image,float)@np.array([.299,.587,.114]))/255.
    gx=sobel(gray,axis=1,mode='nearest');gy=sobel(gray,axis=0,mode='nearest');mag=np.abs(gx)+np.abs(gy)
    angle=(np.rad2deg(np.arctan2(gy,gx))+180)%180;thin=np.zeros_like(mag,dtype=bool)
    for lo,hi,dy,dx in [(0,22.5,0,1),(157.5,180,0,1),(22.5,67.5,1,1),(67.5,112.5,1,0),(112.5,157.5,1,-1)]:
        chosen=(angle>=lo)&(angle<hi);plus=np.roll(mag,(dy,dx),(0,1));minus=np.roll(mag,(-dy,-dx),(0,1));thin|=chosen&(mag>=plus)&(mag>minus)
    weak=thin&(mag>12/255);strong=thin&(mag>30/255)
    return binary_propagation(strong,structure=np.ones((3,3)),mask=weak)



def fit_hole(images,cams,rim,table):
    angle=np.linspace(0,2*np.pi,180,endpoint=False);circle=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))];guess=np.r_[rim[:2],table+.09];dist={}
    for name in ['head','right_wrist']:
        uv=project(cams[name],guess+.022*circle);size=np.array(images[name].shape[:2][::-1]);lo=np.maximum(uv.min(0).astype(int)-10,0);hi=np.minimum(uv.max(0).astype(int)+11,size)
        if np.any(hi<=lo):return Outcome(False,'rack_hole_out_of_view')
        edge=edges(images[name]);allow=np.zeros_like(edge);allow[lo[1]:hi[1],lo[0]:hi[0]]=edge[lo[1]:hi[1],lo[0]:hi[0]]
        if allow.sum()<15:return Outcome(False,'rack_hole_edges_missing')
        dist[name]=distance_transform_edt(~allow)
    def residual(x):
        out=[]
        for name in dist:
            uv=project(cams[name],x[:3]+x[3]*circle);out.extend(map_coordinates(dist[name],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100))
        return np.array(out)
    best=None
    for dz in [.07,.09,.11]:
        x0=np.r_[rim[:2],table+dz,.016];fit=least_squares(residual,x0,bounds=(np.r_[rim[:2]-.022,table+.05,.011],np.r_[rim[:2]+.022,table+.14,.024]),diff_step=1e-4,loss='soft_l1',max_nfev=300);cost=float(np.mean(residual(fit.x)))
        if best is None or cost<best[0]:best=(cost,fit.x)
    cost,x=best;data={'xyz':x[:3].tolist(),'radius':float(x[3]),'error_pixels':cost}
    print('rack_hole',data,flush=True)
    if cost>1.5 or not .012<x[3]<.018:return Outcome(False,'rack_hole_fit_uncertain',data)
    return Outcome(True,'rack_hole_measured',data)


def measure_hole(cb,ctx):
    o=cb.robo.observe(cameras=['right_wrist']);current=cameras(cb,o,cb.robo.calibration());Rh,th=base_transform(ctx['pour_view_base']);Rn,tn=base_transform(o.base);h=ctx['pour_head_camera']
    head={'C':Rn.T@(Rh@np.asarray(h['C'])+th-tn),'R':Rn.T@Rh@np.asarray(h['R']),'K':np.asarray(h['K'])}
    out=fit_hole({'head':ctx['pour_head_image'],'right_wrist':o.images['right_wrist']},{'head':head,'right_wrist':current['right_wrist']},np.asarray(cb.to_root(ctx['rim_room'])),cb.table_z)
    if out.ok:ctx['hole_room']=cb.to_room(out.data['xyz'])
    return out


def reverse_pour(cb,ctx):
    phases=ctx['pour_plan'];points=[];names=list(phases)
    for i in range(len(names)-1,-1,-1):
        points.extend(list(reversed(phases[names[i]])));points.append(phases[names[i-1]][-1] if i else ctx['pour_start_arm'][7:])
    o=cb.robo.observe(cameras=[]);last=np.array(o.arm_cmd[7:]);rows=[];arm=cb.arms['left'];original=dict(zip(cb.arm_names,o.arm_q));original.update(zip(cb.dex_names,o.dex_q));arm.base(original)
    try:
        for p in points:
            p=np.asarray(p);delta=np.max(np.abs(p-last))
            if delta<1e-7:continue
            if delta>.35:return Outcome(False,'reverse_transition')
            n=max(2,int(np.ceil(np.max(np.abs(p-last)/arm.speed)*cb.hz)))
            for q in np.linspace(last,p,n+1)[1:]:
                arm.at(q);z,_=arm.lowest(over=cb.table_area)
                if z<cb.table_z+.015:return Outcome(False,'reverse_table')
                rows.append(q.tolist())
            last=p
    finally:arm.base(original)
    out=send_phase(cb,rows,0)
    if not out.ok:return out
    scene=ctx['pour_scene'];P=np.array(cb.to_root(scene['high_palm_room']));R=np.array(scene['high_R'])
    out=cb.move_to('left',P,frame='base',path='line',reconfigure=True)
    if not out.ok and out.reason!='off_target':return out
    return cb.hold(20)


def plan_return(cb,ctx):
    o=cb.robo.observe(cameras=[]);scene=ctx['pour_scene'];off=np.array(scene['mouth_local']);ax=np.array(scene['axis_local']);arm=cb.arms['left'];frame=Frame(np.zeros(3),[1,0,0],[0,1,0]);original=dict(zip(cb.arm_names,o.arm_q));original.update(zip(cb.dex_names,o.dex_q));arm.base(original);qstart=np.array(o.arm_cmd[7:]);arm.at(qstart);P,A,O=arm.where(frame);Rstart=np.column_stack([A,O,np.cross(A,O)]);Mstart=P+Rstart@off
    hole=np.asarray(cb.to_root(ctx['hole_room']));rim=np.asarray(cb.to_root(ctx['rim_room']));target=np.r_[hole[:2],rim[2]];Rorig=np.array(ctx['prelift_rotation']);u=Rorig@ax;v=np.cross(u,[0,0,1]);angle=np.arctan2(np.linalg.norm(v),u[2]);Ru=(Rotation.from_rotvec(v/max(np.linalg.norm(v),1e-8)*angle).as_matrix())@Rorig;hand=cb.robot.subtree(arm.palm)
    try:
        for yaw in [0,10,20,-10,30,-20]:
            Rgoal=Rotation.from_rotvec([0,0,np.deg2rad(yaw)]).as_matrix()@Ru;rv=Rotation.from_matrix(Rgoal@Rstart.T).as_rotvec();high=target+[0,0,.12];q=qstart.copy();phases={};valid=True
            for name,poses in [('above_hole',[(Mstart+(high-Mstart)*t,Rotation.from_rotvec(rv*t).as_matrix()@Rstart) for t in np.linspace(0,1,51)[1:]]),('insert',[(high+[0,0,-.12*t],Rgoal) for t in np.linspace(0,1,61)[1:]])]:
                rows=[]
                for m,R in poses:
                    p=m-R@off;before=q.copy();q,ok,pe,ae=arm.solve(frame,Goal(p,R[:,0],R[:,1]),q,steps=160);arm.at(q);z,_=arm.lowest(over=cb.table_area);hz,_=cb.robot.lowest(hand)
                    if not ok or z<cb.table_z+.015 or hz<hole[2]+.015 or np.max(np.abs(q-before))>.25:valid=False;break
                    n=max(2,int(np.ceil(np.max(np.abs(q-before)/arm.speed)*cb.hz)));rows.extend(np.linspace(before,q,n+1)[1:].tolist())
                if not valid:break
                phases[name]=rows
            if valid:return Outcome(True,'return_planned',{'phases':phases,'target':target.tolist(),'yaw':yaw})
    finally:arm.base(original)
    return Outcome(False,'return_unreachable')


def return_tube(cb,ctx):
    if cb.ended:return Outcome(False,'episode_ended')
    if not ctx.get('pour_motions_done'):return Outcome(False,'pour_not_done')
    if cb.steps_left()<800:return Outcome(False,'return_steps_short')
    p=np.array(cb.state()['right']['grasp_point']);out=cb.move_to('right',p+[0,.35,.10])
    if not out.ok and out.reason!='off_target':return out
    out=cb.hold(15)
    if not out.ok:return out
    measured=measure_hole(cb,ctx)
    p=np.array(cb.state()['right']['grasp_point']);out=cb.move_to('right',p-[0,.35,.10],path='line',reconfigure=True)
    if not out.ok and out.reason!='off_target':return out
    out=cb.hold(15)
    if not out.ok:return out
    if not measured.ok:return measured
    out=reverse_pour(cb,ctx)
    if not out.ok:return out
    # Explicitly support the empty tube before changing its load direction.
    o=cb.robo.observe(cameras=[]);q=np.array(o.dex_cmd);end=q.copy();end[list(cb.dex_names).index('left_thumb_IP')]=.2
    cb.robo.step(dex=np.linspace(q,end,11)[1:]);cb.sync();out=cb.hold(10)
    if not out.ok:return out
    out=plan_return(cb,ctx)
    if not out.ok:return out
    print('return_plan',out.data['target'],out.data['yaw'],flush=True)
    for name,rows in out.data['phases'].items():
        sent=send_phase(cb,rows,15)
        if not sent.ok:return sent
    out=cb.shape('left','pad',width=.025,fingers=['index','middle'],clearance=.025)
    if not out.ok and out.reason!='blocked':return out
    out=cb.hold(15)
    if not out.ok:return out
    P,R=palm(cb);out=cb.move_to('left',P+[0,0,.12],frame='base',path='line',reconfigure=True)
    if not out.ok and out.reason!='off_target':return out
    out=cb.hold(30)
    if out.ok:ctx['return_motions_done']=True
    return out
