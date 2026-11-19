"""Reagent identification and horizontal glass-rim fitting from RGB cameras."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import differential_evolution

def capture(cb):
    tubes=cb.robo.segment('head',['test tube'],min_score=.15)
    liquids=cb.robo.segment('head',['orange liquid','tube with red liquid'],min_score=.08)
    if not tubes:
        return None
    ranked=[]
    for t in tubes:
        m=t['mask']; ys,xs=np.nonzero(m)
        if len(xs)<50: continue
        box=np.array(t['box'],dtype=float);h=box[3]-box[1]
        if h<25: continue
        best=0.
        for l in liquids:
            lm=l['mask']; inter=np.count_nonzero(m&lm)
            overlap=inter/max(1,np.count_nonzero(lm))
            # A liquid portion must lie below the mouth, and inside a taller tube.
            fraction=(l['centroid'][1]-box[1])/max(h,1)
            score=overlap*float(l['score'])*(1+.3*float(t['score']))
            if fraction>.25 and inter>30 and h>1.15*(l['box'][3]-l['box'][1]):
                best=max(best,score)
        if best>0: ranked.append((best,t))
    obs=cb.robo.observe(cameras=['head']);cal=cb.robo.calibration()['head']
    b=obs.base; yaw=b['yaw'];c=np.cos(yaw);s=np.sin(yaw);B=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    C=B@np.array(cal['pos'])+np.array([b['x'],b['y'],b['z']]);R=B@np.array(cal['rot']);K=np.array(cal['K'])
    if ranked:
        _,tube=max(ranked,key=lambda item:item[0])
    else:
        # The reagent occupies the leftmost rack hole. Use a common mouth plane
        # to compare tube candidates, rather than each locator's guessed height.
        plausible=[]
        for t in tubes:
            box0=np.array(t['box'],float);ys0,xs0=np.nonzero(t['mask'])
            if len(xs0)<50 or box0[3]-box0[1]<35 or box0[3]-box0[1]<.95*(box0[2]-box0[0]):continue
            top0=ys0<ys0.min()+max(4,.13*(box0[3]-box0[1]))
            uv0=np.array([np.median(xs0[top0]),ys0.min()+.08*(box0[3]-box0[1])])
            ray=R@np.linalg.solve(K,np.r_[uv0,1.]);z=cb.to_room([0,0,cb.table_z])[2]+.195
            xyz=C+ray*((z-C[2])/ray[2])
            widthpx=np.ptp(xs0[top0]);span=widthpx*np.linalg.norm(xyz-C)/K[0,0]
            if .33<xyz[0]<1.06 and .012<span<.055:plausible.append((xyz[1],t))
        if not plausible:return None
        _,tube=max(plausible,key=lambda item:item[0])
        print('rim_identity_fallback',tube['box'],flush=True)
    box=np.array(tube['box'],float)
    gray=np.asarray(obs.images['head'],float).mean(axis=2)/255.
    dog=gaussian_filter(gray,.55)-gaussian_filter(gray,2.)
    mask=tube['mask'];ys,xs=np.nonzero(mask);ymin=ys.min();top=ys<ymin+max(4,.13*(box[3]-box[1]))
    uv=np.array([np.median(xs[top]),ymin+.08*(box[3]-box[1])])
    print('rim_detection',box.tolist(),uv.tolist(),float(tube['score']),flush=True)
    return {'C':C,'R':R,'K':K,'dog':dog,'box':box,'uv':uv,'base':b,'semantic_identity':bool(ranked)}

def fit(views, table_room, with_score=False):
    if len(views)>1 and np.linalg.norm(views[0]['C']-views[-1]['C'])<.03:
        views=[max(views,key=lambda v:v.get('semantic_identity',False))]
    v=views[-1];K=v['K'];ray=v['R']@np.linalg.solve(K,np.r_[v['uv'],1.])
    z=table_room+.195;seed=v['C']+ray*((z-v['C'][2])/ray[2])
    theta=np.linspace(0,2*np.pi,128,endpoint=False);cs=np.c_[np.cos(theta),np.sin(theta),np.zeros(128)]
    def objective(q):
        center=np.asarray(q[:3]);radius=q[3];pts=center+radius*cs;score=0.
        for w in views:
            pc=(pts-w['C'])@w['R'];uv=pc@w['K'].T;uv=uv[:,:2]/uv[:,2:]
            if np.any(pc[:,2]<.05):return 1000.
            box=w['box'];mid=uv.mean(axis=0)
            if np.linalg.norm(mid-w['uv'])>10.:
                return 100.+np.linalg.norm(mid-w['uv'])
            # Fit only the mouth of this candidate, not the liquid/rack below it.
            if mid[0]<box[0]-8 or mid[0]>box[2]+8 or mid[1]<box[1]-6 or mid[1]>box[1]+.32*(box[3]-box[1]):
                return 100.+np.linalg.norm(mid-w['uv'])
            vals=map_coordinates(w['dog'],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=-.1)
            score+=np.mean(vals)
        return -score
    z_bounds=(table_room+.17,table_room+.215) if len(views)>1 else (table_room+.194,table_room+.196)
    bounds=[(seed[0]-.035,seed[0]+.035),(seed[1]-.035,seed[1]+.035),z_bounds,(.011,.014)]
    opt=differential_evolution(objective,bounds,seed=8,popsize=16,maxiter=130,tol=.0001,polish=True)
    print('rim_fit',opt.x.tolist(),'edge_score',-float(opt.fun),flush=True)
    center = opt.x[:3] if opt.fun<-.001 else None
    return (center, -float(opt.fun)) if with_score else center
