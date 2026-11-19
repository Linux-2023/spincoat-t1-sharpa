# s01; revised perception after the identity failure in rehearsal event 0068. Earlier bundle source is absent from code/; this revision has no confirming event through 0076. Source: seeds/s01/code/spincoat/vision.py
"""Calibrated tube-rim localization from two head views; no saved layout targets."""
import numpy as np
from scipy.optimize import least_squares

def rim_rays(cb):
    cal = cb.robo.calibration()['head']
    K = np.asarray(cal['K']); R = np.asarray(cal['rot']); c = np.asarray(cal['pos'])
    candidates = cb.robo.segment('head', ['test tube'], min_score=.2)
    liquids = cb.robo.segment('head', ['red liquid', 'orange liquid'], min_score=.1)
    if not candidates:
        candidates = cb.robo.segment('head', ['test tube with red liquid'], min_score=.08)
    valid=[]
    gravity = K @ R.T @ np.array([0.,0.,1.]); van = gravity[:2]/gravity[2]
    for candidate in candidates:
        yy,xx=np.nonzero(candidate['mask'])
        if len(xx)<60: continue
        pixels=np.stack((xx,yy),axis=1); mid=pixels.mean(axis=0)
        down=van-mid;down/=np.linalg.norm(down);across=np.array([-down[1],down[0]])
        length=np.ptp(pixels@down);width=np.ptp(pixels@across)
        if length<1.6*width or width<5:continue
        # Upper silhouette is the rim's upper half, unlike the occluded bottom.
        angles=np.linspace(0,2*np.pi,100,endpoint=False)
        dirs=np.stack((np.cos(angles),np.sin(angles)),axis=1)
        dirs=dirs[dirs@down<-.30]
        supports=np.max(pixels@dirs.T,axis=0)
        a=np.linspace(0,2*np.pi,180,endpoint=False)
        ring=np.stack((.0125*np.cos(a),.0125*np.sin(a),np.zeros_like(a)),axis=1)
        def project(x):
            pc=(ring+x-c)@R;uv=pc@K.T
            return uv[:,:2]/uv[:,2:]
        ray=R@np.linalg.solve(K,np.r_[mid,1.])
        guess=c+ray*((cb.table_z+.15-c[2])/ray[2])
        fit=least_squares(lambda x: np.max(project(x)@dirs.T,axis=0)-supports,guess,diff_step=1e-4,max_nfev=100)
        pc=R.T@(fit.x-c);uv=K@pc;uv=uv[:2]/uv[2]
        err=float(np.mean(np.abs(fit.fun)))
        if err>2 or not np.all(np.isfinite(uv)):continue
        # The fit supplies a rim image centre; stereo, not assumed radius, supplies depth.
        d=R@np.linalg.solve(K,np.r_[uv,1.]);d/=np.linalg.norm(d)
        cr=np.asarray(cb.to_room(c));dr=np.asarray(cb.to_room(c+d))-cr
        overlap = max((float(np.count_nonzero(candidate['mask'] & liq['mask'])) / max(1, np.count_nonzero(liq['mask'])) * liq['score'] for liq in liquids),default=0.)
        association = overlap + .03*candidate['score']
        valid.append((association,cr,dr,uv,err))
    print('rim_candidates',[(round(v[0],3),v[3].round(2).tolist(),round(v[4],3)) for v in valid],flush=True)
    return sorted(valid,key=lambda x:-x[0])

def triangulate(a,b):
    c1,d1=a;c2,d2=b
    t=np.linalg.lstsq(np.column_stack((d1,-d2)),c2-c1,rcond=None)[0]
    p1=c1+t[0]*d1;p2=c2+t[1]*d2
    return (p1+p2)/2,float(np.linalg.norm(p1-p2))
