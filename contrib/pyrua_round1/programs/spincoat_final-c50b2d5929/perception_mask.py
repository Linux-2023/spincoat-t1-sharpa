# s06: Measurement dependency of the later controller derived from event 0032; no integrated rehearsal event through 0040. Source: seeds/s06/code/spinprep/perception.py
"""Fit the known 25 mm tube rim to segmented silhouettes in calibrated views."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pyrua.codebox.outcome import Outcome


def snapshot(cb):
    candidates=cb.robo.segment('head',['test tube'],min_score=.5)
    candidates=[c for c in candidates if c['area']>100]
    if not candidates:
        return None
    # The reagent tube is the leftmost of the four tubes in the rack.
    left=min(c['centroid'][0] for c in candidates)
    candidate=max((c for c in candidates if c['centroid'][0]<left+20),key=lambda c:c['score'])
    cal=cb.robo.calibration()['head']
    base=cb.base()
    return dict(mask=candidate['mask'],C=np.array(cb.to_room(cal['pos'])),
                R=Rotation.from_euler('z',base['yaw']).as_matrix()@np.array(cal['rot']),K=np.array(cal['K']))


def fit_rim(views,top_guess,radius=.0125):
    a=np.linspace(0,2*np.pi,128,endpoint=False)
    circle=np.column_stack([radius*np.cos(a),radius*np.sin(a),np.zeros(len(a))])
    first=views[-1]; yy,xx=np.where(first['mask']); ymin=yy.min()
    pix=np.array([xx[yy<=ymin+2].mean(),ymin+10,1.])
    ray=first['R']@np.linalg.solve(first['K'],pix)
    initial=first['C']+ray*(top_guess-first['C'][2])/ray[2]
    def project(v,points):
        p=(points-v['C'])@v['R'];u=p@v['K'].T
        return u[:,:2]/u[:,2:]
    terms=[]
    for v in views:
        pts=initial+circle
        uv=project(v,np.r_[pts,pts-[0,0,.05]])
        hull=ConvexHull(uv).vertices;indices=hull[hull<128]
        mask=v['mask'];edge=np.logical_xor(mask,binary_erosion(mask))
        terms.append((v,indices,distance_transform_edt(~edge)))
    def residual(point):
        parts=[]
        for v,indices,dist in terms:
            uv=project(v,point+circle[indices])
            parts.append(map_coordinates(dist,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100))
        return np.concatenate(parts)
    fit=least_squares(residual,initial,bounds=(initial-[.07,.07,.06],initial+[.07,.07,.06]),
                      diff_step=1e-4,max_nfev=200,gtol=1e-7)
    error=float(np.mean(residual(fit.x)))
    return fit.x,error


def measure_tube(cb,ctx):
    table=cb.table()
    if not table.ok:return table
    base=cb.base()
    if abs(base['y']-.4)>.02 or abs(base['x'])>.01:
        out=cb.go_to(0,.4,settle=15)
        if not out.ok:return out
    cb.locate(['test tube'],name='tube_candidate')
    before=snapshot(cb)
    if before is None:return Outcome(False,'no_mask',{'object':'tube'})
    out=cb.go_to(0,.15,settle=20)
    if not out.ok:return out
    after=snapshot(cb)
    if after is None:return Outcome(False,'no_mask',{'object':'tube'})
    top_guess=cb.to_room([0,0,cb.table_z+.1913])[2]
    point,error=fit_rim([before,after],top_guess)
    if error>2.5:return Outcome(False,'rim_fit_uncertain',{'error_pixels':error})
    ctx['mouth_room']=point.tolist()
    ctx['mouth']=np.array(cb.to_root(point))
    cb.mark('reagent',ctx['mouth'])
    print('MEASURED_RIM',ctx['mouth'].tolist(),'edge_error_pixels',error,flush=True)
    return Outcome(True,'located',{'xyz':ctx['mouth'].tolist(),'error_pixels':error})
