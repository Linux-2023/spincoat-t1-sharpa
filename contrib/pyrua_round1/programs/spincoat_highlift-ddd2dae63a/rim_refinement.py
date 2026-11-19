"""Refine a semantic circle estimate against calibrated visible rim edges."""
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from returning import edges
from pour_geometry import project


def refine_circle(images,cams,seed,radius=.0205):
    seed=np.asarray(seed);angle=np.linspace(0,2*np.pi,180,endpoint=False);ring=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))];dists=[]
    for image,cam in zip(images,cams):
        pix=project(cam,seed+radius*ring);lo=np.maximum(pix.min(0).astype(int)-9,0);hi=np.minimum(pix.max(0).astype(int)+10,np.array(image.shape[:2][::-1]));ed=edges(image);allow=np.zeros_like(ed);allow[lo[1]:hi[1],lo[0]:hi[0]]=ed[lo[1]:hi[1],lo[0]:hi[0]]
        if allow.sum()<15:return seed,{'accepted':False,'reason':'rim_edges_missing'}
        dists.append(distance_transform_edt(~allow))
    def residual(x):
        out=[]
        for cam,dist in zip(cams,dists):
            uv=project(cam,x[:3]+x[3]*ring);out.extend(map_coordinates(dist,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100))
        return np.asarray(out)
    best=None
    for dz in [0,-.005,.005]:
        fit=least_squares(residual,np.r_[seed+[0,0,dz],radius],bounds=(np.r_[seed-.012,.018],np.r_[seed+.012,.023]),loss='soft_l1',diff_step=1e-4,max_nfev=300);cost=float(np.mean(residual(fit.x)))
        if best is None or cost<best[0]:best=(cost,fit.x)
    cost,x=best;info={'accepted':bool(cost<1.2),'mean_pixels':cost,'radius':float(x[3]),'shift':float(np.linalg.norm(x[:3]-seed))}
    return (x[:3] if info['accepted'] else seed),info
