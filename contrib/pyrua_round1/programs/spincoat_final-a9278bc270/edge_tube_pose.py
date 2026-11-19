"""Track the held glass cylinder from RGB side-edge pairs and its visible rim."""
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates,binary_dilation
from scipy.optimize import differential_evolution
from pour_geometry import project,medial_line


def side_line(image,cam,mouth,axis,radius=.0125,mask=None,temporal=False):
    mouth=np.asarray(mouth);axis=np.asarray(axis);axis/=np.linalg.norm(axis)
    ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0,1.,0]);u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u)
    distances=np.linspace(.028,.115,48);centers=mouth-distances[:,None]*axis;uv=project(cam,centers)
    mid=uv.mean(axis=0);direction=uv[-1]-uv[0];direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
    constrained=False
    if mask is not None and not temporal:
        medial=medial_line(mask)
        if medial is not None:
            line,_=medial;n=line[:2]/np.linalg.norm(line[:2]);c=line[2]/np.linalg.norm(line[:2])
            uv=uv-(uv@n+c)[:,None]*n;mid=uv.mean(axis=0);direction=uv[-1]-uv[0];direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]]);constrained=True
    theta=np.linspace(0,2*np.pi,64,endpoint=False);radial=radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
    radii=[]
    for p,c in zip(centers,uv):radii.append(np.ptp((project(cam,p+radial)-c)@normal)/2)
    radii=np.asarray(radii);band=(6. if mask is not None else 12.) if temporal else (5. if constrained else max(20.,float(np.max(radii)*2.5)));gray=np.asarray(image,float).mean(axis=2)/255.;dog=gaussian_filter(gray,.55)-gaussian_filter(gray,2.)
    def curves(params):
        shift,angle,scale=params;c=np.cos(angle);s=np.sin(angle);R=np.array([[c,-s],[s,c]]);d=R@direction;n=np.array([-d[1],d[0]])
        center=(uv-mid)@R.T+mid+shift*n
        return center,n,d,center+scale*radii[:,None]*n,center-scale*radii[:,None]*n
    allowed=None if mask is None else binary_dilation(mask,iterations=5).astype(float)
    def evaluate(params,details=False):
        center,n,d,left,right=curves(params);valid=np.ones(len(center),bool)
        for pts in (left,right):
            valid&=(pts[:,0]>3)&(pts[:,0]<image.shape[1]-4)&(pts[:,1]>3)&(pts[:,1]<image.shape[0]-4)
            if allowed is not None:valid&=map_coordinates(allowed,[pts[:,1],pts[:,0]],order=0,mode='constant',cval=0)>.5
        if valid.sum()<12:return (1.,None) if details else 1.
        sides=[]
        for pts in (left,right):
            z=[]
            for offset in [-1.5,0,1.5]:
                sample=pts[valid]+offset*n;z.append(map_coordinates(dog,[sample[:,1],sample[:,0]],order=1,mode='constant',cval=-.1))
            sides.append(np.max(z,axis=0))
        pair=(sides[0]+sides[1])/2;score=.65*np.percentile(pair,40)+.35*np.mean(pair)
        if details:return -float(score),(center,n,d,left,right,valid,sides)
        return -float(score)
    opt=differential_evolution(evaluate,[(-band,band),(-.04,.04) if temporal else ((-.10,.10) if constrained else (-.35,.35)),(.85,1.15)],seed=13,popsize=12,maxiter=100,tol=1e-5,polish=True)
    _,data=evaluate(opt.x,True)
    if data is None:return None,{'reason':'not_visible'},None
    center,n,d,left,right,valid,sides=data;line=np.r_[n,-np.dot(n,center.mean(axis=0))]
    return line,{'contrast':-float(opt.fun),'shift_pixels':float(opt.x[0]),'angle_change':float(opt.x[1]),'radius_scale':float(opt.x[2]),'visible_samples':int(valid.sum())},(left,right)


def track(images,cams,mouth_seed,axis_seed,rim_radius=.0115,masks=None,temporal=False):
    normals=[];rhs=[];fits=[];overlays={}
    for name in ['head','left_wrist']:
        line,info,edges=side_line(images[name],cams[name],mouth_seed,axis_seed,mask=None if masks is None else masks.get(name),temporal=temporal);fits.append(info);overlays[name]=edges
        if line is None or info.get('contrast',0)<.003:return None,{'reason':'edges_uncertain','fits':fits},overlays
        n=cams[name]['R']@(cams[name]['K'].T@line);n/=np.linalg.norm(n);normals.append(n);rhs.append(np.dot(n,cams[name]['C']))
    A=np.array(normals);axis=np.cross(*normals);conditioning=np.linalg.norm(axis)
    if conditioning<.05:return None,{'reason':'parallel_planes','condition':float(conditioning),'fits':fits},overlays
    axis/=conditioning
    if np.dot(axis,axis_seed)<0:axis=-axis
    seed=np.asarray(mouth_seed);anchor=seed-A.T@np.linalg.solve(A@A.T,A@seed-np.array(rhs))
    ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0,1.,0]);u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u);theta=np.linspace(0,2*np.pi,180,endpoint=False);unit=np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v
    gray=np.asarray(images['head'],float).mean(axis=2)/255.;dog=gaussian_filter(gray,.55)-gaussian_filter(gray,2.)
    end_direction=project(cams['head'],anchor+.02*axis)-project(cams['head'],anchor);end_direction/=np.linalg.norm(end_direction)
    end_limit=None
    if masks is not None and masks.get('head') is not None:
        yy,xx=np.where(masks['head']);end_limit=float(np.quantile(np.c_[xx,yy]@end_direction,.995))
    def loss(params):
        center=anchor+params[0]*axis;ring=center+params[1]*unit;uv=project(cams['head'],ring)
        if end_limit is not None and abs(float(np.max(uv@end_direction))-end_limit)>4.:return 1.
        vals=map_coordinates(dog,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=-.1)
        return -float(np.mean(vals))
    opt=differential_evolution(loss,[(-.035,.035),(rim_radius-.0005,rim_radius+.0007)],seed=14,popsize=14,maxiter=100,tol=1e-6,polish=True)
    mouth=anchor+opt.x[0]*axis;overlays['rim']=mouth+opt.x[1]*unit
    info={'edge_fits':fits,'condition':float(conditioning),'axis_line_shift':float(np.linalg.norm(anchor-seed)),'rim_contrast':-float(opt.fun),'rim_radius':float(opt.x[1]),'mouth_shift':float(np.linalg.norm(mouth-seed)),'axis':axis.tolist(),'mouth':mouth.tolist()}
    if info['rim_contrast']<.008 or info['mouth_shift']>.04:return None,info,overlays
    return {'mouth':mouth,'axis':axis},info,overlays
