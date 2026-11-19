"""Calibrated circle and cylinder-axis geometry using public RGB masks and robot FK."""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares


def cameras(cb, observation, calibration):
    joints=dict(zip(cb.arm_names,observation.arm_q));joints.update(zip(cb.dex_names,observation.dex_q));cb.robot.fk(joints)
    out={}
    for name in ('head','left_wrist','right_wrist'):
        c=calibration[name];K=np.asarray(c['K'])
        if name=='head':C=np.asarray(c['pos']);R=np.asarray(c['rot'])
        else:
            side=name.split('_')[0];p,Q=cb.robot.pose(cb.arms[side].palm);p=p.copy();Q=Q.copy()
            C=p+Q@np.asarray(c['offset_xyz']);R=Q@np.asarray(c['offset_rot'])
        out[name]={'C':C.copy(),'R':R.copy(),'K':K.copy()}
    return out


def project(camera,points):
    p=(np.asarray(points)-camera['C'])@camera['R'];uv=p@camera['K'].T
    return uv[...,:2]/uv[...,2:]


def cap_edge(mask,camera,center,axis):
    y,x=np.where(mask & ~binary_erosion(mask));pixels=np.c_[x,y]
    a=project(camera,np.array([center,np.asarray(center)+.02*np.asarray(axis)]));u=a[1]-a[0]
    if np.linalg.norm(u)<.3:return pixels
    u/=np.linalg.norm(u);t=pixels@u;span=float(np.ptp(t))
    return pixels[t>t.max()-max(5,.13*span)]


def circle_fit(views,masks,seed,normal,radius,z_bounds):
    seed=np.asarray(seed);normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal)
    ref=np.array([1.,0,0]) if abs(normal[0])<.9 else np.array([0,1.,0]);u=np.cross(normal,ref);u/=np.linalg.norm(u);v=np.cross(normal,u)
    angle=np.linspace(0,2*np.pi,256,endpoint=False);ring=radius*(np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*v)
    edges=[cap_edge(mask,cam,seed,normal) for mask,cam in zip(masks,views)]
    def residual(p):
        parts=[]
        for cam,edge in zip(views,edges):
            uv=project(cam,ring+p);parts.append(np.sqrt(np.min(np.sum((edge[:,None,:]-uv[None,:,:])**2,axis=2),axis=1)))
        return np.concatenate(parts)
    lo=np.r_[seed[:2]-.06,z_bounds[0]];hi=np.r_[seed[:2]+.06,z_bounds[1]]
    opt=least_squares(residual,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=600)
    return opt.x,{'rms_pixels':float(np.sqrt(np.mean(opt.fun**2))),'edge_points':[len(e) for e in edges],'radius':radius,'normal':normal.tolist()}


def medial_line(mask):
    y,x=np.where(mask);P=np.c_[x,y].astype(float);center=P.mean(axis=0);e,V=np.linalg.eigh(np.cov(P.T));direction=V[:,-1];perp=np.array([-direction[1],direction[0]])
    t=(P-center)@direction;s=(P-center)@perp;lo,hi=np.quantile(t,[.15,.85]);samples=[]
    for at in np.arange(lo,hi,2):
        w=s[(t>=at)&(t<at+2)]
        if len(w)>=4:
            a,b=np.quantile(w,[.02,.98]);samples.append((at+1,(a+b)/2,b-a))
    if len(samples)<5:return None
    samples=np.asarray(samples);width=np.median(samples[:,2]);samples=samples[(samples[:,2]>.55*width)&(samples[:,2]<1.6*width)]
    if len(samples)<5:return None
    X=np.c_[samples[:,0],np.ones(len(samples))];coef=np.linalg.lstsq(X,samples[:,1],rcond=None)[0]
    residual=samples[:,1]-X@coef;keep=np.abs(residual)<max(1.,3*np.median(np.abs(residual)))
    if keep.sum()>=5:coef=np.linalg.lstsq(X[keep],samples[keep,1],rcond=None)[0]
    point=center+coef[1]*perp;d=direction+coef[0]*perp;d/=np.linalg.norm(d)
    line=np.r_[-d[1],d[0],d[1]*point[0]-d[0]*point[1]]
    return line,float(np.sqrt(np.mean((samples[:,1]-X@coef)**2)))


def cylinder_axis(views,masks):
    normals=[];errors=[]
    for cam,mask in zip(views,masks):
        fit=medial_line(mask)
        if fit is None:return None,{'reason':'short_medial_line'}
        line,error=fit;n=cam['R']@(cam['K'].T@line);n/=np.linalg.norm(n);normals.append(n);errors.append(error)
    axis=np.cross(*normals);condition=np.linalg.norm(axis)
    if condition<.05:return None,{'reason':'parallel_planes','condition':float(condition)}
    axis/=condition
    if axis[2]<0:axis=-axis
    return axis,{'condition':float(condition),'line_rms_pixels':errors}


def nearest(candidates,camera,point):
    if not candidates:return None
    uv=project(camera,np.asarray(point))
    return min(candidates,key=lambda m:np.linalg.norm(np.asarray(m['centroid'])-uv))


def refine_mouth(image,camera,cap,axis):
    from scipy.ndimage import gaussian_filter,map_coordinates
    from scipy.optimize import differential_evolution
    cap=np.asarray(cap);axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
    ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0,1.,0]);u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u)
    angle=np.linspace(0,2*np.pi,180,endpoint=False);unit=np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*v
    gray=np.asarray(image,float).mean(axis=2)/255.;dog=gaussian_filter(gray,.55)-gaussian_filter(gray,2.)
    def loss(x):
        points=cap+x[0]*axis+x[1]*unit;uv=project(camera,points)
        values=map_coordinates(dog,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=-.1)
        return -float(np.mean(values))
    opt=differential_evolution(loss,[(.07,.105),(.0115,.013)],seed=12,popsize=12,maxiter=100,tol=1e-6,polish=True)
    mouth=cap+opt.x[0]*axis
    return mouth,{'depth':float(opt.x[0]),'radius':float(opt.x[1]),'contrast':-float(opt.fun)},mouth+opt.x[1]*unit
