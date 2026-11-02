"""Display adapter for RECORDED gravity-liquid state; never reintegrates flow.

Attach mesh at scene construction (reset), then update/upload mesh during display.
The body is a convex clipped round-bottom cavity with a horizontal free surface.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from t1_liquid_palette import LIQUID_RGBA

MESH='gravity_liquid_v2_mesh';GEOM='gravity_liquid_v2_tube'
def topology(nlat=20,nlon=40):
    vertices=[[0,0,1.]]
    for i in range(1,nlat):
        p=np.pi*i/nlat
        for j in range(nlon):
            a=2*np.pi*j/nlon;vertices.append([np.sin(p)*np.cos(a),np.sin(p)*np.sin(a),np.cos(p)])
    vertices.append([0,0,-1.]);faces=[];bottom=len(vertices)-1
    for j in range(nlon):faces.append([0,1+j,1+(j+1)%nlon])
    for i in range(nlat-2):
        for j in range(nlon):
            a=1+i*nlon+j;b=1+i*nlon+(j+1)%nlon;c=a+nlon;d=b+nlon;faces.extend([[a,c,b],[b,c,d]])
    for j in range(nlon):faces.append([bottom,1+(nlat-2)*nlon+(j+1)%nlon,1+(nlat-2)*nlon+j])
    return np.asarray(vertices),np.asarray(faces)

def attach_gravity_visuals(spec):
    directions,faces=topology()
    spec.add_mesh(name=MESH,uservert=(directions*.01).ravel().tolist(),userface=faces.ravel().tolist(),smoothnormal=False)
    spec.worldbody.add_geom(name=GEOM,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=MESH,rgba=LIQUID_RGBA,contype=0,conaffinity=0,density=0,group=2)

def liquid_vertices(state,c):
    directions,faces=topology();rotation=np.asarray(state['tube_rotation_world']);origin=np.asarray(state['tube_origin_world_m']);surface=-state['free_surface_plane_world'][3]
    R=c.tube_outer_radius_m;ri=R-c.tube_wall_thickness_m;L=c.tube_length_m
    # Interior seed from deterministic volume samples; mean remains inside convex liquid.
    z=np.linspace(R-ri+.00005,L-.00005,60);a=np.linspace(0,2*np.pi,24,endpoint=False)
    rr=np.where(z<R,np.sqrt(np.maximum(0,ri**2-(z-R)**2)),ri)
    pts=np.vstack([np.c_[np.outer(rr*f,np.cos(a)).ravel(),np.outer(rr*f,np.sin(a)).ravel(),np.repeat(z,len(a))] for f in [0.,.35,.7,.95]])
    world=pts@rotation.T+origin;wet=world[:,2]<=surface
    if not wet.any():return np.tile(origin,(len(directions),1)),faces
    seed=pts[wet].mean(axis=0)
    # Match ray tessellation to the long tube aspect ratio in LOCAL coordinates.
    dirs=directions*np.array([ri,ri,L*.5]);dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    lo=np.zeros(len(directions));hi=np.full(len(directions),.35)
    for _ in range(30):
        mid=(lo+hi)/2;p=seed+mid[:,None]*dirs
        radius2=p[:,0]**2+p[:,1]**2
        inside=(p[:,2]<=L)&(p[:,2]>=R-ri)&(radius2<=ri**2)&((p[:,2]>=R)|(radius2+(p[:,2]-R)**2<=ri**2))&((p@rotation[2]+origin[2])<=surface)
        lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
    return (seed+lo[:,None]*dirs)@rotation.T+origin,faces

class GravityLiquidReplay:
    def __init__(self,record_path,config):
        self.path=Path(record_path);self.raw=json.loads(self.path.read_text());self.states=self.raw['states'];self.times=np.asarray([s['time_s'] for s in self.states]);self.config=config;self.current=None
    def update(self,model,data,time,renderer=None):
        # Nearest previous recorded state; no synthetic parcel emissions or hydro solve.
        j=int(np.clip(np.searchsorted(self.times,time,side='right')-1,0,len(self.times)-1));self.current=self.states[j]
        for i in range(model.ngeom):
            name=model.geom(i).name or ''
            if any(t in name for t in ('tube_liquid_proxy','tube_bottom_liquid_','pour_visual_')):model.geom_rgba[i,3]=0
        try:gid=model.geom(GEOM).id
        except KeyError:raise RuntimeError('Scene must call attach_gravity_visuals(spec) before compile')
        mesh=int(model.geom_dataid[gid]);adr=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh]);vertices,faces=liquid_vertices(self.current,self.config)
        assert count==len(vertices)
        # Vertices now explicitly world coordinates; display geometry transform is identity.
        model.geom_pos[gid]=0;model.geom_quat[gid]=[1,0,0,0];model.geom_sameframe[gid]=0;model.geom_rbound[gid]=3
        model.mesh_vert[adr:adr+count]=vertices
        face_vectors=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        normals=np.zeros_like(vertices)
        for column in range(3):np.add.at(normals,faces[:,column],face_vectors)
        normals/=np.maximum(1e-16,np.linalg.norm(normals,axis=1,keepdims=True))
        # Analytic planar normals avoid radial-tessellation shading on the flat free surface.
        surface=-self.current['free_surface_plane_world'][3]
        normals[np.abs(vertices[:,2]-surface)<2e-7]=[0.,0.,1.]
        normal_adr=int(model.mesh_normaladr[mesh]);normal_num=int(model.mesh_normalnum[mesh])
        if normal_num==count:model.mesh_normal[normal_adr:normal_adr+count]=normals
        model.geom_rgba[gid]=[*LIQUID_RGBA[:3],LIQUID_RGBA[3] if self.current['tube_volume_m3']>1e-12 else 0]
        # Cup follows this same saved liquid state, including explicit offline diagnostics.
        try:
            cup=model.geom(self.config.prefix+'receiver_liquid_proxy_visual').id
            height=max(.00001,self.current['receiver_volume_m3']/(np.pi*self.config.receiver_inner_radius_m**2))
            model.geom_pos[cup,2]=self.config.receiver_bottom_z_m+height/2;model.geom_size[cup,1]=height/2
            model.geom_rgba[cup]=[*LIQUID_RGBA[:3],LIQUID_RGBA[3] if self.current['receiver_volume_m3']>1e-12 else 0]
        except KeyError:pass
        if renderer is not None:mujoco.mjr_uploadMesh(model,renderer._mjr_context,mesh)
        mujoco.mj_forward(model,data)
    def draw(self,scene):
        if self.current is None:return
        parcels=self.current['parcels'];positions=np.asarray([p['position_m'] for p in parcels]);volumes=np.asarray([p['volume_m3'] for p in parcels])
        for i,p in enumerate(parcels):
            if scene.ngeom>=scene.maxgeom:break
            radius=max(.00035,(3*p['volume_m3']/(4*np.pi))**(1/3))
            geom=scene.geoms[scene.ngeom];pos=np.asarray(p['position_m'])
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,[radius]*3,pos,np.eye(3).ravel(),np.asarray(LIQUID_RGBA,np.float32))
            # Dense ballistic parcel chain, connected only across nearby points.
            if i+1<len(parcels) and np.linalg.norm(positions[i+1]-pos)<.008:
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,pos,positions[i+1])
            scene.ngeom+=1

def draw_fluid_with_palette(draw_function):
    """Wrap spincoat fluid_render; recolor only geoms that function appends."""
    def draw(scene,*args,**kwargs):
        before=scene.ngeom;result=draw_function(scene,*args,**kwargs)
        for i in range(before,scene.ngeom):scene.geoms[i].rgba[:3]=LIQUID_RGBA[:3]
        return result
    return draw
