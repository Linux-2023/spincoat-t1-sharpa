"""Original T1 fitted lab coat, optional visual dressing only.

Usage, before compile: ``from t1_chemist_coat import add_chemist_coat;
metadata = add_chemist_coat(spec)``. No scene/controller mutation is required.
Segmented rigid visual cloth follows the torso and arm links, not cloth dynamics.
This is an identity/appearance experiment, not validated protective equipment.
"""
from pathlib import Path
import json
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets/chemist_coat'


def add_chemist_coat(spec, export_assets=True):
    """Attach zero-mass non-colliding mesh panels to existing T1 bodies."""
    if any(g.name.startswith('chemist_coat/') for g in spec.geoms):
        raise ValueError('Coat already attached; use a fresh MjSpec')
    base=spec.compile(); d=mujoco.MjData(base); mujoco.mj_forward(base,d)
    ASSETS.mkdir(parents=True,exist_ok=True)
    mats={
        'fabric': [.89,.90,.88,1], 'fold': [.97,.975,.955,1],
        'seam': [.62,.66,.65,1], 'button':[.70,.73,.70,1],
        'badge':[.12,.29,.37,1], 'trim':[.78,.81,.79,1],
    }
    for name,color in mats.items():
        spec.add_material(name='chemist_coat/'+name,rgba=color,specular=.06,shininess=.025)
    records=[]

    def local(body,verts):
        i=base.body(body).id
        return (np.asarray(verts)-d.xpos[i])@d.xmat[i].reshape(3,3)

    def mesh(name,body,verts,faces,mat='fabric'):
        verts=local(body,verts); faces=np.asarray(faces,dtype=int)
        # Mesh thickness is explicitly modelled in each panel. Export original
        # meshes so their construction remains inspectable and reusable.
        asset='chemist_coat/'+name
        spec.add_mesh(name=asset,uservert=verts.ravel().tolist(),userface=faces.ravel().tolist(),smoothnormal=True)
        spec.body(body).add_geom(name=asset,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=asset,
            material='chemist_coat/'+mat,group=2,contype=0,conaffinity=0,mass=0,density=0)
        if export_assets:
            with (ASSETS/(name+'.obj')).open('w') as f:
                f.write('# Original parametric T1 lab coat; body-local coordinates\n')
                for p in verts: f.write('v '+' '.join(f'{x:.8f}' for x in p)+'\n')
                for face in faces: f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
        records.append({'name':asset,'body':body,'vertices':len(verts),'triangles':len(faces),'material':mat})

    def panel(name,body,grid,mat='fabric',thickness=.0012):
        g=np.asarray(grid); nr,nc,_=g.shape; v=g.reshape(-1,3)
        # A small geometric thickness makes arbitrary panel orientations robust.
        normal=np.cross(np.gradient(g,axis=0),np.gradient(g,axis=1))
        normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-12)
        verts=np.vstack([v,v+thickness*normal.reshape(-1,3)]); n=len(v); fs=[]
        for i in range(nr-1):
            for j in range(nc-1):
                a=i*nc+j;b=a+nc
                fs += [[a,b,a+1],[a+1,b,b+1],[a+n,a+1+n,b+n],[a+1+n,b+1+n,b+n]]
        border=list(range(nc))+[i*nc+nc-1 for i in range(1,nr)]+list(range((nr-1)*nc+nc-2,(nr-1)*nc-1,-1))+[i*nc for i in range(nr-2,0,-1)]
        for a,b in zip(border,border[1:]+border[:1]):fs += [[a,a+n,b],[b,a+n,b+n]]
        mesh(name,body,verts,fs,mat)

    def tube(name,body,points,radii,mat='fabric',rings=48):
        pts=np.asarray(points);g=[]
        for k,p in enumerate(pts):
            tangent=pts[min(k+1,len(pts)-1)]-pts[max(k-1,0)];tangent/=np.linalg.norm(tangent)
            ref=np.array([1.,0,0]) if abs(tangent[0])<.9 else np.array([0.,1,0])
            u=np.cross(tangent,ref);u/=np.linalg.norm(u);v=np.cross(tangent,u)
            ring=[]
            for a in np.linspace(0,2*np.pi,rings):
                r=radii[k]*(1+.018*np.sin(7*a+k*.65)+.012*np.sin(11*a-k*.8))
                ring.append(p+r*(np.cos(a)*u+np.sin(a)*v))
            g.append(ring)
        panel(name,body,g,mat)

    def seam(name,body,points,r=.00065):
        tube(name,body,points,[r]*len(points),'seam',rings=8)

    torso='astribot_torso_link_4'
    # Front is +X; leave a narrow open front and lower walking/waist vent.
    zs=np.linspace(.57,1.325,50);angles=np.linspace(.11,2*np.pi-.11,104)
    grid=[]
    for z in zs:
        f=(z-.57)/(.755)
        ry=np.interp(z,[.57,.9,1.18,1.325],[.120,.130,.145,.150])
        rx=np.interp(z,[.57,.9,1.18,1.325],[.185,.164,.160,.145])
        row=[]
        for a in angles:
            front=max(np.cos(a),0.)
            # Deep front neckline, raised back shoulder, gently draped hem.
            zz=z-.087*front**7*f**9 + .004*np.sin(5*a)*((1-f)**7)
            wave=.0024*np.sin(9*a+f*1.2)*(.25+.75*(1-f))+.0012*np.sin(17*a-f*5)
            row.append([-.030+(rx+wave)*np.cos(a),(ry+wave)*np.sin(a),zz])
        grid.append(row)
    panel('coat_body',torso,grid)
    seam('hem',torso,grid[0])
    for side,idx in [('left',0),('right',-1)]:
        seam('front_seam_'+side,torso,[r[idx] for r in grid])
    # Two sculpted folded lapels: pointed polygon panels project from chest.
    for sign,label in [(1,'left'),(-1,'right')]:
        pts=np.array([[.111,sign*.024,1.244],[.109,sign*.095,1.316],[.143,sign*.113,1.206],[.146,sign*.050,1.135]])
        lapel=np.array([[pts[0]*(1-t)+pts[3]*t,pts[1]*(1-t)+pts[2]*t] for t in np.linspace(0,1,12)])
        panel('lapel_'+label,torso,lapel,'fold',.0025)
        seam('lapel_stitch_'+label,torso,[pts[0],pts[1],pts[2],pts[3]])
    # Collar around neck/back avoids obstructing the T1 head or cameras.
    g=[]
    for z,r in [(1.304,.107),(1.343,.100)]:
        g.append([[.005+.114*np.cos(a),r*np.sin(a),z] for a in np.linspace(.48,2*np.pi-.48,55)])
    panel('standing_collar',torso,g,'fold',.002)
    # Front patch pockets conform to the elliptical coat front, with rounded hem.
    for label,y,z,w,h in [('left_hip',.073,.835,.060,.102),('right_hip',-.073,.835,.060,.102),('breast',.080,1.135,.050,.060)]:
        p=[]
        for v in np.linspace(0,1,12):
            row=[]
            for u in np.linspace(-1,1,16):
                yy=y+u*w/2;zz=z+v*h-.005*(1-v)*u*u
                ry=np.interp(zz,[.57,.9,1.18,1.325],[.120,.130,.145,.150]);rx=.164
                xx=-.030+rx*np.sqrt(max(.01,1-(yy/ry)**2))+.005+.002*np.sin(np.pi*v)
                row.append([xx,yy,zz])
            p.append(row)
        panel('pocket_'+label,torso,p,'fold')
        seam('pocket_edge_'+label,torso,p[-1],r=.0009)
        seam('pocket_side_'+label,torso,[r[0] for r in p]+p[0]+[r[-1] for r in p])
    # Small stitched identity tab on breast pocket; no safety/PPE claim.
    panel('identity_tab',torso,[[[.115,.060,1.175],[.095,.099,1.175]],[[.115,.060,1.192],[.095,.099,1.192]]],'badge')
    for k,z in enumerate(np.linspace(.77,1.11,5)):
        # Low-poly convex fabric-covered buttons, face normal towards viewer +X.
        pts=[[.141,.024,z],[.144,.024,z]]
        tube('button_'+str(k),torso,pts,[.0032,.0032],'button',rings=16)
    for sign,side in [(1,'left'),(-1,'right')]:
        upper=f'astribot_arm_{side}_link_3';lower=f'astribot_arm_{side}_link_5'
        z0=1.2695
        points=np.array([[0,sign*(.181-.03*t),z0+.019-.285*t] for t in np.linspace(0,1,28)])
        points=np.vstack([[0,sign*.177,1.324],points])
        tube(side+'_upper_sleeve',upper,points,np.r_[.025,np.linspace(.067,.057,len(points)-1)])
        # Long sleeve tapers and stops above wrist; Sharpa hand remains exposed.
        points=np.array([[0,sign*.15,.9995-.222*t] for t in np.linspace(0,1,26)])
        tube(side+'_forearm_sleeve',lower,points,np.linspace(.058,.046,len(points)))
        points=np.array([[0,sign*.15,z] for z in np.linspace(.788,.771,5)])
        tube(side+'_wrist_cuff',lower,points,np.linspace(.047,.046,5),'fold')
        for name,body,zz,yy,rr in [('upper',upper,.989,sign*.151,.057),('cuff',lower,.775,sign*.15,.0468)]:
            seam(side+'_'+name+'_stitch',body,[[rr*np.cos(a),yy+rr*np.sin(a),zz] for a in np.linspace(0,2*np.pi,65)])
        seam(side+'_sleeve_seam',lower,[[.057-.010*t,sign*.15,.998-.219*t] for t in np.linspace(0,1,20)])
    result={'schema_version':1,'kind':'original parametric segmented visual lab coat','creator':'T1 spin-coating research project',
            'source':'authored for the official T1 kinematics; no third-party garment geometry copied',
            'appearance_reference':'https://www.innerscene.com/tools/library/3d-parts/laboratory-researcher-3d-person-healthcare-7a3ceb81',
            'limitations':['No cloth dynamics or cloth/environment contact','No protective-equipment validation','Sleeves are segmented rigid visual surfaces attached to arm links'],
            'collision':False,'geom_mass_kg':0,'parts':records}
    if export_assets:(ASSETS/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    from PIL import Image,ImageDraw
    from t1_spincoat_scene import scene,arm_indices
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'logs/spincoat/t1/chemist_coat'));args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    spec,bare,db,children=scene();meta=add_chemist_coat(spec)
    clothed=spec.compile();dc=mujoco.MjData(clothed)
    mujoco.mj_forward(clothed,dc)
    checks={'nq':bare.nq==clothed.nq,'nv':bare.nv==clothed.nv,'nu':bare.nu==clothed.nu,
            'body_mass_exact':np.array_equal(bare.body_mass,clothed.body_mass),'body_inertia_exact':np.array_equal(bare.body_inertia,clothed.body_inertia),
            'initial_qpos_exact':np.array_equal(bare.qpos0,clothed.qpos0),'initial_qvel_exact':np.array_equal(db.qvel,dc.qvel)}
    assert all(checks.values()),checks
    ids=[g for g in range(clothed.ngeom) if clothed.geom(g).name.startswith('chemist_coat/')]
    checks['all_coat_noncolliding']=bool(np.all(clothed.geom_contype[ids]==0) and np.all(clothed.geom_conaffinity[ids]==0))
    checks['all_coat_group2']=bool(np.all(clothed.geom_group[ids]==2))
    checks['all_coat_massless']=all(g.mass==0 for g in spec.geoms if g.name.startswith('chemist_coat/'))
    mujoco.mj_forward(bare,db);mujoco.mj_forward(clothed,dc)
    checks['initial_qacc_exact']=bool(np.array_equal(db.qacc,dc.qacc))
    checks['initial_contacts_equal']=bare.nbody==clothed.nbody and db.ncon==dc.ncon
    assert all(checks.values()),checks
    checks.update({'nq_value':bare.nq,'nv_value':bare.nv,'nu_value':bare.nu,'all_scene_body_mass_sum_kg':float(bare.body_mass.sum()),'new_visual_geoms':len(meta['parts'])})
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'t1_chemist_coat_preview.xml').write_text(spec.to_xml())
    work=ROOT/'logs/spincoat/t1/home_01/episode.npz'
    poses={'initial':bare.qpos0.copy()}
    if work.exists():
        data=np.load(work);poses['working']=data['qpos'][np.argmin(abs(data['time']-27.0))] if 'time' in data else data['qpos'][min(1350,len(data['qpos'])-1)]
    for model in [bare,clothed]:
        model.vis.global_.offwidth=1200;model.vis.global_.offheight=1200
        model.vis.headlight.ambient[:]=.35;model.vis.headlight.diffuse[:]=.5
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0]
    shots=[]
    for pose,q in poses.items():
        for view,az,el in [('front',180,-6),('side',80,-6),('back',0,-6),('portrait',180,-3)]:
            panels=[]
            for label,model,data in [('without',bare,db),('with',clothed,dc)]:
                data.qpos[:]=q;mujoco.mj_forward(model,data)
                groups=model.geom_group.copy()
                if view=='portrait':
                    # Hide the workbench/props in this one garment inspection view.
                    for g in range(model.ngeom):
                        bname=model.body(model.geom_bodyid[g]).name
                        if not (bname.startswith('astribot') or bname.startswith('sharpa_') or model.geom(g).name.startswith('chemist_coat/') or model.geom(g).name=='floor'):
                            model.geom_group[g]=5
                camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.94];camera.distance=2.08;camera.azimuth=az;camera.elevation=el
                with mujoco.Renderer(model,height=1000,width=850) as r:
                    r.update_scene(data,camera=camera,scene_option=opt)
                    # Thin garment layers otherwise suffer shadow-map acne.
                    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    im=Image.fromarray(r.render())
                model.geom_group[:]=groups
                ImageDraw.Draw(im).text((16,16),f'T1 {pose} {view} | {label} visual lab coat',fill='white')
                im.save(out/f'{pose}_{view}_{label}.png');panels.append(im)
            combined=Image.new('RGB',(1700,1040),(28,31,34));combined.paste(panels[0],(0,40));combined.paste(panels[1],(850,40))
            ImageDraw.Draw(combined).text((20,12),'Optional visual garment | segmented rigid cloth | mass / dynamics unchanged',fill='white')
            combined.save(out/f'{pose}_{view}_comparison.png');shots.append(combined)
    print(json.dumps({'output':str(out),'checks':checks,'poses':list(poses)},indent=2))
