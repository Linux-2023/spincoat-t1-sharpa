"""A documented research-laboratory setting around the existing task geometry.

Layout references: Stanford SNSF spinner bench and MIT.nano manual-coater
photographs. Props are original dimensioned geometry, not facility CAD copies.
"""
from pathlib import Path
import json,os,io
import numpy as np
import mujoco
import trimesh
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets/lab_context'

def atomic_asset(path,data):
    if isinstance(data,str):data=data.encode('utf-8')
    if path.exists() and path.read_bytes()==data:return
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_bytes(data);temporary.replace(path)

def add_spincoat_lab_context(s):
    ASSETS.mkdir(parents=True,exist_ok=True)
    b=s.worldbody.add_body(name='lab_bench_context')
    silver=[.66,.69,.71,1];white=[.87,.89,.88,1];dark=[.12,.16,.18,1]
    s.add_material(name='lab_satin_steel',rgba=silver,specular=.75,shininess=.55)
    s.add_material(name='lab_white_polymer',rgba=white,specular=.3,shininess=.25)
    s.add_material(name='lab_amber_glass',rgba=[.30,.13,.025,.92],specular=.8,shininess=.8)
    s.add_material(name='lab_clear_guard',rgba=[.67,.85,.90,.12],specular=.8,shininess=.8)
    s.add_material(name='lab_black_rubber',rgba=dark,specular=.15,shininess=.15)
    count=0
    def geom(name,pos,size,kind=mujoco.mjtGeom.mjGEOM_BOX,material='lab_white_polymer',collision=True,rgba=None,**kw):
        nonlocal count
        count+=1
        args=dict(name='lab_context_'+name,type=kind,pos=pos,size=size,material=material,group=2,contype=int(collision),conaffinity=int(collision))
        if rgba is not None:args['rgba']=rgba
        args.update(kw);return b.add_geom(**args)
    def box(name,pos,half,**kw):return geom(name,pos,half,**kw)
    def cyl(name,pos,r,h,**kw):return geom(name,pos,[r,h/2,0],mujoco.mjtGeom.mjGEOM_CYLINDER,**kw)
    def line(name,a,c,r=.002,**kw):return b.add_geom(name='lab_context_'+name,type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[r,0,0],fromto=list(a)+list(c),material='lab_satin_steel',group=2,contype=0,conaffinity=0,**kw)
    def label(name,text,pos,width,height,background=(242,239,219),ink=(32,43,52),face='front'):
        file=ASSETS/(name+'.png')
        im=Image.new('RGB',(768,256),background);dr=ImageDraw.Draw(im)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',39)
        lines=text.split('\n')
        for i,t in enumerate(lines):dr.text((25,24+i*58),t,fill=ink,font=font)
        dr.rectangle((7,7,760,248),outline=ink,width=3)
        payload=io.BytesIO();im.save(payload,format='PNG');atomic_asset(file,payload.getvalue())
        s.add_texture(name='label_'+name,type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(file))
        s.add_material(name='label_'+name,textures=['','label_'+name]+['']*8,texuniform=False,rgba=[1,1,1,1],specular=.1)
        # Front means facing the operator (world -X). Explicit UV avoids
        # cube-map repetitions and preserves legibility of instrument labels.
        if face=='front':vertices=[[0,-width/2,-height/2],[0,-width/2,height/2],[0,width/2,height/2],[0,width/2,-height/2]]
        else:vertices=[[-width/2,-height/2,0],[width/2,-height/2,0],[width/2,height/2,0],[-width/2,height/2,0]]
        uv=[1,1,1,0,0,0,0,1] if face=='front' else [0,1,1,1,1,0,0,0]
        s.add_mesh(name='label_mesh_'+name,uservert=np.asarray(vertices).flatten().tolist(),userface=[0,1,2,0,2,3],usertexcoord=uv,inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        b.add_geom(name='lab_context_label_'+name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname='label_mesh_'+name,pos=pos,material='label_'+name,contype=0,conaffinity=0,group=2)
    # Stainless worktop, rear exhaust plenum, side guards and raised sash.
    # Retain the source worktop geometry and all existing collision envelopes.
    for g in s.geoms:
        if g.name=='floor':g.rgba[:]=[.45,.49,.52,1]
        if g.name=='lab_bench_visual_worktop':g.material='lab_satin_steel';g.rgba[:]=[1,1,1,1]
        if g.name=='lab_bench_visual_cabinet':g.rgba[:]=[.73,.76,.77,1]
    box('rear_plenum',[1.045,0,1.225],[.025,.998,.425],material='lab_satin_steel')
    box('upper_exhaust',[.71,0,1.69],[.39,1.025,.08],material='lab_white_polymer')
    box('left_upright',[.335,-1.005,1.22],[.020,.020,.42],material='lab_satin_steel')
    box('right_upright',[.335,1.005,1.22],[.020,.020,.42],material='lab_satin_steel')
    for y in [-.99,.99]:
        box('guard_'+str(y),[.70,y,1.20],[.35,.004,.40],material='lab_clear_guard')
    box('raised_sash',[.335,0,1.535],[.004,.98,.105],material='lab_clear_guard')
    box('sash_handle',[.318,0,1.43],[.014,.96,.010],material='lab_satin_steel')
    for z in [1.05,1.18,1.31]:
        box('rear_vent_'+str(z),[1.018,0,z],[.0015,.91,.006],material='lab_black_rubber',collision=False)
    # Shallow perforated rear strip rather than an ornamental full-area grid.
    box('rear_grille',[.975,0,.802],[.045,.96,.0018],material='lab_black_rubber',collision=False)
    for y in np.arange(-.94,.95,.014):
        box('grille_bar_'+str(round(y,3)),[.975,float(y),.805],[.045,.002,.001],material='lab_satin_steel',collision=False)
    # Ambient cleanroom walls, panel joints and ceiling diffusers.
    box('room_back',[1.40,0,1.35],[.025,2.1,1.35],rgba=[.79,.83,.84,1])
    box('room_ceiling',[-.55,0,2.7],[2.,3.,.02],rgba=[.85,.87,.86,1])
    for y in [-3.,3.]:box('room_side_'+str(y),[-.55,y,1.35],[2.,.02,1.35],rgba=[.79,.83,.84,1])
    for y in np.arange(-2.,2.1,.65):box('wall_seam_'+str(y),[1.372,float(y),1.35],[.001,.002,1.34],rgba=[.56,.61,.63,1],collision=False)
    for y in [-.7,.7]:
        box('light_diffuser_'+str(y),[.65,y,2.38],[.46,.24,.012],rgba=[.99,.99,.94,1],collision=False)
        s.worldbody.add_light(name='lab_ceiling_'+str(y),pos=[.55,y,2.15],dir=[0,0,-1],diffuse=[.65,.66,.64],ambient=[.08,.08,.08],castshadow=True)
    s.worldbody.add_light(name='lab_hood_light',pos=[.80,0,1.58],dir=[-.3,0,-1],diffuse=[.65,.59,.44],ambient=[.08,.07,.05],castshadow=True)
    # Instrument and consumable zones are deliberately outside the right-arm
    # transfer corridor. No prop is added underneath the pipette/hand path.
    def tray(name,x,y,w,d):
        box(name+'_floor',[x,y,.803],[d/2,w/2,.003])
        for a in [-1,1]:
            box(name+'_side_y'+str(a),[x,y+a*(w/2-.003),.818],[d/2,.003,.015])
            box(name+'_side_x'+str(a),[x+a*(d/2-.003),y,.818],[.003,w/2,.015])
    tray('chemical_secondary_tray',.84,.24,.29,.22)
    def bottle(name,x,y,height=.145,r=.025,white_plastic=False):
        z=.807
        profile=np.array([[0,0],[r*.84,0],[r,.004],[r,height*.69],[r*.82,height*.80],[r*.48,height*.85],[r*.48,height*.95],[0,height*.95]])
        mesh=trimesh.creation.revolve(profile,sections=64)
        path=ASSETS/(name+'.obj');atomic_asset(path,mesh.export(file_type='obj'));s.add_mesh(name=name,file=str(path))
        b.add_geom(name='lab_context_'+name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,pos=[x,y,z],material='lab_white_polymer' if white_plastic else 'lab_amber_glass',contype=0,conaffinity=0,group=2)
        cyl(name+'_collision',[x,y,z+height*.47],r,height*.94,rgba=[0,0,0,0]).group=3
        cyl(name+'_cap',[x,y,z+height*.96],r*.54,height*.08,material='lab_black_rubber')
        for i in range(24):
            a=i*2*np.pi/24
            line(name+'_cap_ridge_'+str(i),[x+r*.55*np.cos(a),y+r*.55*np.sin(a),z+height*.925],[x+r*.55*np.cos(a),y+r*.55*np.sin(a),z+height],r=.0006)
        label(name,name.replace('_',' ').upper()+'\nSIMULATION',[x-r-.0005,y,z+height*.40],r*1.55,height*.31)
    bottle('coating_solution',.82,.17)
    bottle('rinse_solution',.87,.30,.125,.024)
    tray('waste_tray',.83,.74,.20,.22)
    bottle('solvent_waste',.83,.74,.215,.047,True)
    # Hotplate, ceramic deck and a readable front fascia. Not process simulated.
    box('hotplate_case',[.80,.49,.832],[.095,.09,.032],material='lab_white_polymer')
    box('hotplate_ceramic',[.80,.49,.869],[.083,.078,.006],rgba=[.91,.91,.87,1])
    label('hotplate_display','HOTPLATE\nOFF',[.704,.49,.833],.09,.027,background=(22,37,45),ink=(117,219,207))
    for y in [.43,.55]:cyl('hotplate_knob_'+str(y),[.69,y,.824],.007,.009,material='lab_black_rubber',quat=[2**-.5,0,2**-.5,0])
    # Tip rack with actual repeated well and tip geometry, not a painted box.
    box('tip_rack_base',[.78,-.035,.823],[.060,.045,.023],rgba=[.19,.40,.54,1])
    box('tip_rack_insert',[.78,-.035,.850],[.056,.041,.004],rgba=[.82,.88,.87,1])
    for i in range(12):
        for j in range(8):
            x=.731+i*.0088;y=-.067+j*.0092
            cyl(f'tip_well_{i}_{j}',[x,y,.8547],.0026,.001,material='lab_black_rubber',collision=False)
            if i<8:cyl(f'tip_{i}_{j}',[x,y,.860],.0018,.010,rgba=[.75,.85,.91,.75],collision=False)
    box('tip_rack_lid',[.844,-.035,.893],[.003,.046,.042],rgba=[.32,.66,.78,.30])
    label('tip_rack','PIPETTE TIPS\nSTERILE - SIM',[.719,-.035,.824],.069,.026)
    # Substrate storage, clean tweezers and a closed wipe dispenser.
    cyl('wafer_box',[.61,.16,.812],.057,.024,rgba=[.25,.34,.43,1])
    cyl('wafer_box_lid',[.61,.16,.828],.059,.008,rgba=[.55,.75,.85,.45])
    label('substrates','SUBSTRATES',[.61,.16,.833],.075,.026,face='top')
    tray('tweezers_tray',.48,.29,.18,.10)
    for y in [.26,.31]:
        line('tweezer_a_'+str(y),[.47,y-.04,.813],[.48,y+.04,.813],.0013)
        line('tweezer_b_'+str(y),[.49,y-.04,.813],[.48,y+.04,.813],.0013)
    box('wipes_box',[.82,-.18,.832],[.065,.043,.032],rgba=[.68,.77,.71,1])
    box('wipes_slot',[.82,-.18,.865],[.041,.004,.001],material='lab_black_rubber',collision=False)
    box('protruding_wipe',[.82,-.18,.877],[.025,.001,.014],rgba=[.95,.96,.93,1],collision=False)
    label('wipes','LINT-FREE\nWIPES',[.754,-.18,.833],.065,.031)
    label('station','SPIN COATING WORKSTATION\nRESEARCH SIMULATION',[.293,0,1.686],.64,.075)
    label('hood_status','EXHAUST\nMODEL ONLY',[.293,.82,1.686],.18,.07)
    # Route a visual power/vacuum service bundle behind the instrument, away
    # from the transfer corridor. It carries no simulated flow or forces.
    for dy in [-.008,.008]:
        line('spinner_service_'+str(dy),[.63,-.34+dy,.808],[1.01,-.34+dy,.808],r=.002)
    # Replace the brown visual-only stand finish with stainless appearance.
    # Support geometry, tool start and all material contact parameters persist.
    for g in s.geoms:
        if g.name.startswith('stand/wood_'):g.material='lab_satin_steel';g.rgba[:]=[1,1,1,1]
        if g.name.startswith('spincoater/') and g.group==2 and g.rgba[0]>.7 and .1<g.rgba[1]<.6:
            g.rgba[:]=[.80,.83,.84,1]
    metadata={'layout_references':['https://snsf.stanford.edu/facilities/fab/npc/wetbench','https://mitnano.mit.edu/photo-gallery-mitnano-iap-courses','https://www.nanofab.ualberta.ca/capabilities/wet-chemical-processing/'],'modeling':'Original geometry inspired by documented facility photographs; no claim of exact reconstruction or manufacturer CAD. Public bench mesh attribution remains in third_party/lab_bench.','bench_dimensions_m':[2.,.75,.8],'objects':['exhaust hood/plenum','raised transparent sash','stainless worktop and perforated rear strip','secondary-containment reagent tray','capped coating and rinse bottles','contained solvent waste bottle','hotplate (off, no thermal simulation)','96-well tip rack','substrate box','tweezers in clean tray','lint-free wipes'],'task_geometry':'Pipette, active rack, spinner, robot and fluid parameters preserved.','limitations':'No airflow, chemical compatibility, ventilation efficacy or chemical safety certification. Dimensions of added objects are design assumptions, not measurements of reference facilities.'}
    from lab_room_facilities import add_lab_room_facilities
    metadata['room_facilities']=add_lab_room_facilities(s,b,box,cyl,line,label)
    (ASSETS/'provenance.json').write_text(json.dumps(metadata,indent=2))
    return metadata
