"""Original surrounding lab fixtures, inspired by documented shared facilities."""
import numpy as np
import mujoco

def add_lab_room_facilities(s,b,box,cyl,line,label):
    steel='lab_satin_steel';dark='lab_black_rubber';glass='lab_clear_guard'
    # Surroundings stay outside the active tabletop and robot reach corridor.
    # Sample preparation bench and enclosed lower storage, to operator's left.
    box('prep_counter',[.87,1.85,.82],[.33,.70,.022],material=steel)
    box('prep_storage',[.89,1.85,.40],[.29,.66,.38],rgba=[.69,.74,.75,1])
    for y in [1.40,1.85,2.30]:
        box('prep_door_'+str(y),[.593,y,.40],[.008,.210,.35],rgba=[.82,.85,.84,1])
        line('prep_handle_'+str(y),[.575,y-.055,.60],[.575,y+.055,.60],.004)
    # Optical inspection microscope: base, frame, stage, turret, objectives,
    # binocular viewing tubes, adjustment knobs and darkened specimen deck.
    box('microscope_base',[.84,1.55,.87],[.11,.085,.027],material=dark)
    box('microscope_frame',[.92,1.55,1.045],[.027,.037,.155],rgba=[.74,.78,.78,1])
    box('microscope_head',[.86,1.55,1.19],[.075,.045,.034],rgba=[.79,.82,.82,1])
    box('microscope_stage',[.80,1.55,1.005],[.075,.065,.009],material=dark)
    cyl('microscope_turret',[.81,1.55,1.105],.039,.018,material=dark)
    for i in range(4):
        a=i*np.pi/2
        cyl('objective_'+str(i),[.81+.025*np.cos(a),1.55+.025*np.sin(a),1.07],.008,.052,material=steel)
    for y in [1.525,1.575]:
        cyl('eyepiece_'+str(y),[.775,y,1.225],.013,.07,material=dark,quat=[.9238795,0,-.3826834,0])
    cyl('focus_knob',[.92,1.60,1.05],.027,.025,material=dark,quat=[2**-.5,2**-.5,0,0])
    box('specimen_slide',[.80,1.55,1.016],[.024,.012,.001],rgba=[.31,.49,.55,1])
    label('inspection_station','OPTICAL INSPECTION\nSAMPLE PREPARATION',[.535,1.55,.78],.26,.05)
    # Weighing station with an enclosed draft shield and a small record terminal.
    box('balance_base',[.83,2.13,.872],[.12,.11,.030],rgba=[.84,.86,.85,1])
    cyl('balance_pan',[.85,2.13,.915],.048,.006,material=steel)
    for y in [2.025,2.235]:box('balance_shield_y'+str(y),[.85,y,1.03],[.10,.002,.12],material=glass)
    for x in [.75,.95]:box('balance_shield_x'+str(x),[x,2.13,1.03],[.002,.107,.12],material=glass)
    box('balance_shield_roof',[.85,2.13,1.152],[.102,.107,.002],material=glass)
    label('balance_display','BALANCE\nSTANDBY',[.709,2.13,.873],.14,.035,background=(20,32,40),ink=(123,208,189))
    # Tall cabinet with visible trays behind acrylic doors; avoid filling the
    # entire interior with one opaque block, preserving real shelf depth.
    x,y=.97,-1.88
    box('storage_back',[1.235,y,1.09],[.015,.47,.99],rgba=[.74,.78,.79,1])
    for yy in [y-.47,y+.47]:box('storage_side_'+str(yy),[x,yy,1.09],[.265,.014,.99],rgba=[.78,.81,.82,1])
    for z in [.10,2.08]:box('storage_end_'+str(z),[x,y,z],[.28,.48,.018],rgba=[.80,.83,.84,1])
    for z in [.43,.83,1.23,1.63]:
        box('storage_shelf_'+str(z),[x,y,z],[.26,.45,.009],material=steel)
        for j,yy in enumerate([y-.27,y,y+.27]):
            box('sample_bin_'+str(z)+'_'+str(j),[.96,yy,z+.065],[.17,.108,.055],rgba=[.65,.75,.79,.88])
            box('sample_lid_'+str(z)+'_'+str(j),[.96,yy,z+.123],[.175,.112,.005],rgba=[.25,.39,.46,1])
    for yy in [y-.235,y+.235]:
        box('cabinet_glass_'+str(yy),[.692,yy,1.09],[.003,.226,.96],material=glass)
        line('cabinet_handle_'+str(yy),[.672,yy,.94],[.672,yy,1.18],r=.005)
    label('sample_storage','SAMPLE STORAGE\nRESEARCH MATERIALS',[.68,y,1.98],.55,.07)
    # Room envelope: no black gaps beyond the first wall; tile-grid ceiling
    # provides scale and fluorescent diffusers provide broad, neutral fill.
    for g in s.geoms:
        if g.name=='lab_context_room_back':g.size[:]=[.025,3.02,1.35]
        if g.name=='floor':g.rgba[:]=[.53,.58,.60,1]
    box('room_front',[-2.54,0,1.35],[.02,3.02,1.35],rgba=[.80,.84,.85,1])
    for yy in np.arange(-3.,3.01,.60):box('ceiling_grid_y'+str(yy),[-.55,float(yy),2.674],[1.98,.003,.004],rgba=[.60,.66,.68,1],collision=False)
    for xx in np.arange(-2.5,1.41,.60):box('ceiling_grid_x'+str(xx),[float(xx),0,2.674],[.003,2.98,.004],rgba=[.60,.66,.68,1],collision=False)
    s.add_material(name='lab_diffuser_luminous',rgba=[.94,.97,1.,1],emission=.35,specular=.0)
    for xx in [-1.,.2]:
        for yy in [-1.8,0,1.8]:box('ceiling_lamp_'+str(xx)+'_'+str(yy),[xx,yy,2.660],[.265,.265,.006],material='lab_diffuser_luminous',collision=False)
    # Large-window frames in the far wall and an access door provide room cues.
    box('door_panel',[1.369,2.60,1.06],[.008,.34,1.02],rgba=[.67,.73,.75,1])
    box('door_window',[1.358,2.60,1.42],[.002,.23,.37],rgba=[.21,.34,.40,1])
    line('door_handle',[1.335,2.37,.98],[1.335,2.37,1.15],r=.009)
    label('door_sign','SAMPLE PREPARATION\nRESEARCH LABORATORY',[1.344,2.60,1.94],.55,.10)
    # Additional fill sources use a neutral color temperature, so white plastic
    # no longer relies on a camera-mounted headlight for all illumination.
    s.worldbody.add_light(name='lab_front_key',pos=[-.65,-1.2,2.3],dir=[.5,.4,-1],diffuse=[.48,.51,.54],ambient=[.05]*3,castshadow=False)
    s.worldbody.add_light(name='lab_front_fill',pos=[-.55,1.5,2.2],dir=[.6,-.5,-1],diffuse=[.32,.34,.36],ambient=[.035]*3,castshadow=False)
    return {'surroundings':['optical inspection bench','analytical balance with draft shield','sample storage cabinet','ceiling grid and fluorescent diffusers','laboratory access door'],'references':['https://www-prod.media.mit.edu/posts/meet-yellowbox/','https://snsf.stanford.edu/facilities/fab/npc'],'geometry':'Original approximate fixtures; not manufacturer equipment models or a measured room reconstruction.'}
