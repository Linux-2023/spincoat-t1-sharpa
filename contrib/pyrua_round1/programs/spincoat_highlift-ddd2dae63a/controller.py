"""Camera-measured pickup, mixing, pouring, and supported tube return.

The first four stages were counted on E005 with rewinds. This integration
regenerates geometry and motion; fresh results are in memory/REPORT.md.
"""
import numpy as np
from pyrua.codebox.outcome import Outcome
from vision import rim_rays,triangulate
from mixing import wrist_mix
from entry import plan_entry

STAGES=['tube_lifted','mixed','poured','tube_returned']


def setup(cb,ctx):
    """Measure the reagent rim using two calibrated head views before grasping."""
    out=cb.table()
    if not out.ok:return out
    first=rim_rays(cb)
    if not first:return Outcome(False,'rim_not_visible_before_drive')
    out=cb.go_to(0,.15,settle=20)
    if not out.ok:return out
    second=rim_rays(cb)
    if not second:return Outcome(False,'rim_not_visible_at_preparation')
    pairs=[]
    for a in first:
        for b in second:
            room,gap=triangulate((a[1],a[2]),(b[1],b[2]))
            local=np.asarray(cb.to_root(room))
            if gap<.008 and cb.table_z+.15<local[2]<cb.table_z+.24:
                pairs.append((a[0]+b[0]-10*gap,room,gap,local))
    if not pairs:return Outcome(False,'rim_geometry_inconsistent')
    _,room,gap,local=max(pairs,key=lambda p:p[0])
    ctx['rim_room']=room.tolist()
    cb.mark('reagent_rim',local)
    print('triangulated_rim',local.tolist(),'ray_separation',gap,flush=True)
    return Outcome(True,'rim_measured',{'rim':local.tolist(),'separation':gap})


def tube_lifted(cb,ctx):
    """Needs the tube in its rack; leaves a direct pad hold after two rises."""
    if 'rim_room' not in ctx:
        out=cb.go_to(0,.4,settle=20)
        if not out.ok:return out
        out=setup(cb,ctx)
        if not out.ok:return out
    if not ctx.get('grasp_closed'):
        rim=np.asarray(cb.to_root(ctx['rim_room']))
        planned=plan_entry(cb,rim)
        if not planned.ok:return planned
        app=np.array(planned.data['approach']);opening=planned.data['opening']
        pre=np.array(planned.data['pre']);target=np.array(planned.data['target'])
        ctx['entry_orientation']=[planned.data['yaw'],planned.data['pitch']]
        shape={'grasp':'pad','width':.025,'fingers':['index','middle'],'clearance':.018}
        out=cb.move_to('left',pre,approach=app,open=opening,shape=shape)
        if not out.ok:return out
        out=cb.move_to('left',target,approach=app,open=opening,path='line')
        if not out.ok:return out
        out=cb.shape('left','pad',width=.015,fingers=['index','middle'],clearance=0)
        if not out.ok and out.reason!='blocked':return out
        out=cb.hold(10)
        if not out.ok:return out
        q=dict(zip(cb.arm_names,cb.cmd_arm));q.update(zip(cb.dex_names,cb.cmd_dex));q.update(cb.spec['embodiment'].get('held_joints',{}));cb.robot.fk(q)
        _,R=cb.robot.pose(cb.arms['left'].palm);ctx['prelift_rotation']=R.copy().tolist()
        ctx['grasp_closed']=True
        ctx['lift_segments']=0
    for i,dz in enumerate([.11,.08]):
        if ctx.get('lift_segments',0)>i:continue
        p=np.asarray(cb.state()['left']['grasp_point'])
        out=cb.move_to('left',p+[0,0,dz],path='joint')
        if not out.ok:return out
        out=cb.hold(25)
        if not out.ok:return out
        ctx['lift_segments']=i+1
    ctx['high_lift_executed']=True
    touch=cb.touch('left')
    return Outcome(True,'high_lift_executed',{'touch':touch.data})


def mixed(cb,ctx):
    """Needs the direct pad hold at high clearance; leaves it after wrist tilts."""
    if cb.ended:return Outcome(False,'episode_ended')
    if not ctx.get('high_lift_executed'):
        out=tube_lifted(cb,ctx)
        if not out.ok:return out
    out=wrist_mix(cb)
    if out.ok:ctx['mixed_motion_done']=True
    return out


def poured(cb,ctx):
    """Needs a mixed held tube; leaves it tipped over the measured beaker."""
    if not ctx.get("mixed_motion_done"):
        out=mixed(cb,ctx)
        if not out.ok:return out
    from pouring import pour
    return pour(cb,ctx)


def tube_returned(cb,ctx):
    """Needs the completed pour; seats the empty tube, opens, and clears rack."""
    if not ctx.get("pour_motions_done"):
        out=poured(cb,ctx)
        if not out.ok:return out
    from returning import return_tube
    return return_tube(cb,ctx)
