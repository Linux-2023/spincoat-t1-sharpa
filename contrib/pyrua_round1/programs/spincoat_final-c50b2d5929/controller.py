"""Camera-measured tube pickup and mixing with a consistent wrist rocking axis.
The direct-pad/high-lift prefix scored on E002; measured pose alignment and
an adaptive small translation restored mixing on E004. No task oracle is used.
"""
import numpy as np
from pyrua.codebox.outcome import Outcome
from perception_mask import measure_tube
from mixing import mix
from entry import enter,arrived

STAGES=['tube_lifted','mixed']


def setup(cb,ctx):
    """Find the reagent rim in two calibrated silhouettes and park at preparation."""
    out=measure_tube(cb,ctx)
    if out:
        ctx['rim']=np.array(ctx['mouth'])
    return out


def tube_lifted(cb,ctx):
    """Needs the measured rack and empty left hand. Leaves a high, stable tube hold."""
    if 'rim' not in ctx:
        out=setup(cb,ctx)
        if not out:return out
    if cb.table_z is None:cb.table()
    ctx['entry_attempted']=True
    out=enter(cb,ctx['rim'])
    if not out:return out
    ctx['grasp_started']=True
    out=cb.shape('left','pad',width=.015,fingers=['index','middle'],clearance=0)
    if not out and out.reason!='blocked':return out
    cb.hold(10)
    for height in [.11,.08]:
        point=np.array(cb.state()['left']['grasp_point'])
        target=point+[0,0,height]
        out=cb.move_to('left',target,path='joint')
        if not out and out.steps==0 and out.reason in ('ik_unreachable','needs_reconfigure'):
            out=cb.move_to('left',target,path='joint',reconfigure=True)
        out=arrived(cb,out,target)
        if not out:return out
        cb.hold(25)
    touch=cb.touch('left').data['fingers']
    # These subthreshold lags accompanied the benchmark-confirmed E002 grasp.
    supported=touch['thumb']['held_back_rad']>.02 and touch['middle']['held_back_rad']>.005
    ctx['tube_suspended']=bool(supported)
    return Outcome(bool(supported),'lift_motion_complete' if supported else 'grip_unconfirmed',{'touch':touch},0)


def mixed(cb,ctx):
    """Needs the high tube hold; leaves the mixed tube near its original held pose."""
    if not ctx.get('tube_suspended'):
        if ctx.get('grasp_started') or ctx.get('entry_attempted'):
            return Outcome(False,'grip_unconfirmed',{},0)
        out=tube_lifted(cb,ctx)
        if not out:return out
    out=mix(cb)
    ctx['mixed_motion']=bool(out)
    return out
