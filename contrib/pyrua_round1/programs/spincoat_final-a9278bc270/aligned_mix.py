"""Camera-layout independent wrist-axis alignment demonstrated in s04 E006."""
import numpy as np
from align_axis import plan_align, execute_align
from mixing_exact import mix
from mixing import mix as baseline_mix
from pyrua.codebox.outcome import Outcome


def mix_aligned(cb,ctx):
    low,high=cb.spec['arm_joint_limits'][12]
    selected=None
    for dy in (.01,0.,.02,.03,-.01,-.02):
        out,proposal=plan_align(cb,wrist7=-.32194,count=150,delta=(0,dy,0))
        if proposal is not None and out:
            centre=float(proposal['rows'][-1,12])
            if centre-.23>=low and centre+.23<=high:
                selected=proposal;break
            print('ALIGN_WAVE_RANGE_REFUSED',dy,centre,flush=True)
    if selected is None:
        return baseline_mix(cb,ctx)
    out=execute_align(cb,selected)
    if not out:return out
    fingers=cb.touch('left').data['fingers']
    if fingers['thumb']['held_back_rad']<.02 or fingers['middle']['held_back_rad']<.005:
        return Outcome(False,'aligned_support_unconfirmed',fingers,0)
    return mix(cb)
