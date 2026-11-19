"""Event0113 waveform, with s06 layout6 clipping that scored mixed at380.

Call only after a camera-verified tube hold and sufficient rack clearance. The
caller raises the hand to the reported total190mm pickup clearance first.
"""
import numpy as np
from pyrua.codebox.outcome import Outcome

def wrist_mix(cb,amplitude=.22,frequency=1.5,seconds=6.):
    observation=cb.robo.observe(cameras=[])
    q=np.array(observation.arm_cmd)
    low,high=cb.spec['arm_joint_limits'][12]
    if not low<=q[12]<=high:
        return Outcome(False,'mix_joint_limit',{'baseline':float(q[12])})
    time=np.arange(1,round(seconds*cb.hz)+1)/cb.hz
    rows=np.tile(q,(len(time),1));raw=q[12]+amplitude*np.sin(2*np.pi*frequency*time)
    rows[:,12]=np.clip(raw,low+.001,high-.001)
    clipped=int(np.count_nonzero(rows[:,12]!=raw))
    peak=float(np.abs(np.diff(np.r_[q[12],rows[:,12]])).max()*cb.hz)
    if peak>5.5:return Outcome(False,'mix_speed_limit',{'peak':peak})
    cb.robo.step(arm=rows,log=True);cb.sync()
    out=cb.hold(15)
    return Outcome(out.ok,'mix_motion_complete',{'rows':len(time),'peak_speed':peak,'clipped_rows':clipped},len(time)+out.steps)
