# s02: unchanged single-call helper; event 0170 confirms mixed at 530 after wrist-7 alignment. Earlier reconstruction of event 0113's successful split wave; rehearsal 0149 refuses before motion, exploration 0153 executes without mixing. Source: seeds/s02/code/explore/mixing_exact.py
"""E002 scored mixing at step630 with this relative wrist waveform."""
import numpy as np
from pyrua.codebox.outcome import Outcome

def mix(cb):
    obs=cb.robo.observe();t=np.arange(1,151)/25
    rows=np.tile(obs.arm_cmd,(150,1));rows[:,12]+=.22*np.sin(2*np.pi*1.5*t)
    low,high=cb.spec['arm_joint_limits'][12]
    if rows[:,12].min()<low or rows[:,12].max()>high:
        return Outcome(False,'mix_joint_range',{'minimum':float(rows[:,12].min()),'maximum':float(rows[:,12].max())},0)
    result=cb.robo.step(arm=rows);cb.sync()
    if result.episode_ended:return Outcome(False,'episode_ended',{},result.executed)
    cb.hold(15)
    return Outcome(True,'mix_motion_complete',{},165)
