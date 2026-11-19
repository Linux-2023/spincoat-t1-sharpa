"""Measured alignment and wrist rocking; clipped-wave fallback from s06 event0187."""
import numpy as np
from pyrua.codebox.outcome import Outcome
from axis_alignment import plan_adaptive,execute_align


def clipped_wave(cb,reason):
    """Used only when a safe alignment cannot be planned, before any arm motion."""
    cb.sync();obs=cb.robo.observe(cameras=[])
    touch=cb.touch('left').data['fingers']
    if touch['thumb']['held_back_rad']<.015:return Outcome(False,'grip_unconfirmed',{'touch':touch},0)
    low,high=cb.spec['arm_joint_limits'][12];t=np.arange(1,151)/25
    rows=np.tile(obs.arm_cmd,(150,1));raw=rows[:,12]+.22*np.sin(2*np.pi*1.5*t)
    rows[:,12]=np.clip(raw,low+.001,high-.001)
    print('CLIPPED_WAVE_FALLBACK',{'reason':reason,'baseline':float(obs.arm_cmd[12]),'clipped_rows':int(np.count_nonzero(rows[:,12]!=raw))},flush=True)
    begin=cb.steps_used
    for first,last in [(0,4),(4,12),(12,150)]:
        moved=cb.robo.step(arm=rows[first:last]);cb.sync()
        if moved.episode_ended:return Outcome(False,'episode_ended',{},cb.steps_used-begin)
    cb.hold(15)
    if cb.ended:return Outcome(False,'episode_ended',{},cb.steps_used-begin)
    return Outcome(True,'clipped_mix_motion_complete',{},cb.steps_used-begin)


def mix(cb):
    start=cb.steps_used
    out,plan=plan_adaptive(cb)
    if plan is None:return clipped_wave(cb,out.reason)
    low,high=cb.spec['arm_joint_limits'][12]
    mean=out.data['final_joint6']
    if mean-.22<low or mean+.22>high:return clipped_wave(cb,'planned_wave_out_of_range')
    out=execute_align(cb,plan)
    if not out:return out
    obs=cb.robo.observe();t=np.arange(1,151)/25
    rows=np.tile(obs.arm_cmd,(150,1));rows[:,12]+=.22*np.sin(2*np.pi*1.5*t)
    if rows[:,12].min()<low or rows[:,12].max()>high:return Outcome(False,'mix_joint_range',{},0)
    result=cb.robo.step(arm=rows,log=True);cb.sync()
    if result.episode_ended:return Outcome(False,'episode_ended',{},result.executed)
    cb.hold(15)
    if cb.ended:return Outcome(False,'episode_ended',{},cb.steps_used-start)
    print('MIX_ALIGNED_COMPLETE',{'baseline':float(obs.arm_cmd[12]),'joint7':float(obs.arm_cmd[13]),'rows':150},flush=True)
    return Outcome(True,'mix_motion_complete',{},cb.steps_used-start)
