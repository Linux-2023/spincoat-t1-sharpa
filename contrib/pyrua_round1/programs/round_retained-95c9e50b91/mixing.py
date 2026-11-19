# s04 exact spincoat_submit mixing helper; event 0165 confirms mixed 386 after conditional wrist-only centering; source seeds/s04/code/spincoat_submit/mixing.py.
"""Wrist-six motion counted in s02 event 0113; keeps the complete waveform in range."""
import numpy as np


def mix(cb, ctx):
    Outcome = ctx['outcome_type']
    obs = cb.robo.observe(cameras=[])
    baseline = obs.arm_cmd.copy()
    low, high = cb.spec['arm_joint_limits'][12]
    amplitude = .22
    if baseline[12] - amplitude < low or baseline[12] + amplitude > high:
        # This small positioning motion retained the counted lift in s04.
        # Keep every finger command and leave a margin for the full oscillation.
        target = baseline.copy()
        target[12] = np.clip(baseline[12], low + amplitude + .02, high - amplitude - .02)
        moved = cb.robo.step(arm=np.linspace(baseline, target, 16))
        cb.sync()
        if moved.episode_ended:
            return Outcome(False, 'episode_ended', {}, moved.executed)
        cb.hold(15)
        fingers = cb.touch('left').data['fingers']
        if fingers['thumb']['held_back_rad'] < .015:
            return Outcome(False, 'grip_unconfirmed_after_positioning', {'touch': fingers}, 0)
        obs = cb.robo.observe(cameras=[])
        baseline = obs.arm_cmd.copy()
    rows = np.tile(baseline, (150, 1))
    rows[:, 12] += amplitude * np.sin(2 * np.pi * 1.5 * np.arange(1, 151) / 25)
    if rows[:, 12].min() < low or rows[:, 12].max() > high:
        return Outcome(False, 'mix_joint_range', {}, 0)
    # Preserve the demonstrated call partition without adding simulated holds.
    for start, end in ((0, 21), (21, 25), (25, 150)):
        moved = cb.robo.step(arm=rows[start:end])
        if moved.episode_ended:
            cb.sync()
            return Outcome(False, 'episode_ended', {}, moved.executed)
    cb.sync()
    cb.hold(15)
    return Outcome(True, 'mix_motion_complete', {}, 165)
