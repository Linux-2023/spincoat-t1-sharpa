"""Uncapped wrist targets and base command policy; the physical plant is unchanged."""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import inspect
import math
import textwrap

from v34_uncapped_clock import FACTOR, patch, record

PATCHED_SYMBOLS = (
    't1_wrist_mixing_v27.command_plan',
    't1_wrist_mixing_v27.advance_mouth_pivot_shake',
    't1_mobile_stage.MobileDockController.__init__',
    't1_mobile_stage.MobileDockController.step',
    't1_mobile_stage.official_driver',
    't1_coarse_stop.CoarseStopController.step',
    't1_coarse_stop.advance_coarse_stop',
)
_INSTALLED = False


def _mix_config(source):
    values = (source.frequency_hz, source.ramp_s, source.steady_s)
    if not all(math.isfinite(float(value)) and value > 0 for value in values):
        raise ValueError('Finite positive mixing frequency and durations required')
    result = replace(source, frequency_hz=source.frequency_hz * FACTOR,
                     ramp_s=source.ramp_s / FACTOR,
                     steady_s=source.steady_s / FACTOR)
    record('mixing_reference', source.phase_name, requested_factor=FACTOR,
           source_config=asdict(source), executed_reference_config=asdict(result),
           software_command_rate_clipping=False,
           physical_servo_parameters_unchanged=True,
           position_overshoot_clamp_retained=True,
           actual_tracking_or_success_inferred=False)
    return result


def _replace_body(function, body, label):
    source = textwrap.dedent(inspect.getsource(function.__code__))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('Expected one native wrapper: ' + label)
    node = tree.body[0]
    lines = source.splitlines(True)
    old = ''.join(lines[node.body[0].lineno - 1:node.body[-1].end_lineno])
    new = textwrap.indent(textwrap.dedent(body).strip() + '\n', '    ')
    return patch(function, [(old, new, 1)], label)


def install():
    global _INSTALLED
    if _INSTALLED:
        return PATCHED_SYMBOLS
    import t1_wrist_mixing_v27 as mixing
    import t1_mobile_stage as mobile
    import t1_coarse_stop as coarse

    patch(mixing.command_plan, [
        ('0<c.frequency_hz<=8', 'np.isfinite(c.frequency_hz) and 0<c.frequency_hz', 1),
        ('for k in range(1,count+1):commands[k,j]=commands[k-1,j]+np.clip(target[k]-commands[k-1,j],-2*dt,2*dt)',
         'commands[1:,j]=target[1:]', 1),
    ], 'wrist.command_plan.remove_command_rate_clip')
    patch(mixing.advance_mouth_pivot_shake, [
        ('c=config or MixingConfig()', 'c=_uncapped_mix_config(config or MixingConfig())', 1),
        ("if peak>2+1e-9 or margin<.005 or cm<0:raise RuntimeError('Original command speed/range guards failed')",
         "if margin<.005 or cm<0:raise RuntimeError('Original command range guards failed')", 1),
        ('report=dict(config=asdict(c),',
         'report=dict(software_command_speed_limit_applied=False,requested_reference_factor=_uncapped_mix_factor,config=asdict(c),', 1),
    ], 'wrist.executed_reference_and_range_guard',
       {'_uncapped_mix_config': _mix_config, '_uncapped_mix_factor': FACTOR})

    patch(mobile.MobileDockController.__init__, [
        ('self.speed_limit=speed_limit;', 'self.speed_limit=None;', 1),
    ], 'base.software_speed_limit_disabled')
    patch(mobile.MobileDockController.step, [
        ('cmd=np.clip(2*err+.8*self.integral-.5*d.qvel[v:v+2],-self.speed_limit,self.speed_limit)',
         'cmd=2*err+.8*self.integral-.5*d.qvel[v:v+2]', 1),
        ('wz=np.clip(-2*yaw-.4*d.qvel[v+5],-.2,.2)',
         'wz=-2*yaw-.4*d.qvel[v+5]', 1),
    ], 'base.remove_translational_and_yaw_command_caps')
    patch(mobile.official_driver, [
        ('max_wheel_speed=10.', 'max_wheel_speed=None', 1),
    ], 'base.disable_official_software_wheel_cap')

    if (not coarse.CoarseStopController.step.__code__.co_filename.endswith('v34_speedtransport.py')
            or coarse.CoarseStopController.step.__code__.co_name != 'step'):
        raise RuntimeError('Expected the installed SpeedTransport command-envelope wrapper')
    _replace_body(coarse.CoarseStopController.step, '''
        scope = _ACTIVE_TRANSFER.get()
        was_approaching = self.mode == "approach"
        stopped = old_step(self, data, target_xy, enable_gate=enable_gate)
        if (scope is not None and scope["data"] is data
                and np.array_equal(self.station_xy, scope["target"])):
            if was_approaching:
                scope["approach_steps"] += 1
            if was_approaching and self.mode != "approach":
                scope["latched_speed_m_s"] = float(np.linalg.norm(data.qvel[self.dv:self.dv + 2]))
        return stopped
    ''', 'base.remove_loaded_velocity_envelope_and_slew')

    advance = coarse.advance_coarse_stop
    if (not advance.__code__.co_filename.endswith('v34_speedtransport.py')
            or advance.__code__.co_name != 'advance'):
        raise RuntimeError('Expected the installed SpeedTransport audit wrapper')
    patch(advance, [
        ('''    if len(route) > 7:
        route[7] = profile["far_speed_limit_m_s"]
    else:
        changed["speed_limit"] = profile["far_speed_limit_m_s"]
''', '', 1),
        ('''    scope = dict(data=data, target=np.asarray(target),
                 last_time=float(data.time),
                 speed_limit=profile["near_speed_limit_m_s"],
                 minimum_limit_m_s=profile["near_speed_limit_m_s"],
                 maximum_limit_m_s=profile["near_speed_limit_m_s"],
                 approach_steps=0, deceleration_steps=0, slow_steps=0,
                 deceleration_started_at_s=None,
                 slow_region_entered_at_s=None,
                 latched_speed_m_s=None, latched_limit_m_s=None)''',
         '''    scope = dict(data=data, target=np.asarray(target),
                 approach_steps=0, latched_speed_m_s=None)''', 1),
        ('        profile=profile,', '        source_profile=profile,', 1),
        ('        cap_schedule="C2 actual-distance blend with bounded per-step slew",',
         '        cap_schedule=None, software_command_velocity_caps_removed=True, software_velocity_slew_cap_removed=True,', 1),
        ('        earlier_docks_changed=False,',
         '        earlier_dock_command_caps_removed=True, earlier_dock_physical_gates_unchanged=True,', 1),
        ('if key not in ("data", "target", "last_time", "speed_limit")',
         'if key not in ("data", "target")', 1),
    ], 'base.report_uncapped_policy_without_obsolete_cap_statistics')
    record('base_policy', 'uncapped_native_feedback',
           feedback_gains_unchanged=True, integral_antiwindup_retained=True,
           model_actuator_control_and_force_limits_unchanged=True,
           physical_brake_and_existing_gain_schedule_unchanged=True,
           actual_stopped_gate_unchanged=True,
           fixed_factor_claim_for_feedback_docks=False)
    _INSTALLED = True
    return PATCHED_SYMBOLS
