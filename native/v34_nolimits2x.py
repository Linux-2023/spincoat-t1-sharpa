"""Install the user-authorized uncapped reference policy in a private episode."""
from __future__ import annotations

import json
from pathlib import Path

import v34_uncapped_clock as clock
import v34_uncapped_left as left
import v34_uncapped_right as right
import v34_uncapped_wrist_base as wrist_base

PATCHED_SYMBOLS = (
    'v34_topp_core.retime',
    'v34_topp_core.save_report',
) + wrist_base.PATCHED_SYMBOLS + left.PATCHED_SYMBOLS + right.PATCHED_SYMBOLS
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('NoLimits2x must be installed once per frozen native process')
    import v34_topp_core as core
    import v34_dualarmoverlap as overlap
    if not getattr(overlap, '_INSTALLED', False):
        raise RuntimeError('NoLimits2x requires the complete frozen TOPP/overlap parent chain')
    # Installation evidence belongs to the owned sandbox, not the native run
    # directory: the native runner must create that directory exclusively.
    directory = Path.cwd()
    result = dict(
        schema_version=1, policy='NoLimits2x', installation_complete=False,
        requested_motion_reference_factor=clock.FACTOR,
        source_clock_basis='Native/R08 nominal references through existing geometry adapters',
        software_command_speed_caps_removed=True,
        old_topp_and_velocity_certificates_valid=False,
        physical_timestep_s=clock.DT,
        actuator_physical_limits_and_gain_schedules_unchanged=True,
        geometry_and_physical_acceptance_unchanged=True,
        whole_episode_half_duration_claim=False,
        physical_success=None,
        patched_symbols=list(PATCHED_SYMBOLS))
    try:
        core.retime = clock.retime
        core.save_report = clock.save_report
        wrist_base.install()
        left.install()
        right.install()
        clock.record('policy_install', 'NoLimits2x',
                     requested_motion_factor=clock.FACTOR,
                     software_velocity_caps_enforced=False,
                     physical_plant_and_measured_acceptance_unchanged=True,
                     live_pose_velocity_or_time_shortcuts_added=False,
                     physical_success_inferred=False)
        _INSTALLED = True
        result['installation_complete'] = True
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result['source_patch_records'] = clock.RECORDS
        path = directory / 'uncapped_install.json'
        temporary = path.with_suffix('.json.writing')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
