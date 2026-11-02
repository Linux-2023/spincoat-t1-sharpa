"""Restore only the mixing reference clock; keep the other uncapped policies."""
PATCHED_SYMBOLS = (
    'v34_uncapped_wrist_base._mix_config',
    't1_wrist_mixing_v27._uncapped_mix_factor',
)
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Native wrist timing must be installed once per frozen process')
    import v34_heldsupporttolerance as support
    import v34_uncapped_wrist_base as wrist
    import t1_wrist_mixing_v27 as mixing
    from v34_uncapped_clock import patch, record
    if not support._INSTALLED or not wrist._INSTALLED:
        raise RuntimeError('Native wrist timing requires the exact installed experimental parent')
    if mixing._uncapped_mix_config is not wrist._mix_config:
        raise RuntimeError('The executing wrist config helper no longer has its original alias')
    patch(wrist._mix_config, [
        ('frequency_hz=source.frequency_hz * FACTOR', 'frequency_hz=source.frequency_hz', 1),
        ('ramp_s=source.ramp_s / FACTOR', 'ramp_s=source.ramp_s', 1),
        ('steady_s=source.steady_s / FACTOR', 'steady_s=source.steady_s', 1),
        ('requested_factor=FACTOR', 'requested_factor=1.0', 1),
    ], 'wrist.only_restore_original_reference_clock')
    mixing._uncapped_mix_factor = 1.0
    record('install', 'NativeWristTiming', mixing_reference_factor=1.0,
           other_motion_reference_factor_unchanged=2.0,
           software_command_rate_clipping_restored=False,
           support_tolerance_unchanged=True,
           physical_parameters_and_acceptance_unchanged=True)
    _INSTALLED = True
    return PATCHED_SYMBOLS
