"""Explicit experimental admission of at most 1 ms non-thumb support loss.

The native strict held gate and its evidence remain unchanged. This policy is
not a physical-model change, a strict-grip certificate, or a hardware limit.
"""
from functools import wraps
import math

POLICY_ID = 'held_support_1ms_non_thumb_gap'
TIMESTEP_S = .001
WINDOW_SAMPLES = 500
MAX_TOTAL_GAP_S = .001
FORCE_THRESHOLD_N = .01
PATCHED_SYMBOLS = ('t1_v10_v27_stage_v28.adapt_left_stage',)
_INSTALLED = False


def evaluate_support_window(window, strict_gate, timestep_s):
    """Evaluate a complete native window without mutating the strict result."""
    if not math.isfinite(timestep_s) or abs(timestep_s-TIMESTEP_S) > 1e-12:
        raise ValueError('Support admission requires the unchanged 1 ms timestep')
    if len(window) != WINDOW_SAMPLES or strict_gate['last_halfsecond_samples'] != WINDOW_SAMPLES:
        raise ValueError('Support admission requires the full 500-sample window')
    opposed = []
    thumb = []
    times = []
    for row in window:
        time, counts, forces = float(row[0]), row[2], row[3]
        if len(counts) != 5 or len(forces) != 5:
            raise ValueError('Expected the five native finger observations')
        if not all(math.isfinite(float(value)) for value in (time, *counts, *forces)):
            raise ValueError('Nonfinite support observation')
        if times and abs(time-times[-1]-timestep_s) > 1e-8:
            raise ValueError('Support observations must be consecutive native ticks')
        supported = [count > 0 and force > FORCE_THRESHOLD_N for count, force in zip(counts, forces)]
        thumb.append(bool(supported[0]))
        opposed.append(bool(supported[0] and sum(supported[1:]) >= 2))
        times.append(time)
    lift = float(strict_gate['tube_lift_mm'])
    required_lift = float(strict_gate['required_lift_mm'])
    translation = float(strict_gate['relative_translation_last_halfsecond_mm'])
    rotation = float(strict_gate['relative_rotation_last_halfsecond_deg'])
    if not all(math.isfinite(value) for value in (lift, required_lift, translation, rotation)):
        raise ValueError('Nonfinite native held-gate measurement')
    lift_ok = lift >= required_lift
    stable = translation < 2. and rotation < 3.
    original_opposed = all(opposed)
    original_pass = bool(lift_ok and stable and original_opposed)
    if (original_opposed != bool(strict_gate['opposing_thumb_plus_two_non_thumb_last_halfsecond'])
            or original_pass != bool(strict_gate['pass_'])):
        raise ValueError('Strict held-gate result and observation window disagree')
    gaps = [index for index, supported in enumerate(opposed) if not supported]
    gap_s = len(gaps)*timestep_s
    longest = current = 0
    for supported in opposed:
        current = 0 if supported else current+1
        longest = max(longest, current)
    checks = dict(lift=lift_ok, relative_pose=stable, thumb_continuous=all(thumb),
                  total_non_thumb_gap=gap_s <= MAX_TOTAL_GAP_S+1e-12,
                  terminal_opposing_support=opposed[-1])
    admitted = all(checks.values())
    return dict(policy_id=POLICY_ID, acceptance_scope='user_authorized_experimental_continuity_tolerance',
                admitted=admitted, original_strict_pass=original_pass,
                experimental_override_used=admitted and not original_pass,
                original_strict_gate=dict(strict_gate), samples=len(window), timestep_s=timestep_s,
                time_range_s=[times[0], times[-1]], force_threshold_N=FORCE_THRESHOLD_N,
                supported_samples=sum(opposed), missing_support_samples=len(gaps),
                missing_support_times_s=[times[index] for index in gaps],
                total_missing_support_s=gap_s, longest_missing_support_s=longest*timestep_s,
                allowed_total_missing_support_s=MAX_TOTAL_GAP_S,
                checks=checks, failed_checks=[name for name, passed in checks.items() if not passed],
                unchanged_pose_limits=dict(translation_exclusive_mm=2., rotation_exclusive_deg=3.),
                original_strict_evidence_preserved=True, physical_parameters_changed=False,
                hardware_certification=False)


def _once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'Support tolerance source anchor expected once, found {count}: {old!r}')
    return source.replace(old, new, 1)


def adapt_stage(source):
    source = _once(source, 'held_gate=None;fine_started=',
                   'held_gate=None;held_support_admission=None;fine_started=')
    source = _once(source,
        "if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')",
        "from v34_heldsupporttolerance import evaluate_support_window\n"
        "  held_support_admission=evaluate_support_window(window,held_gate,m.opt.timestep)\n"
        "  (out/'held_support_admission.json').write_text(json.dumps(held_support_admission,indent=2,allow_nan=False))\n"
        "  if not held_support_admission['admitted']:raise RuntimeError('experimental tube held/lift admission failed: '+','.join(held_support_admission['failed_checks']))")
    source = _once(source, 'held_gate=held_gate,pour_entry_gate=pour_entry_gate,',
        "held_gate=held_gate,experimental_held_support=held_support_admission,"
        "acceptance_scope='experimental_held_support_1ms; original strict held gate remains independently recorded',"
        'pour_entry_gate=pour_entry_gate,')
    return source


def install():
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Support tolerance must be installed once per frozen process')
    import v34_nolimits2x as parent
    import t1_v10_v27_stage_v28 as stage
    if not parent._INSTALLED:
        raise RuntimeError('Support tolerance requires the exact installed NoLimits2x parent')
    previous = stage.adapt_left_stage

    @wraps(previous)
    def adapted(source):
        return adapt_stage(previous(source))

    stage.adapt_left_stage = adapted
    _INSTALLED = True
    return PATCHED_SYMBOLS
