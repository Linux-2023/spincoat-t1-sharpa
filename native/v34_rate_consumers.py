"""Keep direct reference consumers coherent with per-label 4x/2x clocks.

Physical timestep, feedback gains, original native18 restorations and physical
qualification waits are untouched. Each direct reference has a named rollback
key; the historical uncapped module constant remains unchanged.
"""
from __future__ import annotations

from v34_local_speed_policy import reference_factor

PATCHED_SYMBOLS = (
    'v34_uncapped_right._VelocityReference.__init__',
    'v34_uncapped_right._VelocityReference.sample',
    'v34_uncapped_right._retimed_transfer',
    'v34_topp_right.retimed_transfer',
    'v34_overlap_control.OverlapSession.release',
    'v34_overlap_control.OverlapSession.before_step',
    'v34_overlap_control.OverlapSession.after_step',
    'v34_overlap_control.OverlapSession.save',
    'v34_dualarmoverlap.OverlapAcquisition.plan_at_actual_dock',
    'v34_dualarmoverlap.OverlapAcquisition.report',
    'v34_speedrightpickup.retime_prepared',
    'v34_speedrightpickup.retime_engine_source',
    't1_right_interior_workstation_v6.adapt_vertical_aspiration_source',
    'v34_topp_right._installed_precision_type',
)
DIRECT_LABELS = (
    'right/overlap_hand_program',
    'right/aspiration_index_preload',
    'right/aspiration_hover_press',
    'right/aspiration_thumb_release',
)
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Direct reference consumers must be installed once')
    import v34_uncapped_clock as clock
    import v34_uncapped_right as uncapped
    import v34_overlap_control as overlap
    import v34_dualarmoverlap as dual
    import v34_speedrightpickup as pickup
    import v34_topp_right as right
    import t1_right_interior_workstation_v6 as interior
    from v34_topp_right_paths import closure

    helpers = {'_wall_edge_reference_factor': reference_factor}
    clock.patch(uncapped._VelocityReference.__init__, [
        ('self.velocity = FACTOR * native.velocity',
         'self.reference_factor = _wall_edge_reference_factor(label)\n'
         '    self.velocity = self.reference_factor * native.velocity', 1),
        ('reference_factor=FACTOR,', 'reference_factor=self.reference_factor,', 1),
    ], 'per-reference velocity and endpoint scale', helpers=helpers)
    clock.patch(uncapped._VelocityReference.sample, [
        ('return p, FACTOR * v, FACTOR**2 * a, phase',
         'return p, self.reference_factor * v, self.reference_factor**2 * a, phase', 1),
    ], 'per-reference entry derivatives')
    clock.patch(uncapped._retimed_transfer, [
        ('FACTOR * through', 'path.reference_factor * through', 1),
    ], 'per-reference through speed')

    for function, count in ((overlap.OverlapSession.release, 1),
                            (overlap.OverlapSession.before_step, 3),
                            (overlap.OverlapSession.after_step, 1),
                            (overlap.OverlapSession.save, 4)):
        clock.patch(function, [('_uncapped_factor',
                    '_wall_edge_reference_factor("right/overlap_hand_program")', count)],
                    'selected overlap hand program ' + function.__name__, helpers=helpers)
    clock.patch(dual.OverlapAcquisition.plan_at_actual_dock, [(
        'base_reference_factor=_uncapped_factor,',
        'base_reference_factor=_wall_edge_reference_factor("right/acquisition_coupled_contact_clock"),', 1,
    )], 'actual coupled-clock factor evidence', helpers=helpers)
    clock.patch(dual.OverlapAcquisition.report, [(
        'Factor-two hand reference executed once during LEFT tail; completed opening held until the shared factor-two seating fade',
        'Selected-rate hand reference executed once during LEFT tail; contact timing follows the shared localized source clock', 1,
    )], 'selected overlap clock evidence')
    clock.patch(pickup.retime_prepared, [(
        'reference_factor=FACTOR, software_arm_hand_rate_limits_enforced=False,',
        'base_reference_factor=_wall_edge_reference_factor("right/acquisition_coupled_contact_clock"), software_arm_hand_rate_limits_enforced=False,', 1,
    ), ('final factor-two timing is in motion_profile.json',
        'final selected-rate timing is in motion_profile.json', 1)],
        'actual acquisition factor evidence', helpers=helpers)
    clock.patch(pickup.retime_engine_source, [(
        '"2.0*(data.time-asp_started)/.5"',
        'repr(_wall_edge_reference_factor("right/aspiration_index_preload"))+"*(data.time-asp_started)/.5"', 1,
    )], 'selected aspiration index-preload reference', helpers=helpers)

    clock.patch(interior.adapt_vertical_aspiration_source, [(
        '    replacement = ("u=smooth(2.0*t/" + repr(hover) + ");above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if 2.0*t<"\n'
        '                   + repr(hover) + " else top+axis*(.014-.029*smooth((2.0*t-" + repr(hover) + ")/2.5))")',
        '    thumb_factor = repr(_wall_edge_reference_factor("right/aspiration_hover_press"))\n'
        '    replacement = ("u=smooth(" + thumb_factor + "*t/" + repr(hover) + ");above=top+axis*.014;thumb_goal=(1-u)*initial_thumb+u*above if " + thumb_factor + "*t<"\n'
        '                   + repr(hover) + " else top+axis*(.014-.029*smooth((" + thumb_factor + "*t-" + repr(hover) + ")/2.5))")', 1,
    ), ('"axis*.024*smooth(2.0*age/3)"',
        '"axis*.024*smooth("+repr(_wall_edge_reference_factor("right/aspiration_thumb_release"))+"*age/3)"', 1),
       ('hover / FACTOR', 'hover / _wall_edge_reference_factor("right/aspiration_hover_press")', 1),
       ('2.5 / FACTOR', '2.5 / _wall_edge_reference_factor("right/aspiration_hover_press")', 1),
       ('3. / FACTOR', '3. / _wall_edge_reference_factor("right/aspiration_thumb_release")', 1),
       ('reference_factor=FACTOR, source_envelope_certificate_valid_for_execution=False',
        'reference_factor=_wall_edge_reference_factor("right/aspiration_hover_press"), source_envelope_certificate_valid_for_execution=False', 1),
       ('base_reference_factor=FACTOR, guarded_descent_reference_factor=1.0',
        'base_reference_factor=_wall_edge_reference_factor("right/aspiration_hover_press"), guarded_descent_reference_factor=1.0', 1),
       ('persistent uncapped factor-two clock', 'persistent selected-rate clock', 1),
    ], 'selected aspiration thumb references with protected descent', helpers=helpers)

    precision = right._installed_precision_type
    native_target = closure(precision.target)['target']
    clock.patch(precision._lower_clock, [
        ('_uncapped_factor', 'timing.factor', 4),
    ], 'selected lower-clock endpoint extension')
    clock.patch(precision._lower_timing, [
        ('_uncapped_factor', 'self._topp_lower.factor', 2),
    ], 'selected lower reference-rate evidence')
    clock.patch(native_target, [(
        'PROFILE.lower_native_tail_s / _uncapped_factor',
        'PROFILE.lower_native_tail_s / self._lower_timing().factor', 1,
    )], 'selected lower-tail source timing')
    clock.record('direct_reference_rate_selection', 'right/direct_reference_consumers',
                 factors={label: reference_factor(label) for label in DIRECT_LABELS},
                 physical_timestep_unchanged=True, native_feedback_and_acceptance_unchanged=True,
                 original_grasp_and_dispense_restorations_unchanged=True,
                 physical_execution_passed=None)
    _INSTALLED = True
    return PATCHED_SYMBOLS
