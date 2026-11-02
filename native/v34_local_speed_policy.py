"""Original-time policy for the current wall/edge planner, never video retiming.

Qualification comes from the matching frozen run, not these settings alone.
Physical conditions, force, geometry and acceptance parameters are unchanged.
"""
POLICY_ID = 'wall_edge_original_time_1x'
TRANSITION_S = 0.2
ORIGINAL_SPEEDPOUR_SEGMENTS = ('tilt', 'return')
FAST_REFERENCE_FACTOR = 1.0
# Every named reference uses its nominal source clock; no accelerated exterior.
REFERENCE_FACTOR_OVERRIDES = {}
SOURCE_FRACTION_WINDOWS = {}


def accelerated_reference(label):
    return str(label).startswith(('left.', 'right/', 'overlap/'))


def reference_factor(label):
    factor = float(REFERENCE_FACTOR_OVERRIDES.get(str(label), FAST_REFERENCE_FACTOR))
    if factor != 1.0:
        raise ValueError('This publication executes only original-time references')
    return factor
