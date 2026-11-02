"""Actual pre-step evidence for LEFT preparation and secondary RIGHT activity.

Record one sample for every consecutive physical tick, then finish at the actual
exclusive end. Recorded ticks and terminal bounds are nonnegative built-in ints;
no bool, float or string coercion is performed. Validate deserialized intervals
once before rendering, then treat that list as read-only during phase lookup.
"""

__all__ = ['ActivityTimeline', 'validate_intervals', 'phase_at_tick']

_PRIMARY_STAGE = 'reagent_preparation'
_SECONDARY_STAGE = 'right_preposition'
_SECONDARY_PHASES = frozenset((
    'right_home_hold', 'right_preposition', 'right_staging_hold',
))
_INTERVAL_FIELDS = frozenset((
    'primary_stage', 'primary_phase', 'secondary_stage', 'secondary_phase',
    'start_tick', 'end_tick',
))


def _require_tick(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative built-in integer')


def _require_phases(primary_phase, secondary_phase):
    if not isinstance(primary_phase, str) or not primary_phase.strip():
        raise ValueError('primary_phase must be a nonempty string')
    if (not isinstance(secondary_phase, str)
            or secondary_phase not in _SECONDARY_PHASES):
        raise ValueError('secondary_phase must name a supported RIGHT activity')


class ActivityTimeline:
    """Coalesce consecutive samples only while both phase labels are unchanged.

    Invalid arguments raise ValueError without advancing the recorder. Access
    before finish, recording after finish and repeated finish raise RuntimeError.
    Both mutating methods return None; ``intervals`` supplies the completed list.
    """

    def __init__(self):
        self._intervals = []
        self._last_tick = None
        self._finished = False

    def record_tick(self, tick, primary_phase, secondary_phase):
        """Record the actual pre-step tick, including unchanged phase pairs."""
        if self._finished:
            raise RuntimeError('cannot record after finish')
        _require_tick(tick, 'tick')
        _require_phases(primary_phase, secondary_phase)
        if self._last_tick is not None and tick != self._last_tick + 1:
            raise ValueError('recorded ticks must be strictly consecutive')
        previous = self._intervals[-1] if self._intervals else None
        if (previous is None or previous['primary_phase'] != primary_phase
                or previous['secondary_phase'] != secondary_phase):
            if previous is not None:
                previous['end_tick'] = tick
            self._intervals.append({
                'primary_stage': _PRIMARY_STAGE,
                'primary_phase': primary_phase,
                'secondary_stage': _SECONDARY_STAGE,
                'secondary_phase': secondary_phase,
                'start_tick': tick,
            })
        self._last_tick = tick

    def finish(self, end_tick):
        """Close the last interval at exactly last recorded tick + 1."""
        if self._finished:
            raise RuntimeError('timeline is already finished')
        _require_tick(end_tick, 'end_tick')
        if self._last_tick is None:
            raise ValueError('cannot finish without a recorded tick')
        if end_tick != self._last_tick + 1:
            raise ValueError('end_tick must equal the last recorded tick + 1')
        self._intervals[-1]['end_tick'] = end_tick
        self._finished = True

    @property
    def intervals(self):
        """Return a JSON-ready snapshot of complete intervals after finish."""
        if not self._finished:
            raise RuntimeError('intervals are unavailable until finish')
        return [dict(interval) for interval in self._intervals]


def validate_intervals(intervals, terminal_tick=None):
    """Return validated six-field intervals, or raise ValueError on bad input.

    The input must be a list of dictionaries with the exact contract fields.
    Intervals are nonempty, ordered and contiguous, but need not start at zero
    or extend to terminal_tick. An empty list represents no secondary activity.
    A supplied terminal_tick is an inclusive upper bound on exclusive ends.
    Validation never sorts, fills gaps or changes recorded phase labels.
    """
    if not isinstance(intervals, list):
        raise ValueError('intervals must be a list')
    if terminal_tick is not None:
        _require_tick(terminal_tick, 'terminal_tick')
    validated = []
    previous_end = None
    for index, interval in enumerate(intervals):
        if (not isinstance(interval, dict)
                or set(interval) != _INTERVAL_FIELDS):
            raise ValueError(f'interval {index} must have the six contract fields')
        if (interval['primary_stage'] != _PRIMARY_STAGE
                or interval['secondary_stage'] != _SECONDARY_STAGE):
            raise ValueError(f'interval {index} has invalid stage identifiers')
        _require_phases(interval['primary_phase'], interval['secondary_phase'])
        start = interval['start_tick']
        end = interval['end_tick']
        _require_tick(start, f'interval {index} start_tick')
        _require_tick(end, f'interval {index} end_tick')
        if start >= end:
            raise ValueError(f'interval {index} must have start_tick < end_tick')
        if previous_end is not None and start != previous_end:
            raise ValueError(f'interval {index} must start at the previous end_tick')
        if terminal_tick is not None and end > terminal_tick:
            raise ValueError(f'interval {index} extends beyond terminal_tick')
        validated.append(dict(interval))
        previous_end = end
    return validated


def phase_at_tick(validated_intervals, tick):
    """Return the matching interval dict, or None outside secondary activity.

    Bounds are start-inclusive/end-exclusive. The caller supplies intervals
    already checked by validate_intervals and leaves them unchanged. Binary
    search takes O(log n) indexed reads without revalidating or copying rows.
    Lookup accepts any built-in integer, including ticks before activity starts.
    Other tick types raise ValueError, even when the activity list is empty.
    """
    if type(tick) is not int:
        raise ValueError('tick must be a built-in integer')
    low = 0
    high = len(validated_intervals)
    while low < high:
        middle = (low + high) // 2
        interval = validated_intervals[middle]
        if tick < interval['start_tick']:
            high = middle
        elif tick >= interval['end_tick']:
            low = middle + 1
        else:
            return interval
    return None
