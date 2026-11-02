"""Preserve a native grasp rejection before entering post-grasp passive settling.

The original integration engine already rejects either failure condition. Return
its unchanged child result instead of running a success-only handoff afterward.
Successful trajectories, passive-settle guards and all physical gates are intact.
"""
from functools import wraps

INSTALLED = False
PASSIVE_IMPORT = "            from t1_right_passive_settle_v5 import passive_wait_before_gate\n"
FAILURE_RETURN = (
    '            if result["failure_reason"] or result["robot_station_contact_violation"]:\n'
    '                return raw, result\n'
)


def install():
    global INSTALLED
    if INSTALLED:
        return
    import t1_right_passive_settle_v5 as passive

    previous = passive.adapt_neutral_source

    @wraps(previous)
    def propagate_native_failure(source, **kwargs):
        transformed = previous(source, **kwargs)
        if transformed.count(PASSIVE_IMPORT) != 1:
            raise RuntimeError('Expected one native post-grasp passive-settle entry')
        return transformed.replace(PASSIVE_IMPORT, FAILURE_RETURN + PASSIVE_IMPORT, 1)

    passive.adapt_neutral_source = propagate_native_failure
    INSTALLED = True
