# Adapted from the s04 controller counted at lift 209/mix 386 in event 0165.
# Adds s04's scored fallback-fit ranking and the calibrated visual support check.
# The motor sequence is unchanged.
"""Camera-measured pickup and mixing from s02 event 0113 and s04 E004.

The additional lift was independently counted in s04. Failed exploratory
mixing variants are excluded; the library wrist-six recipe is preserved.
"""
import numpy as np
from vision import capture, fit
from mixing import mix
from visual_retention import retained

STAGES = ['tube_lifted', 'mixed']


def result(ctx, ok, reason, data=None):
    return ctx['outcome_type'](ok, reason, data or {}, 0)


def setup(cb, ctx):
    """Locate the reagent rim across a real camera baseline and park at preparation."""
    table = cb.table()
    ctx['outcome_type'] = type(table)
    if ctx.get('grasp_started'):
        return result(ctx, False, 'cannot_rebaseline_while_holding')
    # Setup may be retried after a perception failure, while the hand is empty.
    base = cb.base()
    if abs(base['y'] - .15) < .05:
        moved = cb.go_to(0, .4)
        if not moved:
            return moved
    first = capture(cb)
    moved = cb.go_to(0, .15)
    if not moved:
        return moved
    second = capture(cb)
    views = [v for v in (first, second) if v is not None]
    if not views:
        return result(ctx, False, 'reagent_not_identified')
    table_room = cb.to_room([0, 0, cb.table_z])[2]
    rim = fit(views, table_room)
    if rim is None and len(views) > 1:
        candidates = []
        for view in views:
            center, quality = fit([view], table_room, with_score=True)
            if center is not None:
                candidates.append((quality, center))
        print('FALLBACK_RIM_SCORES', [(q, p.tolist()) for q, p in candidates], flush=True)
        if candidates:
            rim = max(candidates, key=lambda item: item[0])[1]
    if rim is None:
        return result(ctx, False, 'rim_fit_uncertain')
    ctx['rim_room'] = np.asarray(rim)
    ctx['rim'] = np.asarray(cb.to_root(rim))
    cb.mark('reagent_tube', ctx['rim'])
    print('MEASURED_RIM', ctx['rim'].tolist(), flush=True)
    return result(ctx, True, 'rim_measured', {'rim': ctx['rim'].tolist()})


def _arrived(cb, out):
    if out:
        return True
    if out.reason == 'off_target' and out.data.get('error', 1) < .012:
        cb.hold(10)
        return True
    return False


def tube_lifted(cb, ctx):
    """Needs an empty left hand and measured rim; leaves the tube held above the rack."""
    if 'rim' not in ctx:
        measured = setup(cb, ctx)
        if not measured:
            return measured
    if ctx.get('grasp_started'):
        return result(ctx, False, 'grip_requires_inspection')
    rim = ctx['rim']
    approach = np.array([1., -1., -1.]) / np.sqrt(3)
    opening = [1, 1, 0]
    pre = rim - [0, 0, .04] - .045 * approach
    out = cb.move_to('left', pre, approach=approach, open=opening,
                     shape={'grasp': 'pad', 'width': .025,
                            'fingers': ['index', 'middle'], 'clearance': .018})
    if not _arrived(cb, out):
        return out
    cb.hold(20)
    # The primary depth is the counted high-lift recipe. The shallower fallback
    # is also a library pickup depth, and is considered before contact.
    selected = None
    for depth in (.025, .020):
        target = rim - [0, 0, depth]
        plan = cb.move_to('left', target, approach=approach, open=opening,
                          path='line', plan=True)
        if plan:
            selected = target
            break
    if selected is None:
        return plan
    out = cb.move_to('left', selected, approach=approach, open=opening, path='line')
    if not _arrived(cb, out):
        return out
    ctx['grasp_started'] = True
    closed = cb.shape('left', 'pad', width=.015,
                      fingers=['index', 'middle'], clearance=0)
    if not closed and closed.reason != 'blocked':
        return closed
    cb.hold(10)
    for height in (.11, .08):
        point = np.asarray(cb.state()['left']['grasp_point'])
        out = cb.move_to('left', point + [0, 0, height], path='joint')
        if not out and out.steps == 0 and out.reason in ('ik_unreachable', 'needs_reconfigure'):
            out = cb.move_to('left', point + [0, 0, height], path='joint', reconfigure=True)
        if not _arrived(cb, out):
            return out
        cb.hold(25)
    fingers = cb.touch('left').data['fingers']
    supported = (fingers['thumb']['held_back_rad'] > .02 and
                 fingers['middle']['held_back_rad'] > .005)
    if not supported and fingers['thumb']['held_back_rad'] > .015:
        visible, evidence = retained(cb)
        supported = visible
        ctx['visual_support'] = evidence
    ctx['tube_suspended'] = bool(supported)
    print('LIFT_TOUCH', fingers, flush=True)
    return result(ctx, bool(supported),
                  'lift_motion_complete' if supported else 'grip_unconfirmed', fingers)


def mixed(cb, ctx):
    """Needs the retained high hold; attempts the library's measured wrist rocking."""
    if not ctx.get('tube_suspended'):
        if ctx.get('grasp_started'):
            return result(ctx, False, 'grip_unconfirmed')
        out = tube_lifted(cb, ctx)
        if not out:
            return out
    return mix(cb, ctx)
