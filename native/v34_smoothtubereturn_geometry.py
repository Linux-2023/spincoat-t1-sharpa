"""Shared post-pour return geometry classification; original helper stays intact.

Environment/crossdigit observations still reject the return plan. The helper's
pre-pour held-tube nonpad/overwall criterion is evaluated unchanged and retained
as a separate rejected/not-certified grip audit, not imposed as a new post-pour
functional gate. No geometry, live state, physics or original guard is patched.
"""
from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np

from t1_left_standing_workflow_v6 import _native
from t1_spincoat_scene import hand_indices


def classify_return_geometry(native_observations):
    """Classify semantic outputs, never exempt a geom or change a depth limit."""
    blocking, grip = [], []
    for item in native_observations:
        if item.get('kind') == 'nonpad_or_overwall':
            grip.append(item)
        else:
            # Original environment/crossdigit rows, and any unknown future
            # category, remain blocking. No exceptions or failures are caught.
            blocking.append(item)
    return blocking, grip


class ReturnGeometryAudit:
    """One audit/classifier shared by entry, dense FK and swept-interval calls."""

    def __init__(self, m, entry_work, palm, tube_name, tube_qpos, rel, E, *, entry_time_s, audit_directory):
        self.m = m
        self.palm = palm
        self.tube_name = tube_name
        self.tube_qpos = tube_qpos
        self.rel = np.asarray(rel).copy()
        self.E = np.asarray(E).copy()
        self.entry_time_s = float(entry_time_s)
        self.audit_directory = Path(audit_directory)
        _, hq, _ = hand_indices(m, 'left')
        self.fixed_finger_q = entry_work.qpos[hq].copy()
        self.entry = None
        self.screening_completed = False
        self.planning_failure = None
        self.samples_by_origin = {}
        self.blocking_observation_count = 0
        self.grip_observation_count = 0
        self.grip_pairs = {}

    def observe(self, work, *, reference_time_s, origin):
        # _native computes conditional tool FK/collisions on this isolated
        # workData and returns its original categories/threshold decisions.
        native = _native(self.m, work, self.palm, self.tube_name, self.tube_qpos, self.rel, self.E)
        blocking, grip = classify_return_geometry(native)
        time = float(reference_time_s)
        self.samples_by_origin[origin] = self.samples_by_origin.get(origin, 0) + 1
        self.blocking_observation_count += len(blocking)
        self.grip_observation_count += len(grip)
        for item in grip:
            key = (item['kind'], tuple(item['geoms']))
            depth = float(item['depth_mm'])
            if key not in self.grip_pairs:
                self.grip_pairs[key] = dict(
                    kind=item['kind'], geoms=list(item['geoms']),
                    strict_grip_geometry_status='rejected', certified=False,
                    present_at_entry=origin == 'entry', observation_count=0,
                    first_reference_time_s=time, last_reference_time_s=time,
                    minimum_depth_mm=depth, maximum_depth_mm=depth,
                    observations_by_origin={},
                )
            row = self.grip_pairs[key]
            row['observation_count'] += 1
            row['first_reference_time_s'] = min(row['first_reference_time_s'], time)
            row['last_reference_time_s'] = max(row['last_reference_time_s'], time)
            row['minimum_depth_mm'] = min(row['minimum_depth_mm'], depth)
            row['maximum_depth_mm'] = max(row['maximum_depth_mm'], depth)
            origins = row['observations_by_origin']
            origins[origin] = origins.get(origin, 0) + 1
        if self.entry is None:
            if origin != 'entry' or time != 0.0:
                raise RuntimeError('Return geometry audit must begin at its unchanged entry state')
            self.entry = dict(
                source_time_s=self.entry_time_s, reference_time_s=0.0,
                strict_grip_geometry_status='rejected' if grip else 'not_certified',
                grip_certified=False, original_native_observations=native,
                environment_or_crossdigit_observations=blocking,
                held_tube_nonpad_or_overwall_observations=grip,
            )
            # Persist exact initial observations before either any motion or a
            # later planning rejection. The completed aggregate is a new file.
            (self.audit_directory / 'return_entry_grip_geometry_audit.json').write_text(
                json.dumps(self.report(), indent=2) + '\n'
            )
        if blocking:
            raise RuntimeError('Continuous return environment/crossdigit geometry rejected at '
                               + str(time) + ': ' + str(blocking))

    def report(self):
        return dict(
            schema_version=1,
            source_helper='t1_left_standing_workflow_v6._native (unchanged)',
            classification_policy='Only original kind=nonpad_or_overwall is a held-grip audit category. Environment/crossdigit and unknown categories still reject.',
            strict_grip_geometry_status='rejected' if self.grip_observation_count else 'not_certified',
            grip_certified=False,
            original_strict_grip_observations_retained=True,
            return_functional_success_not_inferred=True,
            planning_screen_completed=self.screening_completed,
            planning_failure=self.planning_failure,
            entry=self.entry,
            samples_by_origin=dict(self.samples_by_origin),
            blocking_environment_or_crossdigit_observation_count=self.blocking_observation_count,
            held_tube_grip_observation_count=self.grip_observation_count,
            held_tube_grip_pairs=list(self.grip_pairs.values()),
            observation_storage='Exact complete entry observations; all subsequent rejected grip observations aggregated per pair with counts, depth extrema, time bounds and call-site origins. Not a raw chronological contact trace.',
            fixed_planning_finger_q_rad=self.fixed_finger_q.tolist(),
            rigid_hand_to_tube_position_m=self.rel.tolist(),
            rigid_hand_to_tube_rotation=self.E.tolist(),
            fixed_grip_limitation='The fixed-finger/rigid-grip path model cannot repair inherited tube/distal-finger overlaps by changing whole-arm timing. Neither these assumptions nor conditional tool FK is imposed on live dynamics.',
            native_scope_correction='This helper strict-grip rejection belongs to original pre-pour retarget/carry; it was newly imported as a post-pour rejection in failed parent02, not an unchanged original return predicate.',
            unchanged_authority='Original pre-pour _native calls and live held/contact/liquid/return predicates remain unchanged. Parent must compare actual support/slip with R05; audit classification cannot certify grip or manufacture functional success.',
            no_geom_allowlist=True, no_depth_threshold_change=True,
        )

    @contextmanager
    def preserve_failure(self):
        try:
            yield
        except Exception as error:
            self.screening_completed = False
            self.planning_failure = dict(type=type(error).__name__, message=str(error))
            try:
                (self.audit_directory / 'return_grip_geometry_audit.json').write_text(
                    json.dumps(self.report(), indent=2) + '\n'
                )
            except Exception as audit_error:
                error.add_note('Partial grip-audit persistence failed: ' + repr(audit_error))
            raise

    def save_completed(self):
        self.screening_completed = True
        report = self.report()
        (self.audit_directory / 'return_grip_geometry_audit.json').write_text(
            json.dumps(report, indent=2) + '\n'
        )
        return report
