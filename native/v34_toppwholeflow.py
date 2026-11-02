"""Install whole-workflow TOPP after the exact frozen R08 plugin chain."""
from functools import wraps
from pathlib import Path
import sys

import v34_topp_core as core
import v34_topp_left as left
import v34_topp_right as right

PATCHED_SYMBOLS = (
    'run_t1_full_workflow_v28.main',
    't1_coarse_stop.CoarseStopController.__init__',
)
COVERAGE = [
    dict(stage='feedback_base_arrival', decision='native_feedback_clock',
         reason='Wheel commands are velocity-servo inputs, not joint-position paths. Original approach, irreversible brake latch and measured stopped gate remain authoritative.'),
    dict(stage='native_workflow_clock', decision='unchanged',
         reason='Every native callback still owns one1ms physics step; only reference evaluators receive TOPP progress.'),
]
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    left.install()
    right.install()
    import t1_coarse_stop
    previous_init = t1_coarse_stop.CoarseStopController.__init__

    @wraps(previous_init)
    def observed_init(self, *args, **kwargs):
        previous_init(self, *args, **kwargs)
        core.note_fixed('base_feedback_station_'+repr(self.station_xy.tolist()),
            kind='feedback_qualification',
            reason='Unchanged native velocity-servo feedback; original coarse-radius latch, braking and continuous measured stopped window. No base gains or limits changed.')

    t1_coarse_stop.CoarseStopController.__init__ = observed_init
    import run_t1_full_workflow_v28 as native
    previous_main = native.main

    @wraps(previous_main)
    def main(*args, **kwargs):
        try:
            return previous_main(*args, **kwargs)
        finally:
            tag = sys.argv[sys.argv.index('--tag')+1]
            directory = Path.cwd()/'logs/spincoat/t1'/tag
            directory.mkdir(parents=True, exist_ok=True)
            core.save_report(directory)

    native.main = main
    _INSTALLED = True
