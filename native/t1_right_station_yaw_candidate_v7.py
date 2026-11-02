"""Opt-in RESET planning adapter for one75deg stand/yaw15 RIGHT candidate.
Original v6 source/defaults remain unchanged; no running physical state writes.
"""
import inspect
from pathlib import Path
import t1_right_standing_motion_v6 as original

def prepare_standing_right_v7_yaw(m,d,out,plan,time_scale=1.25,config=None):
    if config is None or config.stand_yaw_deg!=75.:raise ValueError('Only explicitly configured75deg candidate is authorized here')
    source=inspect.getsource(original.prepare_standing_right_v6)
    name='def prepare_standing_right_v6('
    anchor='fn.angle=60.;fn.closing_scale=1.'
    if source.count(name)!=1 or source.count(anchor)!=1:raise RuntimeError('Frozen original v6 preparer anchor changed')
    source=source.replace(name,'def _prepare_standing_right_yaw75_v7(',1).replace(anchor,'fn.angle=float(c.stand_yaw_deg);fn.closing_scale=1.',1)
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'executed_standing_yaw_preparer_v7.py').write_text(source)
    # Same globals as original are required by its generated source imports.
    # This adds a private function name; original function and source are intact.
    exec(compile(source,'[private v7 RESET standing yaw75 preparer]','exec'),original.__dict__)
    return original.__dict__['_prepare_standing_right_yaw75_v7'](m,d,out,plan,time_scale=time_scale,config=config)
