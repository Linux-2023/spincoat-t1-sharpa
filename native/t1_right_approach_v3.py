"""Independent initial ready retreat for the moved RIGHT pipette station.

The original smooth curve, finger targets, close timing and gates are retained.
Extra wrist retreat keeps early open-pinky turning away from the closer station.
"""
import inspect
from pathlib import Path

def install_ready_retreat(smooth_module,source_path,retreat_x_m=.08,preclose_retreat_x_m=.002):
    original=smooth_module.build_continuous_profile;source=inspect.getsource(original)
    anchor='start[0]-=.02'
    if source.count(anchor)!=1:raise ValueError('Unsupported smooth initial retreat source')
    source=source.replace(anchor,'start[0]-='+repr(float(retreat_x_m)))
    anchor_end='end=work.xpos[palm].copy();R1='
    if source.count(anchor_end)!=1:raise ValueError('Unsupported smooth preclose endpoint source')
    source=source.replace(anchor_end,'end=work.xpos[palm].copy();end[0]-='+repr(float(preclose_retreat_x_m))+';R1=')
    source=source.replace("'initial_ready_retreat_x_m':.02","'initial_ready_retreat_x_m':"+repr(float(retreat_x_m)))
    metric="'initial_ready_retreat_x_m':"+repr(float(retreat_x_m))
    source=source.replace(metric,metric+",'preclose_palm_retreat_x_m':"+repr(float(preclose_retreat_x_m)))
    path=Path(source_path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    env=dict(original.__globals__);exec(compile(source,str(path),'exec'),env);smooth_module.build_continuous_profile=env[original.__name__]
    return smooth_module.build_continuous_profile
