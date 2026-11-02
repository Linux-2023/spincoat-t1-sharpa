"""Opt-in V8 dwell reduction. Original collision/support/precision gates retained."""
from pathlib import Path
import json

def once(s,a,b):
 if s.count(a)!=1:raise ValueError('V8 anchor mismatch '+repr(a)+' '+str(s.count(a)))
 return s.replace(a,b,1)

def adapt_left_stage(source):
 source=once(source,"  nonlocal armcmd,handcmd", "  nonlocal armcmd,handcmd\n  if name=='seat_open_realignment' and goalarm is not None and np.max(abs(np.asarray(goalarm)-armcmd))<.002:duration=.30\n  if name=='settle_original_ready':duration=.25")
 return source

def adapt_neutral(source):
 source=once(source,"engine_text=engine_text.replace(needle,\"'--max-seconds','14.375'\")", "engine_text=engine_text.replace(needle,\"'--max-seconds','11.625'\")")
 source=once(source,"        runner,path,callback,audit,clock=prepared", "        runner,path,callback,audit,clock=prepared\n        clock=dict(clock);clock['stage_end_s']=float(clock['verticalize_start_s'])+1.0\n        v8_profile=json.loads(Path(path).read_text());v8_profile['staged_grasp_timing']=clock;Path(path).write_text(json.dumps(v8_profile,indent=2))")
 return source

def adapt_launch(source):
 anchor="path=out/'executed_human_glass_stage_v4.py';path.write_text(source)"
 replacement="from t1_transition_flow_v8 import adapt_left_stage\\n        source=adapt_left_stage(source)\\n        "+anchor
 source=once(source,anchor,replacement)
 anchor=" source=Path(baseline.__file__).read_text()"
 # First source read is LEFT nested V2; second is main neutral.
 positions=[i for i in range(len(source)) if source.startswith(anchor,i)]
 if len(positions)!=2:raise ValueError('Expected LEFT and RIGHT source read')
 i=positions[0]+len(anchor)
 source=source[:i]+"\n source=changed(source,'IdleFingerOverlay(m)', 'IdleFingerOverlay(m,idle_segment_s=.6)')"+source[i:]
 anchor=" exec(compile(source,baseline.__file__+' [v3 opt-in adapter]','exec'),namespace)"
 # RIGHT adaptation must occur after all existing transformations and before snapshot/exec.
 needle=" source=adapt_neutral_source_v6(source,RightInteriorWorkstationV6("
 lines=source.splitlines();line=next(x for x in lines if x.startswith(needle))
 source=once(source,line,line+"\n from t1_transition_flow_v8 import adapt_neutral as adapt_flow_neutral\n source=adapt_flow_neutral(source)")
 compile(source,'<v8 flow launch>','exec')
 return source
