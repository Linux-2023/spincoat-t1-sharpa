"""Opt-in v5 phase ordering only: retain folded idle fingers until empty-hand rack withdrawal.
Original API defaults and all models/controls/gates remain unchanged.
"""
import inspect
import t1_human_pour_profile_v4 as original

RESTORE_BLOCK="""    for phase in ['human_pour_idle_restore_fold','human_pour_idle_restore_extend']:
        for _ in range(int(round(profile.get('idle_segment_s',1.2)/m.opt.timestep))):step_callback(cmd,phase)
"""
def _build_deferred_pour():
 source=inspect.getsource(original.advance_human_pour_v4)
 if source.count(RESTORE_BLOCK)!=1:raise RuntimeError('Exact original pour restore anchor must occur once')
 source=source.replace(RESTORE_BLOCK,'')
 env=dict(vars(original));exec(compile(source,__file__+' [original API except deferred restore]','exec'),env)
 return env['advance_human_pour_v4']
_DEFERRED_POUR=_build_deferred_pour()
def advance_human_pour_deferred_restore_v5(*args,**kwargs):
 cmd,report,trace=_DEFERRED_POUR(*args,**kwargs)
 report=dict(report);report['idle_restore_deferred_until']='after release_tube and withdraw_empty_hand; before return_to_original_ready_joints'
 return cmd,report,trace

def adapt_stage_source(source):
 anchor="phase('withdraw_empty_hand',2.,goal(tube_initial+[0,0,.15]-returnR@root_local,returnR));phase('return_to_original_ready_joints'"
 if source.count(anchor)!=1:raise RuntimeError('Empty-hand withdrawal anchor must occur exactly once')
 replacement="phase('withdraw_empty_hand',2.,goal(tube_initial+[0,0,.15]-returnR@root_local,returnR));phase('human_pour_idle_restore_fold',float(HUMAN_POUR_PROFILE.get('idle_segment_s',1.2)));phase('human_pour_idle_restore_extend',float(HUMAN_POUR_PROFILE.get('idle_segment_s',1.2)));phase('return_to_original_ready_joints'"
 return source.replace(anchor,replacement)

def adapt_launch_source(source):
 api='from t1_human_pour_profile_v4 import advance_human_pour_v4 as advance_human_pour_v2'
 newapi='from t1_deferred_idle_restore_v5 import advance_human_pour_deferred_restore_v5 as advance_human_pour_v2'
 if source.count(api)!=1:raise RuntimeError('Generated API import anchor must occur once')
 source=source.replace(api,newapi)
 anchor=' source,_=adapt_loaded_body_stage_source(source)\n return source\n'
 if source.count(anchor)!=1:raise RuntimeError('Outer generated-stage adapter anchor must occur once')
 source=source.replace(anchor,' source,_=adapt_loaded_body_stage_source(source)\n from t1_deferred_idle_restore_v5 import adapt_stage_source as adapt_deferred_idle_restore_source\n source=adapt_deferred_idle_restore_source(source)\n return source\n')
 return source
