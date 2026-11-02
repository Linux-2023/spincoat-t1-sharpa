"""Opt-in v6 recorder-only fix. Controls/events/physics are unchanged; raw states get poststep clocks."""
from pathlib import Path
import inspect,hashlib

def exact(s,old,new):
 if s.count(old)!=1:raise RuntimeError('Physical recorder anchor mismatch: '+repr(old))
 return s.replace(old,new)

def correct_record_source(source,kind):
 if kind not in ('grasp','coat','coating','dispense'):raise ValueError('Unsupported child record kind')
 offset=0. if kind=='grasp' else 9.
 source=exact(source,'    records=[];events=[];last_phase=None;', '    record_clock_origin=float(d.time)\n    records=[];events=[];last_phase=None;')
 source=exact(source,"'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time':t,'phase':phase", "'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time':float(d.time)-record_clock_origin+"+repr(offset)+",'phase':phase")
 source=exact(source,"    arrays['metadata_json']=np.array(json.dumps(metadata))", "    metadata['recorded_state_time_basis']='poststep physical d.time minus explicit stage origin plus original logical offset; events unchanged'\n    metadata['physical_record_clock_origin_s']=record_clock_origin\n    metadata['recording_logical_offset_s']="+repr(offset)+"\n    arrays['metadata_json']=np.array(json.dumps(metadata))")
 compile(source,'<v6 poststep recorded clock>','exec');return source

def adapt_factory_source(source):
 anchor="    source=(ROOT/'scripts/run_t1_spincoat_episode.py').read_text()"
 return exact(source,anchor,anchor+"\n    from t1_actual_record_clock_v6 import correct_record_source\n    source=correct_record_source(source,kind)")

def install_standing_preparer_clock():
 import t1_right_standing_motion_v6 as module
 if getattr(module,'_actual_record_clock_v6_installed',False):raise RuntimeError('Duplicate physical clock install')
 source=inspect.getsource(module.prepare_standing_right_v6)
 anchor='    factory_source=inspect.getsource(ca.adapted_runner)'
 source=exact(source,anchor,anchor+"\n    from t1_actual_record_clock_v6 import adapt_factory_source\n    factory_source=adapt_factory_source(factory_source)")
 # Preserve the module globals used by its imported _ACTIVE_CONFIG/_READY_P; do not clone state.
 import linecache
 filename=str(Path(__file__).with_name('t1_right_standing_motion_v6.py'))+' [recorder only]'
 # Downstream opt-in source adapters must inspect this generated function,
 # not accidentally read an unrelated slice of the original on-disk module.
 linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
 exec(compile(source,filename,'exec'),module.__dict__)
 module._actual_record_clock_v6_installed=True
 return dict(scope=__doc__,adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),standing_preparer_snapshot=source)

def adapt_neutral_source(source):
 anchor="    engine_text=(ROOT/'scripts/run_t1_integrated_workflow.py').read_text(encoding='utf-8-sig')"
 insert="\n    from t1_actual_record_clock_v6 import adapt_factory_source,install_standing_preparer_clock\n    engine_text=adapt_factory_source(engine_text)\n    physical_record_clock_install=install_standing_preparer_clock()\n"
 return exact(source,anchor,anchor+insert)

def adapt_launch_source(source):
 anchor=' from t1_right_standing_motion_v6 import adapt_neutral_source_v6\n'
 insert=' from t1_actual_record_clock_v6 import adapt_neutral_source as adapt_physical_record_clock_source\n source=adapt_physical_record_clock_source(source)\n'
 return exact(source,anchor,anchor+insert)
