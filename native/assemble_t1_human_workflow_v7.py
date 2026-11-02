"""V7 strict ten-stage replay assembly. V6 scripts and records remain unchanged."""
from pathlib import Path
import argparse,json,types
from assemble_t1_human_workflow_v6 import ORDER,sha
BASE=Path(__file__).with_name('assemble_t1_human_workflow_v6.py')
def implementation(provenance_filename='interior_workflow_v7_provenance.json'):
 if Path(provenance_filename).name!=provenance_filename:raise ValueError('Provenance must be a local filename')
 s=BASE.read_text();s=s.replace('stage_recording_manifest_v6.json','stage_recording_manifest_v7.json').replace('interior_workflow_v6_provenance.json',provenance_filename).replace("workflow_version='V6'","workflow_version='V7'")
 ns=dict(__name__='v7_assembler_implementation',__file__=__file__);exec(compile(s,str(BASE)+' [v7 replay only]','exec'),ns);return types.SimpleNamespace(**ns)
def assemble(directory,timing_correction=None,provenance_filename='interior_workflow_v7_provenance.json'):
 impl=implementation(provenance_filename);arrays,meta,report=impl.assemble(directory,timing_correction);report.update(scope=__doc__,v6_base_implementation_sha256=sha(BASE),v7_wrapper_sha256=sha(__file__),provenance_input_name=provenance_filename)
 (Path(directory)/'stage_recording_manifest_v7.json').write_text(json.dumps(report,indent=2));return arrays,meta,report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--physical-time-artifact',type=Path);p.add_argument('--provenance-file',default='interior_workflow_v7_provenance.json');a=p.parse_args();_,_,r=assemble(a.directory,a.physical_time_artifact,a.provenance_file);print(json.dumps(r,indent=2))
