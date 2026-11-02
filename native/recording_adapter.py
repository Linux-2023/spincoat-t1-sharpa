"""Instrument only generated neutral runtime source at the proven native seam."""
from __future__ import annotations

from functools import wraps
import hashlib

from action_capture import save_json, sha256

CAPTURE = None
_INSTALLED = False


def exact(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Recapture source anchor mismatch: ' + old)
    return source.replace(old, new, 1)


def instrument_neutral(source):
    if CAPTURE is None:
        raise RuntimeError('No bound recorder instance for generated source')
    before = hashlib.sha256(source.encode()).hexdigest()
    source = exact(source,
                   'live_observation=[None,None];ignored_observation_steps=[0]',
                   "_final_action_capture=__import__('recording_adapter').CAPTURE;live_observation=[None,None];ignored_observation_steps=[0]")
    source = exact(source, 'live_observation[:]=[m,d]',
                   'live_observation[:]=[m,d];_final_action_capture.bind(m,d)')
    source = exact(source, 'result=native(m,d,*args,**kwargs)',
                   'result=_final_action_capture.step(native,m,d,*args,**kwargs)')
    save_json(CAPTURE.directory / 'source_adapter.json', {
        'neutral_before_sha256': before,
        'neutral_after_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'adapter_sha256': sha256(__file__),
        'binding': 'Existing final-model live_observation assignment inside prepare_clearance05 wrapper',
        'capture': 'Innermost native step after outer control callbacks; original post-step observer retained',
        'control_or_physics_changes': False,
        'original_task_and_safety_gates_unchanged': True,
    })
    return source


def install(capture):
    global CAPTURE, _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Recapture adapter already installed')
    import run_t1_human_workflow_v4 as workflow
    original = workflow.transformed_source
    CAPTURE = capture
    anchor = " exec(compile(source,str(archive/'executed_neutral_human_v3.py'),'exec'),namespace)"

    @wraps(original)
    def transformed_source(*args, **kwargs):
        source = original(*args, **kwargs)
        return exact(source, anchor,
                     " source=__import__('recording_adapter').instrument_neutral(source)\n" + anchor)

    workflow.transformed_source = transformed_source
    _INSTALLED = True
