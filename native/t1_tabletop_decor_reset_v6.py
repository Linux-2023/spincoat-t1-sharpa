"""Precompile-only decor relocation. Native BVH is rebuilt by MjSpec.compile.
No runtime model mutation, physics masking or controller/gate changes.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
DEFAULT=Path('/home/agilex/spincoat-t1-sharpa/research/hardware/tabletop_layout_v6/complete_unified_06/unified_layout_candidate.json')

def apply_decor_to_spec(spec,model,data,proposal=DEFAULT):
    if float(data.time)!=0.:raise ValueError('Decor spec edits are RESET-only')
    plan=json.loads(Path(proposal).read_text()) if not isinstance(proposal,dict) else proposal
    mujoco.mj_forward(model,data)
    pending=[];seen=set()
    for group in plan['decor_exact_geom_translations']:
        delta=np.asarray(group['delta_world_m'],float)
        for name in group['geom_names']:
            if name in seen:raise ValueError('Duplicate relocation '+name)
            seen.add(name);idx=model.geom(name).id;geom=spec.geom(name)
            if geom is None:raise ValueError('Missing spec geom '+name)
            if geom.frame is not None:raise ValueError('Explicit frame relocation not reviewed '+name)
            expected=np.asarray(group['source_geom_world_centers_m'][name])
            if np.linalg.norm(data.geom_xpos[idx]-expected)>1e-7:raise ValueError('Unexpected/double-shift anchor '+name)
            body=int(model.geom_bodyid[idx]);local=data.xmat[body].reshape(3,3).T@delta
            pending.append((name,geom,local,(data.geom_xpos[idx]+delta).copy()))
    hoses=[]
    for row in plan.get('spinner_service_hoses',[]):
        name=row['geom_name'];idx=model.geom(name).id;geom=spec.geom(name)
        if model.geom_contype[idx] or model.geom_conaffinity[idx]:raise ValueError('Refuse changing physical hose')
        if geom.frame is not None:raise ValueError('Unreviewed hose frame')
        body=int(model.geom_bodyid[idx]);R=data.xmat[body].reshape(3,3);origin=data.xpos[body]
        endpoints=np.asarray(row['world_endpoints_m'],float)
        local=(endpoints-origin)@R
        hoses.append((name,geom,local,np.asarray(row['new_center_world_m'])))
    # All anchors are validated before changing the independent spec.
    for name,geom,delta,expected in pending:
        if np.isfinite(np.asarray(geom.fromto)[0]):
            geom.fromto[:]=np.asarray(geom.fromto)+np.tile(delta,2)
        else:geom.pos[:]=np.asarray(geom.pos)+delta
    for name,geom,endpoints,expected in hoses:geom.fromto[:]=endpoints.reshape(6)
    return dict(phase='RESET_PRECOMPILE',translated_decor_geoms=len(pending),rerouted_visual_hoses=len(hoses),native_bvh_requires_compile=True,compiled_expected_world_centers_m={name:expected.tolist() for name,geom,delta,expected in pending+hoses},runtime_model_mutated=False)

def verify_compiled_decor(model,data,report,tolerance=1e-7):
    mujoco.mj_forward(model,data)
    errors={name:float(np.linalg.norm(data.geom_xpos[model.geom(name).id]-np.asarray(xyz))) for name,xyz in report['compiled_expected_world_centers_m'].items()}
    if max(errors.values(),default=0.)>tolerance:raise RuntimeError('Compiled decor failed world-position verification')
    return dict(verified_geoms=len(errors),maximum_world_position_error_m=max(errors.values(),default=0.),native_bvh_from_new_compile=True)
