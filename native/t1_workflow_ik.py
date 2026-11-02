"""Reuse the delivered margin/nullspace IK without invoking its CLI runner."""
import ast,types
import numpy as np,mujoco
from t1_spincoat_scene import ROOT

def install_validated_ik(runner,start_s):
    # This is the exact function from the already reviewed 010 wrapper, with
    # explicit closure dependencies. Keep one numerical implementation.
    path=ROOT/'scripts/t1_yaw_motion_candidate.py'
    tree=ast.parse(path.read_text());function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='margin_ik')
    body=ast.unparse(function).replace('null_step = (np.eye(7) - inverse @ J) @ posture','null_step = (np.eye(7) - inverse @ J) @ posture * float(getattr(runner, "workflow_nullspace_enabled", True))')
    module=ast.parse(body);ast.fix_missing_locations(module)
    namespace={'runner':runner,'args':types.SimpleNamespace(ik_limit_margin_rad=.04,smooth_nullspace=True,nullspace_start_s=float(start_s)),'np':np,'mujoco':mujoco}
    exec(compile(module,str(path),'exec'),namespace)
    runner.ik=namespace['margin_ik']
