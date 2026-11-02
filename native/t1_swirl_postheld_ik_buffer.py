"""POST-HELD ONLY planner buffer. Physical model/limits/control gains untouched."""
import inspect
from t1_left_grip_human_v3 import solve_arm as original_neutral_solver
_source=inspect.getsource(original_neutral_solver)
assert _source.count('+.05;hi=')==1 and _source.count(']-.05;q0=')==1 and _source.count('margin>=.05-1e-8')==1
_source=_source.replace('def solve_arm(', 'def solve_arm_buffered(').replace('+.05;hi=', '+.051;hi=').replace(']-.05;q0=', ']-.051;q0=').replace('margin>=.05-1e-8','margin>=.051-1e-8')
_env=dict(original_neutral_solver.__globals__);exec(compile(_source,__file__+':generated_solver','exec'),_env)
solve_arm_buffered=_env['solve_arm_buffered']
GENERATED_SOLVER_SOURCE=_source
MINIMUM_POSTHELD_PLANNING_MARGIN_RAD=.051
safe_exit_solve_arm=solve_arm_buffered
