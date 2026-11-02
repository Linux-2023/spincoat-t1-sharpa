"""V31 upright preturn planner; original geometric checks remain in force."""
from pathlib import Path
import t1_left_opposite_workflow_v7 as baseline


def retarget_from_live(m,d,profile):
    fraction=float(profile['v31_preturn_fraction'])
    if not 0.<fraction<=1.:raise ValueError('Preturn fraction outside (0,1]')
    source=Path(baseline.__file__).read_text()
    old='gam=gamma_end*u**3*(10+u*(-15+6*u))'
    new='gam=gamma_end*(float(p["v31_preturn_fraction"])+(1-float(p["v31_preturn_fraction"]))*u**3*(10+u*(-15+6*u)))'
    if source.count(old)!=1:raise RuntimeError('Preturn planner anchor mismatch')
    source=source.replace(old,new)
    if source.count('50-wrist_after(v)')!=1:raise RuntimeError('Preturn wrist-margin anchor mismatch')
    source=source.replace('50-wrist_after(v)','48.-wrist_after(v)')
    ns=dict(__name__='v31_preturn_planner',__file__=baseline.__file__)
    exec(compile(source,'<V31 preturn planner>','exec'),ns)
    planned,review=ns['retarget_from_live'](m,d,profile)
    planned['axial_yaw_waypoints_deg']=[[row['angle_deg'],row['gamma_deg']] for row in review['rows'][::-1]]
    planned['v31_axial_yaw_start_deg']=float(planned['axial_yaw_waypoints_deg'][0][1])
    review['v31_preturn_fraction']=fraction
    review['v31_scope']='Own same-live grasp; partial axial rotation moved to upright carry, not deleted or interpolated over a missing stage'
    return planned,review
