"""Check filled-tube visibility near the current hand in calibrated RGB views."""
import numpy as np

def retained(cb):
    cb.sync();obs=cb.robo.observe(cameras=[]);cal=cb.robo.calibration()
    state=cb.state()['left'];expected=np.asarray(state['grasp_point'])-[0,0,.10]
    joints=dict(zip(cb.arm_names,obs.arm_q));joints.update(dict(zip(cb.dex_names,obs.dex_q)));cb.robot.fk(joints)
    views={}
    for camera in ['head','left_wrist']:
        c=cal[camera];K=np.asarray(c['K'])
        if c['mount']=='world':C=np.asarray(c['pos']);R=np.asarray(c['rot'])
        else:
            p,Rp=cb.robot.pose(cb.robot.body(c['parent_link']));C=p+Rp@np.asarray(c['offset_xyz']);R=Rp@np.asarray(c['offset_rot'])
        matches=[]
        for item in cb.robo.segment(camera,['orange liquid','red liquid'],min_score=.08):
            if item['area']<60:continue
            uv=np.r_[item['centroid'],1.];d=R@np.linalg.solve(K,uv);d/=np.linalg.norm(d)
            distance=float(np.dot(expected-C,d));gap=float(np.linalg.norm(expected-(C+distance*d)))
            if distance>.05 and gap<.055:
                matches.append({'score':item['score'],'gap_m':gap,'centroid':item['centroid'],'prompt':item['prompt']})
        views[camera]=min(matches,key=lambda m:m['gap_m']) if matches else None
    print('VISUAL_RETENTION',views,flush=True)
    return all(views.values()),views
