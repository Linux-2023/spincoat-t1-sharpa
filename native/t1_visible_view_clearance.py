"""Use the original RGB quality gate before requesting an arm clearance move."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from t1_visual_relocalization import capture_relocalization
from t1_neutral_view_clearance import advance_view_clearance


def advance_view_clearance_if_needed(m,d,dock,ledger,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=int(m.body('chassis_base').id)
    vision=capture_relocalization(m,d,d.xpos[base],d.xmat[base].reshape(3,3),out=out/'initial_visibility')
    (out/'initial_visibility.json').write_text(json.dumps(vision,indent=2))
    if not vision['measurement']['valid']:
        return advance_view_clearance(m,d,dock,ledger,out)
    wheel=np.array([a for a in range(m.nu) if m.actuator(a).name.startswith('astribot_chassis_motor_')])
    dv=int(m.joint('chassis_free').dofadr[0]);qa=int(m.joint('chassis_free').qposadr[0])
    speed=float(np.linalg.norm(d.qvel[dv:dv+2]));yaw=float(abs(d.qvel[dv+5]))
    workspace=float(np.linalg.norm(d.qpos[qa:qa+2]-np.asarray(dock.station_xy)))
    digits=['index','middle','ring','pinky'];counts=np.zeros(4,int);forces=np.zeros(4)
    for ci,c in enumerate(d.contact):
        ns=[m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom]
        if not any(n.startswith('pipette/') for n in ns):continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
        for i,name in enumerate(digits):
            if any(n.startswith('sharpa_right/right_'+name+'_') for n in ns):counts[i]+=1;forces[i]+=np.linalg.norm(f[:3])
    stopped=bool(dock.stopped and getattr(dock,'mode',None)=='zero_speed_hold')
    wheelmax=float(np.max(abs(d.ctrl[wheel])))
    success=bool(stopped and speed<.001 and yaw<.002 and workspace<=.010 and wheelmax<=1e-12 and np.all(forces>.01))
    arrays=dict(time=np.array([float(d.time)]),qpos=d.qpos[None,:].copy(),qvel=d.qvel[None,:].copy(),ctrl=d.ctrl[None,:].copy(),phase=np.array(['view_already_visible']),tip=d.site_xpos[int(m.site('pipette/tip_site').id)][None,:].copy())
    report=dict(success=success,skipped=True,skip_reason='Original onboard RGB image-quality gate already passed before any clearance movement',start_time_s=float(d.time),end_time_s=float(d.time),physical_clock_advanced_s=0.,fullrate_samples=0,instantaneous_four_finger_contact_counts=counts.tolist(),instantaneous_four_finger_force_N=forces.tolist(),base_speed_mm_s=speed*1000,base_yaw_rate_rad_s=yaw,base_workspace_mm=workspace*1000,wheel_cmd_max_abs=wheelmax,final_stopped_gate=stopped,original_rgb_measurement=vision['measurement'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Zero-motion branch; only current contacts/base gate evaluated here. Continuous neighboring-stage observer and later precision gates remain required. Camera extrinsics, object states and controller targets unchanged.')
    (out/'view_clearance_report.json').write_text(json.dumps(report,indent=2))
    return arrays,report
