"""Opt-in LEFTneutral-wrist IK and independent held diagnostic wrapper; baseline unmodified."""
import numpy as np,mujoco,json
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from t1_spincoat_scene import arm_indices
from t1_left_continuous_ik import solve_arm as ordinary_solve_arm
from t1_glass_preparation_stage import build_glass_stage
from t1_bimanual_glass_preparation import BimanualGlassConfig,BimanualGlassLedger
IK_REPORT=[]

def solve_arm(m,d,target,R,side='left',seed=None):
 if side!='left':return ordinary_solve_arm(m,d,target,R,side=side,seed=seed)
 js,qa,_=arm_indices(m,'left');work=mujoco.MjData(m);work.qpos[:]=d.qpos;pb=m.body('sharpa_left/left_hand_C_MC').id;eb=m.body('astribot_arm_left_link_5').id;wb=m.body('astribot_arm_left_link_6').id;mb=m.body('sharpa_left/left_middle_MCP_VL').id;lo=m.jnt_range[js,0]+.05;hi=m.jnt_range[js,1]-.05;q0=np.clip(np.asarray(d.qpos[qa] if seed is None else seed),lo+1e-8,hi-1e-8);target=np.asarray(target);R=np.asarray(R)
 def pose(q):
  work.qpos[qa]=q;mujoco.mj_kinematics(m,work);return np.r_[work.xpos[pb]-target,Rotation.from_matrix(work.xmat[pb].reshape(3,3)@R.T).as_rotvec()]
 def angle(q):
  pose(q);u=work.xpos[wb]-work.xpos[eb];v=work.xpos[mb]-work.xpos[pb];return float(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
 def objective(q):return 1-angle(q)+.001*np.sum((q-q0)**2)
 starts=[q0]
 try:q,_=ordinary_solve_arm(m,d,target,R,side='left',seed=q0);starts.append(q)
 except RuntimeError:pass
 good=[]
 for qstart in starts:
  fit=minimize(objective,qstart,method='SLSQP',bounds=list(zip(lo,hi)),constraints=dict(type='eq',fun=pose),options=dict(maxiter=180,ftol=1e-11));e=pose(fit.x);a=float(np.degrees(np.arccos(angle(fit.x))));margin=float(np.minimum(fit.x-m.jnt_range[js,0],m.jnt_range[js,1]-fit.x).min())
  if np.linalg.norm(e[:3])<.0001 and np.linalg.norm(e[3:])<np.deg2rad(.1) and margin>=.05-1e-8:good.append((objective(fit.x),fit.x.copy(),a,margin,e.copy()))
 if not good:raise RuntimeError('LEFTv3neutralIK no original6D/marginqualifiedsolution')
 _,q,a,margin,e=min(good,key=lambda v:v[0]);info=dict(position_error_mm=float(np.linalg.norm(e[:3])*1000),orientation_error_deg=float(np.degrees(np.linalg.norm(e[3:]))),joint_margin_rad=margin,forearm_palm_angle_deg=a,ik_method='leftv3constrained6Dneutralforearm-palmnullspace',goal_world_m=target.tolist());IK_REPORT.append(info);return q,info

def build_held_test(out,control_hook=None):
 out=Path(out);fn=build_glass_stage(side='left',config_class=BimanualGlassConfig,ledger_class=BimanualGlassLedger,out=out,control_hook=control_hook);source=(out/'executed_left_glass_stage.py').read_text();anchor="  if not held_gate['pass_']:raise RuntimeError('actual tube held/lift gate failed')";assert source.count(anchor)==1
 source=source.replace(anchor,anchor+"\n  np.savez_compressed(out/'actual_held_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),time=float(d.time))\n  raise RuntimeError('DIAGNOSTIC_STOP_AFTER_ORIGINAL_ACCEPTED_HELD_GATE:mix,pour,returnnotexecuted;notfullsuccess')")
 env=dict(fn.__globals__);env['solve_arm']=solve_arm;env['continuous_left_solve_arm']=solve_arm;env['ENCODER_TRACKER']=fn.encoder_tracker;path=out/'executed_v3_held_stage.py';path.write_text(source);exec(compile(source,str(path),'exec'),env);new=env['advance_reagent_preparation'];new.encoder_tracker=fn.encoder_tracker;return new


class ArmEncoderTracking:
    """Opt-in bounded nominal-position following compensation; never reads contacts.
    Active during close/hold only; fixed prior offset is retained from first lift onward.
    No actuator gain/cap, friction, gravity or object state modifications.
    """
    def __init__(self,m,ki=3.,bound_rad=.02):
        self.model=m;js,self.qa,_=arm_indices(m,'left');self.acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in js]);self.offset=np.zeros(7);self.ki=float(ki);self.bound=float(bound_rad);self.frozen=False;self.active=False;self.rows=[];self.last_time=None
    def __call__(self,m,d,phase):
        nominal=d.ctrl[self.acts].copy();actual=d.qpos[self.qa].copy();active=phase in ['close_on_tube','hold_before_tube_lift']
        if phase=='lift_tube':self.frozen=True
        if active:
            self.active=True
            if not self.frozen:self.offset=np.clip(self.offset+self.ki*(nominal-actual)*m.opt.timestep,-self.bound,self.bound)
        if self.active:
            out=nominal+self.offset
            for j,ac in enumerate(self.acts):
                if m.actuator_ctrllimited[ac]:out[j]=np.clip(out[j],m.actuator_ctrlrange[ac,0],m.actuator_ctrlrange[ac,1])
            d.ctrl[self.acts]=out;self.rows.append((float(d.time),phase,nominal.copy(),actual.copy(),self.offset.copy(),out.copy(),self.frozen))
    def save(self,out):
        out=Path(out);np.savez_compressed(out/'arm_encoder_tracking_v3.npz',time=np.array([r[0] for r in self.rows]),phase=np.array([r[1] for r in self.rows]),nominal_target_rad=np.array([r[2] for r in self.rows]),actual_joint_rad=np.array([r[3] for r in self.rows]),encoder_offset_rad=np.array([r[4] for r in self.rows]),actuator_ctrl_rad=np.array([r[5] for r in self.rows]),integrator_frozen=np.array([r[6] for r in self.rows]));r=dict(scope=self.__doc__,ki_per_s=self.ki,offset_bound_rad=self.bound,actual_max_abs_offset_rad=float(abs(np.array([r[4] for r in self.rows])).max()) if self.rows else 0.,frozen_final=self.frozen,samples=len(self.rows),gain_forcecap_friction_changed=False);(out/'arm_encoder_tracking_v3.json').write_text(json.dumps(r,indent=2));return r


def build_held_continuation_test(out,continuation,control_hook=None):
 """Fresh same-live original held+freeze gate then owned short continuation.
 continuation(m,d,dock,handles,armcmd,step_callback,ledger,plan,out)
 returns (final_arm_command, report). step_callback(qtarget,phase,allow_mix=False)
 only assigns original arm command then original sample; hand/dock/ledger remain live.
 """
 out=Path(out);fn=build_glass_stage(side='left',config_class=BimanualGlassConfig,ledger_class=BimanualGlassLedger,out=out,control_hook=control_hook);source=(out/'executed_left_glass_stage.py').read_text();anchor="  clear_lift=rack_clearance_lift_m(plan,float(d.xpos[body,2]))"
 assert source.count(anchor)>=1
 code="""  np.savez_compressed(out/'actual_held_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),time=float(d.time))
  def held_continuation_step(qtarget,phase_name,allow_mix=False):
   nonlocal armcmd
   armcmd=np.asarray(qtarget).copy();sample(phase_name,mix=allow_mix)
  armcmd,held_continuation_report=HELD_CONTINUATION(m,d,dock,handles,armcmd.copy(),held_continuation_step,ledger,plan,out)
  (out/'held_continuation_report.json').write_text(json.dumps(held_continuation_report,indent=2))
  raise RuntimeError('DIAGNOSTIC_STOP_AFTER_ORIGINAL_HELD_AND_CONTINUATION:not_full_task')
"""
 source=source.replace(anchor,code+anchor,1);env=dict(fn.__globals__);env['solve_arm']=solve_arm;env['continuous_left_solve_arm']=solve_arm;env['ENCODER_TRACKER']=fn.encoder_tracker;env['HELD_CONTINUATION']=continuation;path=out/'executed_v3_held_continuation_stage.py';path.write_text(source);exec(compile(source,str(path),'exec'),env);new=env['advance_reagent_preparation'];new.encoder_tracker=fn.encoder_tracker;return new
