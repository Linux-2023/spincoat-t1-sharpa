"""Real left-arm/left-hand preparation; right stays at actual ready joint targets."""
import json,numpy as np
from pathlib import Path
from t1_spincoat_scene import arm_indices,hand_indices
from t1_glass_preparation_stage import build_glass_stage
from t1_bimanual_glass_preparation import BimanualGlassConfig,BimanualGlassLedger

def advance_reagent_preparation(m,d,dock,handles,out,*,plan=None,return_to_ready_joint_target=None,control_hook=None):
 if handles.get('preparation_side')!='left':raise ValueError('Left-preparation scene handles required')
 if plan is None:raise ValueError('A fresh left-hand3point traditional-tube grasp plan is required')
 _,rightq,_=arm_indices(m,'right');_,righthq,_=hand_indices(m,'right');ready=d.qpos[np.r_[rightq,righthq]].copy();fn=build_glass_stage(side='left',config_class=BimanualGlassConfig,ledger_class=BimanualGlassLedger,out=out,control_hook=control_hook);arrays,report,ledger=fn(m,d,dock,handles,out,plan=plan,return_to_ready_joint_target=return_to_ready_joint_target);indices=np.r_[rightq,righthq];report['left_hand_encoder_tracking']=fn.encoder_tracker.save() if fn.encoder_tracker is not None else None;report['manipulation_side']='left';report['right_ready_passive_joint_deviation_max_rad']=float(np.max(abs(arrays['qpos'][:,indices]-ready)));report['right_arm_hand_targets']='Held at actual ready by existing joint servos, never mirror-written or reset';report['new_left_stage_dynamic_scope']='Only this actualrun can establish left lift/mix/pour/return success; right pipette stage is separate';(Path(out)/'reagent_preparation_report.json').write_text(json.dumps(report,indent=2));return arrays,report,ledger
