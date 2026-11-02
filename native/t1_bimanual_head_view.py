"""Actuator-only head turns between the left preparation and right workstations."""
import json
from pathlib import Path
import numpy as np


class BimanualHeadView:
    def __init__(self, model, data, target=(.2361345290789732,.538), duration_s=6.):
        joints=np.array([model.joint('astribot_head_joint_'+str(i)).id for i in (1,2)])
        self.qa=model.jnt_qposadr[joints]
        self.acts=np.array([int(np.flatnonzero(model.actuator_trnid[:,0]==j)[0]) for j in joints])
        self.original=data.qpos[self.qa].copy()
        self.target=np.asarray(target,dtype=float)
        if np.any(self.target<model.jnt_range[joints,0]) or np.any(self.target>model.jnt_range[joints,1]):
            raise ValueError('Head target outside official joint ranges')
        self.duration=float(duration_s)
        self.anchor=self.original.copy()
        self.goal=self.target.copy()
        self.start=float(data.time)
        self.returning=False
        self.records=[]

    def __call__(self, model, data, phase):
        if phase in ('release_tube','withdraw_empty_hand','return_to_original_ready_joints','settle_original_ready') and not self.returning:
            self.anchor=data.qpos[self.qa].copy()
            self.goal=self.original.copy()
            self.start=float(data.time)
            self.returning=True
        u=float(np.clip((data.time-self.start)/self.duration,0.,1.))
        blend=u*u*u*(10.+u*(-15.+6.*u))
        command=self.anchor+(self.goal-self.anchor)*blend
        data.ctrl[self.acts]=command
        self.records.append((float(data.time),data.qpos[self.qa].copy(),command.copy(),str(phase)))

    def save(self, data, out):
        out=Path(out)
        arrays={key:np.asarray([r[i] for r in self.records]) for i,key in enumerate(['time','actual_head_q_rad','command_head_q_rad','phase'])}
        np.savez_compressed(out/'head_view_actual_trace.npz',**arrays)
        report=dict(scope='Six-second quintic servo-target turns overlap preparation; return begins after the tube has been seated in its rack. Existing camera extrinsics, joint limits and torque caps unchanged.',left_target_q_rad=self.target.tolist(),original_right_q_rad=self.original.tolist(),return_started=self.returning,final_original_error_rad=(data.qpos[self.qa]-self.original).tolist(),samples=len(self.records),runtime_qpos_or_camera_pose_writes=False)
        (out/'head_view_report.json').write_text(json.dumps(report,indent=2))
        return report
