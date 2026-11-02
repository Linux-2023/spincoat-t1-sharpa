"""One-world RIGHT preposition during the unchanged LEFT withdrawal tail.

Only declared RIGHT actuator targets are composed before the original native
step. This module never advances physics or writes pose, velocity, time, gains,
friction, or camera state. Full LEFT and subsequent RIGHT gates remain native.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BPoly

import v34_speedrightpickup as pickup
import v34_topp_core as core
from v34_topp_right_paths import prefix_path, shifted_clock, smooth
from v34_overlap_timeline import ActivityTimeline

SESSION = None
ARM_FRACTION = .8
PREPOSITION_SECONDS = 4.0


def full_entry_envelope(profile, original_ready, acquisition, label):
    original = dict(profile)
    original['continuous_approach'] = dict(profile['continuous_approach'])
    points = np.asarray(profile['continuous_approach']['arm_q_rad']).copy()
    points[0] = original_ready
    original['continuous_approach']['arm_q_rad'] = points.tolist()
    duration = float(acquisition.original_timing['seat_start_s'])
    p = acquisition.profile
    fast = p['entry_native_s']-p['entry_native_tail_s']-p['entry_saved_s']
    clock = shifted_clock(duration, fast, p['entry_saved_s'], start=p['home_s'])
    timing = core.retime(prefix_path(original), duration=duration, label=label,
                         clock=clock, vlim=p['arm_rate_limit_rad_s'],
                         protected=((0., duration),))
    timing.report['executed_reference'] = False
    timing.report['scope'] = 'Unexecuted original full-entry envelope; limits retained for smaller scheduled motions.'
    return timing, clock


class OverlapSession:
    def __init__(self, model, data, runner, profile_path, out):
        from t1_spincoat_scene import arm_indices, hand_indices
        self.model, self.data = model, data
        self.runner, self.path, self.out = runner, Path(profile_path), Path(out)
        self.dt = float(model.opt.timestep)
        self.dt_ns = round(self.dt*1e9)
        if self.dt_ns != 1000000:
            raise RuntimeError('Overlap requires the original 1ms native model')
        self.arm_joints, self.arm_q, self.arm_v = map(np.asarray, arm_indices(model, 'right'))
        self.hand_joints, self.hand_q, self.hand_v = map(np.asarray, hand_indices(model, 'right'))
        self.arm_acts = pickup._actuators(model, self.arm_joints)
        self.hand_acts = pickup._actuators(model, self.hand_joints)
        self.owned_acts = np.r_[self.arm_acts, self.hand_acts]
        self.other_acts = np.setdiff1d(np.arange(model.nu), self.owned_acts)
        self.opened, self.closed = pickup._native_hand_endpoints(runner, model, self.hand_joints)
        self.mcp = np.asarray([i for i,j in enumerate(self.hand_joints)
                               if 'thumb' not in model.joint(j).name and 'MCP_FE' in model.joint(j).name])
        profile = json.loads(self.path.read_text())
        self.original_ready = np.asarray(profile['waypoints'][0]['arm_q_rad']).copy()
        self.nominal_endpoint = np.asarray(profile['continuous_approach']['arm_q_rad'][-1]).copy()
        self.passive_ids = None
        self.right_bodies = set()
        roots = set(int(model.jnt_bodyid[j]) for j in self.arm_joints)
        for body in range(1,model.nbody):
            parent = body
            while parent:
                if parent in roots:
                    self.right_bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.active = False
        self.finished = False
        self.failure = None
        self.start_tick = None
        self.last_tick = None
        self.pending = None
        self.timeline = ActivityTimeline()
        self.timeline_finished = False
        self.rows, self.events, self.contact_events = [], [], []
        self.qualification = dict(release_gate=False, initial_hold_completed=False,
            preshape_reference_completed=False, arm_reference_completed=False,
            original_bounds_passed=None, command_ownership_passed=None,
            precontact_passed=None, right_environment_passed=None,
            interarm_contact_passed=None, native_staging_tracking_guard_passed=None)
        self.dock_observation_count = 0
        self.last_secondary = None

    def identity(self, m, d):
        if m is not self.model or d is not self.data:
            raise RuntimeError('Overlap cannot cross model/data identities')

    def tick(self):
        value = round(float(self.data.time)/self.dt)
        if abs(float(self.data.time)-value*self.dt)>1e-7:
            self.fail('Physical clock is not on the original tick grid')
        return int(value)

    def event(self, event, **details):
        self.events.append(dict(event=event, boundary_tick=self.tick(),
                                physical_time_s=float(self.data.time), **details))

    def fail(self, message):
        self.failure = str(message)
        raise RuntimeError(message)

    def release(self, dock, tube_body, tube_initial):
        if self.start_tick is not None:
            self.fail('Overlap release boundary was reached twice')
        error = float(np.linalg.norm(self.data.xpos[tube_body]-tube_initial))
        if error >= .005 or not dock.stopped or dock.mode != 'zero_speed_hold':
            self.fail('Tube release requires the original return-position and stopped-base conditions')
        for contact in self.data.contact:
            bodies = [int(self.model.geom_bodyid[int(g)]) for g in contact.geom]
            names = [self.model.body(b).name for b in bodies]
            if tube_body in bodies and any(name.startswith('sharpa_left/') for name in names):
                self.fail('Tube remains in native LEFT hand contact at overlap release')
        ready_error = float(np.max(np.abs(self.data.qpos[self.arm_q]-self.original_ready)))
        if ready_error > .03:
            self.fail('Original RIGHT ready guard failed before early motion')
        self.start_tick = self.tick()
        self.arm_start = self.data.ctrl[self.arm_acts].copy()
        self.arm_goal = self.arm_start+ARM_FRACTION*(self.nominal_endpoint-self.arm_start)
        self.home_hand = self.data.qpos[self.hand_q].copy()
        self.nominal_hand = self.home_hand.copy()
        self.hand_command = self.home_hand.copy()
        self.arm_command = self.arm_start.copy()
        self.home_s = float(self.runner._topp_acquisition.profile['home_s'])
        self.preshape_start = float(self.runner._topp_acquisition.profile['preshape_start_s'])
        self.preshape_duration = float(self.runner._topp_acquisition.profile['preshape_duration_s'])
        self.hand_terminal_tick = round((self.preshape_start+self.preshape_duration)/self.dt)
        if abs(self.home_s-.7)>1e-12 or abs(self.preshape_duration-3.2)>1e-12:
            self.fail('Native RIGHT preshape contract changed')
        envelope, _ = full_entry_envelope(json.loads(self.path.read_text()), self.original_ready,
            self.runner._topp_acquisition, 'overlap/unexecuted_full_entry_envelope')
        zeros = np.zeros_like(self.arm_start)
        curve = BPoly.from_derivatives([0.,self.home_s,PREPOSITION_SECONDS],
            [[self.arm_start,zeros,zeros],[self.arm_start,zeros,zeros],[self.arm_goal,zeros,zeros]],
            extrapolate=False)
        self.arm_timing = core.retime(curve, duration=PREPOSITION_SECONDS,
            label='overlap/right_arm_preposition', vlim=pickup.PROFILE['arm_rate_limit_rad_s'],
            alim=envelope.report['acceleration_limits'], protected=((0.,PREPOSITION_SECONDS),))
        self.arm_terminal_tick = round(self.arm_timing.duration/self.dt)
        self.qualification['release_gate'] = True
        self.event('release_boundary', tube_return_error_mm=error*1000,
                   original_ready_error_rad=ready_error, native_left_tube_contact_count=0)
        self.event('preposition_started', arm_fraction=ARM_FRACTION,
                   arm_reference_duration_s=self.arm_timing.duration,
                   original_hand_program_duration_s=self.preshape_start+self.preshape_duration)
        self.active = True

    def left_defaults(self, robotacts, initial_ctrl):
        if not self.active:
            self.data.ctrl[robotacts] = initial_ctrl[robotacts]
            return
        if self.passive_ids is None:
            self.passive_ids = np.setdiff1d(robotacts,self.owned_acts)
        self.data.ctrl[self.passive_ids] = initial_ctrl[self.passive_ids]

    def before_step(self, phase):
        if not self.active:
            return
        tick = self.tick()
        if self.pending is not None or (self.last_tick is not None and tick != self.last_tick+1):
            self.fail('Overlap control tick was duplicated, skipped or left unobserved')
        relative = tick-self.start_tick
        age = relative*self.dt
        if relative % 20 == 0:
            self.arm_command = np.asarray(self.arm_timing(min(age,self.arm_timing.duration))).copy()
            u = smooth((age-self.preshape_start)/self.preshape_duration)
            self.nominal_hand = (1.-u)*self.home_hand+u*self.opened
        opening = smooth((age-self.preshape_start)/1.1)
        self.hand_command = self.nominal_hand.copy()
        self.hand_command[self.mcp] -= .1*opening
        pickup._check_positions(self.model,self.arm_joints,self.arm_acts,self.arm_command)
        pickup._check_positions(self.model,self.hand_joints,self.hand_acts,self.hand_command)
        self.qualification['original_bounds_passed'] = True
        other = self.data.ctrl[self.other_acts].copy()
        self.data.ctrl[self.arm_acts] = self.arm_command
        self.data.ctrl[self.hand_acts] = self.hand_command
        if not np.array_equal(other,self.data.ctrl[self.other_acts]):
            self.qualification['command_ownership_passed'] = False
            self.fail('RIGHT overlap changed an unowned actuator')
        secondary = ('right_home_hold' if age < self.home_s else
                     'right_preposition' if relative <= self.arm_terminal_tick else 'right_staging_hold')
        self.pending = (tick,str(phase),secondary,age,.1*opening)

    def observe_contacts(self, stage, phase):
        for ci,contact in enumerate(self.data.contact):
            bodies = [int(self.model.geom_bodyid[int(g)]) for g in contact.geom]
            names = [self.model.body(b).name for b in bodies]
            hand_pipe = (any(n.startswith(('sharpa_left/','sharpa_right/')) for n in names)
                         and any(n.startswith('pipette/') for n in names))
            right = any(b in self.right_bodies for b in bodies)
            environment = right and any(n.startswith(('lab_bench','stand/','spincoater/','aspiration_source','prep/')) for n in names)
            interarm = right and any(('left' in n and n.startswith(('astribot_','sharpa_'))) for n in names)
            if not (hand_pipe or environment or interarm):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model,self.data,ci,force)
            magnitude = float(np.linalg.norm(force[:3]))
            row = dict(state_tick=self.tick(),time_s=float(self.data.time),primary_stage=stage,
                       primary_phase=phase,bodies=names,geometries=[self.model.geom(int(g)).name for g in contact.geom],
                       force_N=magnitude,penetration_m=max(0.,-float(contact.dist)))
            self.contact_events.append(row)
            if hand_pipe:
                self.qualification['precontact_passed'] = False
                self.fail('Early RIGHT program/dock violated original all-contact prefix clearance')
            if environment and magnitude>.001:
                self.qualification['right_environment_passed'] = False
                self.fail('Early RIGHT program violated original RIGHT environmental force threshold')
            if interarm and magnitude>.001:
                self.qualification['interarm_contact_passed'] = False
                self.fail('Concurrent arms made force-bearing contact')
        self.qualification.update(precontact_passed=True, right_environment_passed=True,
                                  interarm_contact_passed=True)

    def after_step(self, phase):
        if not self.active:
            return
        if self.pending is None:
            self.fail('Overlap post-step observer has no corresponding command tick')
        tick,primary,secondary,age,offset = self.pending
        self.pending = None
        if self.tick()!=tick+1 or primary!=str(phase):
            self.fail('Overlap observer did not follow exactly one original native step')
        mismatch = int(not (np.array_equal(self.data.ctrl[self.arm_acts],self.arm_command)
                           and np.array_equal(self.data.ctrl[self.hand_acts],self.hand_command)))
        self.timeline.record_tick(tick,primary,secondary)
        self.last_tick = tick
        self.rows.append((tick,tick+1,float(self.data.time),primary,secondary,
            self.arm_command.copy(),self.data.qpos[self.arm_q].copy(),self.data.qvel[self.arm_v].copy(),
            self.nominal_hand.copy(),self.data.ctrl[self.hand_acts].copy(),
            self.data.qpos[self.hand_q].copy(),self.data.qvel[self.hand_v].copy(),offset,mismatch))
        if mismatch:
            self.qualification['command_ownership_passed'] = False
            self.fail('Native step overwrote the composed RIGHT command')
        self.qualification['command_ownership_passed'] = True
        if secondary=='right_preposition' and self.last_secondary=='right_home_hold':
            self.qualification['initial_hold_completed'] = True
        if (tick-self.start_tick > round(self.home_s/self.dt)
                and (tick-self.start_tick)%20 == 0
                and not np.array_equal(self.arm_command,self.arm_start)
                and not any(event['event']=='arm_motion_started' for event in self.events)):
            self.event('arm_motion_started',basis='First changed native-cadence arm command after the original initial hold; actual joint states retained in trace')
        self.last_secondary = secondary
        if tick-self.start_tick>=self.hand_terminal_tick and not self.qualification['preshape_reference_completed']:
            if not np.array_equal(self.nominal_hand,self.opened) or abs(offset-.1)>1e-12:
                self.fail('Native terminal hand program was not actually applied')
            self.qualification['preshape_reference_completed'] = True
            self.event('preshape_reference_completed',actual_hand_q_rad=self.data.qpos[self.hand_q].tolist(),
                       commanded_hand_q_rad=self.hand_command.tolist(),tracking_certification=False)
        if tick-self.start_tick>=self.arm_terminal_tick and not self.qualification['arm_reference_completed']:
            self.qualification['arm_reference_completed'] = True
            self.event('preposition_motion_finished',actual_arm_q_rad=self.data.qpos[self.arm_q].tolist(),
                       commanded_arm_q_rad=self.arm_goal.tolist())
        self.observe_contacts('reagent_preparation',primary)

    def require_completed(self):
        required=('release_gate','initial_hold_completed','preshape_reference_completed',
                  'arm_reference_completed','original_bounds_passed','command_ownership_passed',
                  'precontact_passed','right_environment_passed','interarm_contact_passed')
        if self.failure or not all(self.qualification[k] for k in required):
            self.fail('Original early hand/arm program has not completed with valid observations')

    def finish_left(self):
        self.require_completed()
        if self.pending is not None:
            self.fail('LEFT completion left an unobserved overlap command')
        self.timeline.finish(self.tick())
        self.timeline_finished = True
        self.active = False
        self.finished = True
        self.event('primary_preparation_completed')
        self.save()

    def observe_dock(self):
        self.require_completed()
        self.dock_observation_count += 1
        self.observe_contacts('v6_right_grasp_dock','holding_completed_preshape')

    def prepare_remaining(self, profile_path):
        self.require_completed()
        error=float(np.max(np.abs(self.data.qpos[self.arm_q]-self.arm_goal)))
        if error>.03:
            self.fail('RIGHT staging tracking failed the unchanged 0.03rad ready guard')
        profile=json.loads(Path(profile_path).read_text())
        for waypoint in profile['waypoints']:
            if float(waypoint['time_s'])<=.7+1e-8:
                waypoint['arm_q_rad']=self.arm_goal.tolist()
        profile['continuous_approach']['arm_q_rad'][0]=self.arm_goal.tolist()
        profile['overlap_commanded_staging_target_rad']=self.arm_goal.tolist()
        Path(profile_path).write_text(json.dumps(profile,indent=2))
        self.qualification['native_staging_tracking_guard_passed']=True
        self.event('staging_ready_guard_passed',maximum_arm_error_rad=error,threshold_rad=.03)

    def save(self):
        if self.last_tick is not None and not self.timeline_finished:
            self.timeline.finish(self.last_tick+1)
            self.timeline_finished=True
        names=('tick','state_tick','time_s','primary_phase','secondary_phase','arm_command',
               'arm_actual_q','arm_actual_qvel','hand_nominal','hand_command','hand_actual_q',
               'hand_actual_qvel','mcp_opening_offset_rad','ownership_violation')
        arrays={}
        for i,name in enumerate(names):
            dtype=('<i8' if name in ('tick','state_tick','ownership_violation') else
                   '<U64' if name=='primary_phase' else '<U32' if name=='secondary_phase' else '<f8')
            values=np.asarray([row[i] for row in self.rows],dtype=dtype)
            if not self.rows and name.startswith('arm_'):values=values.reshape(0,len(self.arm_q))
            if not self.rows and name.startswith('hand_'):values=values.reshape(0,len(self.hand_q))
            arrays[name]=values
        np.savez_compressed(self.out/'dual_arm_overlap_trace.npz',**arrays)
        report=dict(schema_version=1,source_case_id=self.out.name,output_directory=str(self.out.resolve()),
            model_identity=id(self.model),data_identity=id(self.data),physics_dt_ns=self.dt_ns,
            status='completed' if self.finished and self.failure is None else 'failed',failure=self.failure,
            intervals=self.timeline.intervals if self.timeline_finished else [],events=self.events,
            qualification=self.qualification,contact_events=self.contact_events,
            dock_observation_count=self.dock_observation_count,trace='dual_arm_overlap_trace.npz',
            right_arm_actuators=self.arm_acts.tolist(),right_hand_actuators=self.hand_acts.tolist(),
            scope='Secondary activity only; original full workflow acceptance remains authoritative.',
            original_hand_cadence_hz=50,original_mcp_cadence_hz=1000,physical_state_or_clock_writes=False,
            original_initial_hold_s=.7,original_preshape_s=3.2,original_mcp_ramp_s=1.1,
            command_completion_is_not_exact_hand_convergence=True)
        (self.out/'dual_arm_overlap.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def bind(m,d,runner,path,out):
    global SESSION
    if SESSION is not None:
        raise RuntimeError('Only one isolated overlap episode is permitted per process')
    SESSION=OverlapSession(m,d,runner,path,out)
    return SESSION


def session(m=None,d=None):
    if SESSION is None:
        raise RuntimeError('Overlap controller was not bound by the native engine')
    if m is not None:SESSION.identity(m,d)
    return SESSION


def left_defaults(m,d,robotacts,initial_ctrl):
    session(m,d).left_defaults(robotacts,initial_ctrl)


def release(m,d,dock,tube_body,tube_initial):
    session(m,d).release(dock,tube_body,tube_initial)


def before_step(m,d,phase):
    session(m,d).before_step(phase)


def after_step(m,d,phase):
    session(m,d).after_step(phase)


def finish_left(m,d):
    session(m,d).finish_left()


def restore_hand_entry(m,d):
    state=session(m,d)
    state.require_completed()
    d.ctrl[state.hand_acts]=state.hand_command


def completed_nominal_hand(m,d,opened):
    state=session(m,d)
    state.require_completed()
    if not np.array_equal(opened,state.opened):
        state.fail('Later acquisition changed the completed native preshape endpoint')
    return state.opened.copy()


def completed_opening(m,d):
    session(m,d).require_completed()
    return 1.


def observe_dock(m,d):
    session(m,d).observe_dock()


def save_if_bound():
    if SESSION is not None:SESSION.save()
