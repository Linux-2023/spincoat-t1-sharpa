"""Standalone CPU MuJoCo integration rig; frozen liquid source stays unchanged."""
from pathlib import Path
import importlib,sys,types,os
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path('/home/agilex/DexBioLab')
BENCH_ORIGIN=np.array([.32,0.,0.])  # front edge; bench extends away from robot
BENCH_HEIGHT=.8
PIPETTE_START=np.array([.40,-.48,BENCH_HEIGHT+.082])
# Turn the chassis away from the robot; keep the bowl within right-arm reach.
STATION=np.array([.505,-.34,BENCH_HEIGHT+.0035])
STATION_QUAT=np.array([2**-.5,0.,0.,2**-.5])
HAND_QUAT=np.array([2**-.5,2**-.5,0.,0.])
GRASP_WRIST_OFFSET=np.array([-.025,.09,.11])


def scene():
    s=mujoco.MjSpec.from_file(str(ROOT/'research/hardware/generated/t1_sharpa_preview.xml'))
    s.option.timestep=.001
    s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.iterations=100
    s.option.ccd_iterations=1500
    s.option.viscosity=0
    s.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[3,3,.1],rgba=[.17,.19,.22,1])
    bench=mujoco.MjSpec.from_file(str(ROOT/'assets/third_party/lab_bench/lab_bench.xml'))
    s.worldbody.add_frame(pos=BENCH_ORIGIN).attach_body(bench.body('lab_bench'),prefix='')
    s.worldbody.add_light(pos=[1,-1,3],dir=[-.3,.3,-1],diffuse=[.8,.8,.8])
    children=[bench]
    for rel,body,position,prefix in [
        ('assets/mjcf/object/pipette.gen.xml','pipette',PIPETTE_START,'pipette/'),
        ('assets/mjcf/misc/pipette_stand.xml','pipette_stand',PIPETTE_START+[-.06,0,.15],'stand/'),
        ('assets/mjcf/object/spincoater_prototype.xml','spincoater',STATION,'spincoater/')]:
        source=ORIGINAL/rel
        child=mujoco.MjSpec.from_file(str(source));children.append(child)
        b=child.body(body);b.pos[:]=0;b.quat[:]=[1,0,0,0]
        if prefix=='pipette/':
            b.add_freejoint(name='free')
            for joint in child.joints:
                if joint.name in ('pipette_button','pipette_ejector'):
                    joint.solref_limit[:]=[.002,1]
                    joint.solimp_limit[:]=[.99,.99,.001,.5,2]
            for g in child.geoms:
                if g.contype or g.conaffinity:
                    g.friction[:]=[1.2,.01,.001];g.condim=4
        quat=STATION_QUAT if prefix=='spincoater/' else [1,0,0,0]
        s.worldbody.add_frame(pos=position,quat=quat).attach_body(b,prefix=prefix)
    for a in s.actuators:
        if a.target.startswith('astribot_arm_'):
            a.gainprm[0]=2500;a.biasprm[:3]=[0,-2500,-100]
        elif a.name.startswith('sharpa_'):
            # Correct the upstream positive velocity coefficient for this rig:
            # use dissipative PD, retain vendor force ranges and kinematics.
            kp=20. if ('MCP' in a.name or 'CMC' in a.name) else 10.
            a.gainprm[0]=kp;a.biasprm[:3]=[0,-kp,-.6]
            a.forcelimited=True
            cap=1.5 if ('MCP' in a.name or 'CMC' in a.name) else .5
            a.forcerange[:]=[-cap,cap]
    if os.environ.get('T1_LAB_CONTEXT')=='1':
        from spincoat_lab_context import add_spincoat_lab_context
        add_spincoat_lab_context(s)
    if os.environ.get('T1_CHEMIST_COAT')=='1':
        from t1_chemist_coat import add_chemist_coat
        add_chemist_coat(s)
    m=s.compile();d=mujoco.MjData(m)
    m.vis.global_.offwidth=1600;m.vis.global_.offheight=1000
    m.vis.headlight.ambient[:]=[.4,.4,.4]
    m.vis.headlight.diffuse[:]=[.6,.6,.6]
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            d.qpos[m.jnt_qposadr[j]]=np.clip(0,*m.jnt_range[j])
    mujoco.mj_forward(m,d)
    return s,m,d,children


def fluid_module(name):
    package=types.ModuleType('frozen_spincoat');package.__path__=[str(ROOT/'src/tasks/config/shadow_spincoat_dispense')]
    sys.modules[package.__name__]=package
    return importlib.import_module(package.__name__+'.'+name)


def arm_indices(m,side='right'):
    ids=np.array([m.joint(f'astribot_arm_{side}_joint_{i}').id for i in range(1,8)])
    return ids,m.jnt_qposadr[ids],m.jnt_dofadr[ids]


def hand_indices(m,side='right'):
    ids=np.array([j for j in range(m.njnt) if m.joint(j).name.startswith(f'sharpa_{side}/')])
    return ids,m.jnt_qposadr[ids],m.jnt_dofadr[ids]


def ik(m,d,position,quat,seed=None,iterations=150):
    ids,qids,dofs=arm_indices(m)
    work=mujoco.MjData(m);work.qpos[:]=d.qpos
    if seed is not None:work.qpos[qids]=seed
    body=m.body('sharpa_right/right_hand_C_MC').id
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    for _ in range(iterations):
        mujoco.mj_forward(m,work)
        ep=np.asarray(position)-work.xpos[body]
        eq=np.zeros(3);mujoco.mju_subQuat(eq,np.asarray(quat),work.xquat[body])
        # mju_subQuat is local to the current body orientation.
        eq=work.xmat[body].reshape(3,3)@eq
        if np.linalg.norm(ep)<1e-6 and np.linalg.norm(eq)<1e-5:break
        mujoco.mj_jacBody(m,work,jp,jr,body)
        J=np.vstack([jp[:,dofs],.3*jr[:,dofs]])
        error=np.r_[ep,.3*eq]
        dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,error)
        work.qpos[qids]=np.clip(work.qpos[qids]+np.clip(dq,-.08,.08),m.jnt_range[ids,0]+1e-5,m.jnt_range[ids,1]-1e-5)
    mujoco.mj_forward(m,work)
    return work.qpos[qids].copy(),float(np.linalg.norm(work.xpos[body]-position))
