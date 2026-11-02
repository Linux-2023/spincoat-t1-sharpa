"""Compile official T1 + two official Sharpa hands as a fixed-base preview.

Mount transforms inherit T1's original tool frames. They are provisional,
not measured adapters. This is not yet the validated pipetting task.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"research/hardware"
T1=BASE/"astribot-t1-aos"
SHARPA=BASE/"sharpa-models"


def expand(path):
    tree=ET.parse(path).getroot()
    def visit(parent):
        for child in list(parent):
            if child.tag=="include":
                subtree=expand((path.parent/child.attrib["file"]).resolve())
                i=list(parent).index(child)
                parent.remove(child)
                for element in subtree:
                    parent.insert(i,element);i+=1
            else:
                if child.tag in ("mesh","texture") and child.get("file"):
                    child.set("file",str((path.parent/child.get("file")).resolve()))
                visit(child)
    visit(tree)
    return tree


def main():
    revisions={str(p.relative_to(BASE)):subprocess.check_output(
        ["git","-C",str(p),"rev-parse","HEAD"],text=True).strip() for p in (T1,SHARPA)}
    assert revisions["astribot-t1-aos"]=="5ca7235afa19180f7ee5d87bfb2dd11392853143"
    assert revisions["sharpa-models"]=="0d19cac602f46456b819e4b6a2c09a74982c9a3e"
    source=T1/"astribot_t1/model/mjcf/robot/t1.xml"
    original=mujoco.MjModel.from_xml_path(str(source))
    tree=expand(source)
    for compiler in tree.iter("compiler"):
        compiler.attrib.pop("meshdir",None);compiler.attrib.pop("texturedir",None)
    removed=[]
    # Remove original grippers and their joint/body references, retain both
    # original tool frames and their relative transforms.
    for parent in tree.iter():
        for child in list(parent):
            if child.tag=="body" and child.get("name","").startswith("astribot_gripper_"):
                removed.extend(e.get("name") for e in child.iter() if e.get("name"))
                parent.remove(child)
    removed=set(removed)
    removed.update(e.get("name") for e in tree.iter()
                   if e.get("name","").startswith("gripper_"))
    changed=True
    while changed:
        changed=False
        for parent in tree.iter():
            for child in list(parent):
                if child.get("name") in removed or any(v in removed for k,v in child.attrib.items() if k!="name"):
                    if child.get("name"):
                        removed.add(child.get("name"))
                    parent.remove(child);changed=True
    s=mujoco.MjSpec.from_string(ET.tostring(tree,encoding="unicode"))
    s.delete(s.joint("chassis_free"))
    s.option.timestep=.002
    hands=[]
    for side in ("left","right"):
        hand_path=SHARPA/f"wave_01/{side}_sharpa_wave/{side}_sharpa_wave.xml"
        hand_tree=ET.parse(hand_path).getroot()
        hand_compiler=hand_tree.find("compiler")
        mesh_dir=hand_path.parent/hand_compiler.attrib.pop("meshdir","")
        for mesh in hand_tree.iter("mesh"):
            if mesh.get("file"):
                mesh.set("file",str((mesh_dir/mesh.get("file")).resolve()))
        hand=mujoco.MjSpec.from_string(ET.tostring(hand_tree,encoding="unicode"))
        hands.append(hand)
        mount=s.body(f"astribot_arm_{side}_tool_link").add_frame(name=f"sharpa_{side}_mount_provisional")
        mount.attach_body(hand.body(f"{side}_hand_C_MC"),prefix=f"sharpa_{side}/")
    m=s.compile();d=mujoco.MjData(m)
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) and m.jnt_limited[j]:
            d.qpos[m.jnt_qposadr[j]]=np.clip(0,*m.jnt_range[j])
    mujoco.mj_forward(m,d)
    assert np.isfinite(d.xpos).all()
    hand_actuators=[m.actuator(a).name for a in range(m.nu) if m.actuator(a).name.startswith("sharpa_")]
    assert len(hand_actuators)==44
    assert not any(m.body(b).name.startswith("astribot_gripper_") for b in range(m.nbody))
    output=BASE/"generated";output.mkdir(exist_ok=True)
    (output/"t1_sharpa_preview.xml").write_text(s.to_xml())
    reloaded=mujoco.MjModel.from_xml_path(str(output/"t1_sharpa_preview.xml"))
    assert reloaded.nq==m.nq and reloaded.nu==m.nu
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,0,.85];camera.distance=2.5;camera.azimuth=135;camera.elevation=-15
    m.vis.global_.offwidth=1000;m.vis.global_.offheight=1000
    with mujoco.Renderer(m,height=1000,width=1000) as renderer:
        renderer.update_scene(d,camera=camera)
        Image.fromarray(renderer.render()).save(BASE/"t1_sharpa_preview.png")
    report={"revisions":revisions,"source_T1_compiled":True,
        "source_T1_nq_nv_nu":[original.nq,original.nv,original.nu],
        "combined_nq_nv_nu":[m.nq,m.nv,m.nu],"sharpa_actuators":hand_actuators,
        "fixed_base_preview":True,"saved_xml_reloaded":True,"mount_calibrated":False,
        "pipette_grasp_validated":False,"spincoat_task_integrated":False,
        "fluid_model_modified":False,"policy_uses_tactile":False,
        "mounts":"original T1 tool frames, identity adapter transform; provisional",
        "generated_xml_sha256":hashlib.sha256((output/"t1_sharpa_preview.xml").read_bytes()).hexdigest()}
    (BASE/"t1_sharpa_build.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
