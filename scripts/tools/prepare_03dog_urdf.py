# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Normalize the SolidWorks 03dog export for the 12-DoF locomotion tasks.

This is idempotent and requires only Python. Mesh filenames, joint frames,
axes, inertias and revolute-joint limits are preserved. The exported left
rear foot's continuous joint is fixed, matching the other three feet.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_URDF = Path(__file__).resolve().parents[2] / "source/robot_lab/data/Robots/03dog/urdf/03dog.urdf"
LEG_PREFIXES = {"left_front": "FL", "right_front": "FR", "left_back": "RL", "right_back": "RR"}


def prepare_urdf(path: Path) -> bool:
    """Validate and normalize an original or already prepared URDF in place."""
    original = path.read_text(encoding="utf-8")
    root = ET.fromstring(original)
    link_names, joint_names = {}, {}
    for old, leg in LEG_PREFIXES.items():
        first_suffix = "" if leg == "FL" else "1"
        for suffix, part in ((first_suffix, "hip"), ("2", "thigh"), ("3", "calf"), ("4", "FOOT")):
            link_names[f"{old}_link{suffix}"] = f"{leg}_{part}"
            joint_names[f"{old}_joint{suffix}"] = f"{leg}_{part}_joint"
    # The right front fixed joint was exported with a link name.
    joint_names["right_front_link4"] = "FR_FOOT_joint"
    for link in root.findall("link"):
        link.set("name", link_names.get(link.get("name"), link.get("name")))
    for joint in root.findall("joint"):
        joint.set("name", joint_names.get(joint.get("name"), joint.get("name")))
        for tag in ("parent", "child"):
            ref = joint.find(tag)
            ref.set("link", link_names.get(ref.get("link"), ref.get("link")))
        if joint.get("name") == "RL_FOOT_joint" and joint.get("type") == "continuous":
            joint.set("type", "fixed")

    expected_links = {"base_link"} | {f"{leg}_{part}" for leg in LEG_PREFIXES.values() for part in ("hip", "thigh", "calf", "FOOT")}
    if {link.get("name") for link in root.findall("link")} != expected_links:
        raise ValueError("Expected base_link and four hip/thigh/calf/FOOT chains")
    joints = root.findall("joint")
    expected_driven = {f"{leg}_{part}_joint" for leg in LEG_PREFIXES.values() for part in ("hip", "thigh", "calf")}
    expected_fixed = {f"{leg}_FOOT_joint" for leg in LEG_PREFIXES.values()}
    if (len(joints) != 16 or {j.get("name") for j in joints if j.get("type") == "revolute"} != expected_driven
            or {j.get("name") for j in joints if j.get("type") == "fixed"} != expected_fixed):
        raise ValueError("Expected 12 revolute joints and four fixed feet")
    for mesh in root.iter("mesh"):
        # Portable paths also work without ROS package discovery in Isaac Sim.
        filename = Path(mesh.get("filename")).name
        if not (path.parent.parent / "meshes" / filename).is_file():
            raise FileNotFoundError(f"Missing 03dog mesh: {filename}")

    # Edit attributes in the source text to preserve the CAD export's formatting.
    updated = original
    for old, new in link_names.items():
        updated = updated.replace(f'link="{old}"', f'link="{new}"')
        updated = re.sub(r'(<link\s+name=")' + re.escape(old) + r'(")', rf'\g<1>{new}\2', updated)
    for old, new in joint_names.items():
        updated = re.sub(r'(<joint\s+name=")' + re.escape(old) + r'(")', rf'\g<1>{new}\2', updated)
    updated = re.sub(r'(<joint\s+name="RL_FOOT_joint"\s+type=")continuous(")', r'\g<1>fixed\2', updated)
    updated = updated.replace("package://03dog/meshes/", "../meshes/")
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", nargs="?", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    changed = prepare_urdf(args.urdf.resolve())
    print(f"{'Prepared' if changed else 'Already prepared'}: {args.urdf}")


if __name__ == "__main__":
    main()
