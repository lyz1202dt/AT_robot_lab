# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Prepare the converted CSDog1 USD for Isaac Lab training.

The MJCF importer creates a valid visual/physics USD, but for this asset it
also leaves multiple articulation-root candidates and does not author contact
reporter APIs on the rigid bodies. This script fixes those USD schemas once so
training can load the USD directly.
"""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Prepare converted CSDog1 USD for Isaac Lab.")
parser.add_argument(
    "usd_path",
    nargs="?",
    default="source/robot_lab/data/Robots/csdog1/usd/csdog1.usd",
    help="Path to the converted CSDog1 USD file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Tf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402


def _find_root_body(stage: Usd.Stage) -> Usd.Prim:
    """Find the real CSDog1 root rigid body prim."""
    candidates = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = prim.GetPath().pathString
            if path.endswith("/base_link/base_link") or path.endswith("/base_link"):
                candidates.append(prim)

    for prim in candidates:
        if prim.GetPath().pathString.endswith("/base_link/base_link"):
            return prim
    if candidates:
        return candidates[0]

    raise RuntimeError("Could not find a CSDog1 base_link rigid body in the USD.")


def _standardize_root_xform(root_body: Usd.Prim):
    """Standardize only the root body used by Isaac Lab sensors.

    Some MJCF wrapper prims already contain importer-authored ops with types
    that collide with Isaac Lab's helper. We leave those wrappers untouched and
    only prepare the actual rigid body that RayCaster tracks.
    """
    if not root_body.IsA(UsdGeom.Xformable):
        return False
    try:
        return sim_utils.standardize_xform_ops(root_body)
    except Tf.ErrorException as exc:
        print(f"[WARN] Could not standardize root xform ops for {root_body.GetPath()}: {exc}")
        return False


def _keep_single_articulation_root(stage: Usd.Stage, root_body: Usd.Prim):
    root_path = root_body.GetPath().pathString
    removed = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.GetPath().pathString != root_path:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            removed.append(prim.GetPath().pathString)

    if not root_body.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(root_body)
    if not root_body.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(root_body)

    return removed


def _enable_contact_reporters(stage: Usd.Stage):
    enabled = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        else:
            cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPath())
        cr_api.CreateThresholdAttr().Set(0.0)
        enabled.append(prim.GetPath().pathString)
    return enabled


def _list_sensor_visible_rigid_bodies(stage: Usd.Stage):
    """Return rigid bodies that a standard ``/Robot/.*`` sensor can see.

    Isaac Lab's ContactSensor only scans the leaf prims directly under the
    configured parent path. For a robot spawned at ``/Robot``, the default
    locomotion sensor therefore sees ``/Robot/<body>`` but not nested paths
    such as ``/Robot/<body>/<body>``.
    """
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return []

    visible = []
    for prim in default_prim.GetChildren():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            visible.append(prim.GetPath().pathString)
    return visible


def main():
    usd_path = args_cli.usd_path
    if not os.path.isabs(usd_path):
        usd_path = os.path.abspath(usd_path)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    root_body = _find_root_body(stage)
    standardized_root_xform = _standardize_root_xform(root_body)
    removed_roots = _keep_single_articulation_root(stage, root_body)
    contact_bodies = _enable_contact_reporters(stage)
    sensor_visible_bodies = _list_sensor_visible_rigid_bodies(stage)

    stage.GetRootLayer().Save()

    print(f"Prepared USD: {usd_path}", flush=True)
    print(f"Articulation root: {root_body.GetPath().pathString}", flush=True)
    print(f"Standardized root xform ops: {standardized_root_xform}", flush=True)
    print(f"Removed extra articulation roots: {removed_roots}", flush=True)
    print(f"Enabled contact reporters on {len(contact_bodies)} rigid bodies.", flush=True)
    print("Rigid bodies with contact reporters:", flush=True)
    for body_path in contact_bodies:
        print(f"  - {body_path}", flush=True)
    print("Rigid bodies visible to ContactSensor('/Robot/.*'):", flush=True)
    for body_path in sensor_visible_bodies:
        print(f"  - {body_path}", flush=True)
    if not sensor_visible_bodies:
        print(
            "[WARN] No direct child rigid bodies found under the USD default prim. "
            "The default locomotion ContactSensor path '/Robot/.*' will not match "
            "this USD; use a flattened USD or disable/replace contact rewards.",
            flush=True,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
