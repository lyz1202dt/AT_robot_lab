"""Restore independently reported FOOT links while preserving aggregate leg inertia."""
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone


def point_inertia(offset):
    return np.dot(offset,offset)*np.eye(3)-np.outer(offset,offset)


def split_inertia(mass,center,inertia,foot_mass,foot_center,foot_inertia):
    calf_mass = mass-foot_mass
    calf_center = (mass*center-foot_mass*foot_center)/calf_mass
    calf_inertia = inertia-foot_inertia-foot_mass*point_inertia(foot_center-center)-calf_mass*point_inertia(calf_center-center)
    return calf_mass,calf_center,calf_inertia


@clone
def spawn_cyberstar(prim_path,cfg,translation=None,orientation=None,**kwargs):
    prim = sim_utils.spawn_from_usd(prim_path,cfg,translation,orientation,**kwargs)
    stage = prim.GetStage()
    for leg in ("FL","FR","RL","RR"):
        calf = stage.GetPrimAtPath(f"{prim_path}/base_link/{leg}_calf")
        if not calf.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Expected CSDog1 calf rigid body at {calf.GetPath()}")
        foot_path = f"{prim_path}/base_link/{leg}_FOOT"
        foot = UsdGeom.Xform.Define(stage,foot_path).GetPrim()
        sign = 1 if leg in ("FL","RL") else -1
        origin = np.array([.13949,-sign*.004,-.14978])
        sphere_center = np.array([.141411,-sign*.008,-.162839])
        relative = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*origin))
        calf_matrix = UsdGeom.Xformable(calf).GetLocalTransformation()
        UsdGeom.Xformable(foot).AddTransformOp().Set(relative*calf_matrix)
        UsdPhysics.RigidBodyAPI.Apply(foot)
        PhysxSchema.PhysxRigidBodyAPI.Apply(foot)
        PhysxSchema.PhysxContactReportAPI.Apply(foot).CreateThresholdAttr(0.)
        # A small fixed-link mass is transferred from calf, not added to the robot.
        foot_mass, radius = .001,.0225
        foot_tensor = np.eye(3)*(.4*foot_mass*radius**2)
        api = UsdPhysics.MassAPI(calf)
        mass,center = api.GetMassAttr().Get(),np.asarray(api.GetCenterOfMassAttr().Get(),dtype=float)
        rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(api.GetPrincipalAxesAttr().Get())),dtype=float).T
        tensor = rotation@np.diag(api.GetDiagonalInertiaAttr().Get())@rotation.T
        mass,center,tensor = split_inertia(mass,center,tensor,foot_mass,sphere_center,foot_tensor)
        eigenvalues,axes = np.linalg.eigh(tensor)
        if np.any(eigenvalues<=0):
            raise RuntimeError(f"Nonpositive calf inertia after FOOT split: {eigenvalues}")
        if np.linalg.det(axes)<0:
            axes[:,0] *= -1
        principal = Gf.Matrix3d(*axes.T.flatten()).ExtractRotation().GetQuat()
        api.GetMassAttr().Set(float(mass))
        api.GetCenterOfMassAttr().Set(Gf.Vec3f(*center))
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*eigenvalues))
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(principal))
        foot_api = UsdPhysics.MassAPI.Apply(foot)
        foot_api.CreateMassAttr(foot_mass)
        foot_api.CreateCenterOfMassAttr(Gf.Vec3f(*(sphere_center-origin)))
        foot_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.diag(foot_tensor)))
        foot_api.CreatePrincipalAxesAttr(Gf.Quatf(1))
        # Keep exactly the existing foot sphere geometry; only its rigid-body owner changes.
        old = f"{calf.GetPath()}/collisions/{leg}_FOOT_collision/{leg}_FOOT_collision"
        parent = stage.GetPrimAtPath(old).GetParent()
        while parent and parent.GetPath() != calf.GetPath():
            if parent.IsInstance():
                parent.SetInstanceable(False)
            parent = parent.GetParent()
        sphere = UsdGeom.Sphere.Define(stage,f"{foot_path}/collision")
        sphere.CreateRadiusAttr(radius)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*(sphere_center-origin)))
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdGeom.Imageable(sphere).CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(old)).GetCollisionEnabledAttr().Set(False)
        joint = UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/joints/{leg}_FOOT_fixed")
        joint.CreateBody0Rel().SetTargets([calf.GetPath()])
        joint.CreateBody1Rel().SetTargets([foot.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateCollisionEnabledAttr(False)
        # The feet are created after spawn_from_usd applies these properties.
        sim_utils.modify_rigid_body_properties(foot_path,cfg.rigid_props,stage=stage)
        sim_utils.modify_collision_properties(str(sphere.GetPath()),cfg.collision_props,stage=stage)
    return prim
