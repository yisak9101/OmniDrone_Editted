# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Sequence, Union, Optional

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import torch

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform.rotation import Rotation

import omni_drones.utils.kit as kit_utils


def design_scene():
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True
    )
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
    )


def create_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_links: int = 24,
    link_length: float = 0.06,
    rope_damping: float = 10.0,
    rope_stiffness: float = 1.0,
    color=(0.4, 0.2, 0.1),
    enable_collision: bool = False,
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()

    # 1. Capsule (Bar) 생성
    capsuleGeom = UsdGeom.Capsule.Define(stage, f"{prim_path}/Capsule")
    capsuleGeom.CreateHeightAttr(length)
    capsuleGeom.CreateRadiusAttr(0.005)
    capsuleGeom.CreateAxisAttr("Z")
    capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(*translation))
    capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
    capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    capsuleGeom.CreateDisplayColorAttr().Set([color])

    # Physics 적용
    UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
    massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
    massAPI.CreateMassAttr().Set(mass)

    UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
    prim: Usd.Prim = capsuleGeom.GetPrim()
    prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

    # 2. 하단 연결 (Payload <-> Bar)
    if from_prim is not None:
        # Sphere 위치: 막대 중심에서 길이 절반만큼 아래로
        sphere_loc_z = translation[2] - (length / 2.0)
        
        sphere = prim_utils.create_prim(
            f"{prim_path}/Sphere",
            "Sphere",
            translation=(0, 0, sphere_loc_z), 
            attributes={"radius": 0.02},
        )
        UsdPhysics.RigidBodyAPI.Apply(sphere)
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.GetAttribute("physics:collisionEnabled").Set(False)
        
        massAPIsphere = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
        massAPIsphere.CreateMassAttr().Set(0.0001)

        # [Fixed] Bar <-> Sphere
        fixed_joint = script_utils.createJoint(stage, "Fixed", prim, sphere)
        # Bar의 맨 아래쪽 (로컬 좌표)
        fixed_joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(0, 0, -length / 2.0))
        fixed_joint.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0, 0, 0))

        # [D6] Payload <-> Sphere
        d6_joint = script_utils.createJoint(stage, "D6", from_prim, sphere)
        # Payload의 모서리 오프셋 (여기가 핵심이었습니다)
        d6_joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*from_offset))
        d6_joint.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0, 0, 0))

        # Joint Limits
        d6_joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        d6_joint.GetAttribute("limit:rotX:physics:high").Set(120)
        d6_joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        d6_joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(d6_joint, "rotX")
        UsdPhysics.DriveAPI.Apply(d6_joint, "rotY")
        d6_joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        d6_joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    # 3. 상단 연결 (Bar <-> Drone)
    if to_prim is not None:
        joint_prim = script_utils.createJoint(stage, "D6", prim, to_prim)
        
        # [핵심 수정 사항]
        # Bar의 맨 위쪽 (로컬 좌표): translation을 더하지 않고 length/2 만 사용
        joint_prim.GetAttribute("physics:localPos0").Set(Gf.Vec3f(0, 0, length / 2.0))
        # Drone의 중심
        joint_prim.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0, 0, 0))

        joint_prim.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint_prim.GetAttribute("limit:rotX:physics:high").Set(120)
        joint_prim.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint_prim.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint_prim, "rotX")
        UsdPhysics.DriveAPI.Apply(joint_prim, "rotY")
        joint_prim.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint_prim.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    return prim


def create_bar(
    prim_path: str,
    length: float,
    translation=(0, 0, 0),
    from_prim: str = None,
    to_prim: str = None,
    mass: float = 0.02,
    enable_collision=False,
    color=(0.4, 0.4, 0.2),
    from_offset=None
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()

    capsuleGeom = UsdGeom.Capsule.Define(stage, f"{prim_path}/Capsule")
    capsuleGeom.CreateHeightAttr(length)
    capsuleGeom.CreateRadiusAttr(0.005)
    capsuleGeom.CreateAxisAttr("Z")
    capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(*translation))
    capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
    capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    capsuleGeom.CreateDisplayColorAttr().Set([color])

    UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
    massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
    massAPI.CreateMassAttr().Set(mass)

    UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
    prim: Usd.Prim = capsuleGeom.GetPrim()
    prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

    if from_prim is not None:
        sphere = prim_utils.create_prim(
            f"{prim_path}/Sphere",
            "Sphere",
            translation=(0, 0, -length),
            attributes={"radius": 0.02},
        )
        UsdPhysics.RigidBodyAPI.Apply(sphere)
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.GetAttribute("physics:collisionEnabled").Set(False)

        massAPIsphere = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
        massAPIsphere.CreateMassAttr().Set(0.0001)

        script_utils.createJoint(stage, "Fixed", from_prim, sphere)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, sphere)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        UsdPhysics.DriveAPI.Apply(joint, "rotZ")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.002)
        joint.GetAttribute("drive:rotZ:physics:damping").Set(0.002) # 값을 높게 설정

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, to_prim)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        UsdPhysics.DriveAPI.Apply(joint, "rotZ")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.002)
        joint.GetAttribute("drive:rotZ:physics:damping").Set(0.002)

    return prim


