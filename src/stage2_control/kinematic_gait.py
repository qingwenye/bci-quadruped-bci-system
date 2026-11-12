# -*- coding: utf-8 -*-
"""
kinematic_gait.py
================
阶段 2：CoppeliaSim 六足机器人运动学控制（6 腿 × 3 关节 = 18 关节）。

设计目标
--------
1. 纯运动学控制：只调用 sim.setJointPosition()，不调用
   sim.setJointTargetPosition()，避免调动力学参数。
2. 步进模式：sim.setStepping(True)，仿真时间只在 Python 调用
   controller.step() 时前进，为阶段 3 的脑电实时闭环打基础。
3. 通用关节发现：优先按关节名称自动识别 6 条腿 × 3 个关节；
   识别不了时，可手工填写 JOINT_PATHS（脚本运行时会打印真实路径）。
4. 六足经典三角步态（Tripod Gait）：
      Group A = FL, MR, RL，相位 0
      Group B = FR, ML, RR，相位 π
   两组交替形成两个稳定的三角形支撑。Ant 的 joint1 yaw 做正弦
   前后扫动，joint2 pitch / joint3 knee 在 cos 正半周抬腿，
   向后摆时落地。左右两侧步幅差速实现转向。
   注意：运动学模式只改关节角，不主动平移机器人基座，所以视觉上是
   “原地迈步”；如果后续要让机身真的前进，可以再对 base 调用
   sim.setObjectPosition() 做开环位移。

运行前准备
----------
1. 先启动 CoppeliaSim，加载包含六足机器人的场景（18 个 revolute joints），
   例如 scenes/my_robot_debugged.ttt。
2. 保持仿真处于停止状态（不要点界面上的 Start）；本脚本会自己调用
   sim.startSimulation()。
3. 控制器会自动把 /Ant 下 shape 设为 Static，并解除 targetLeg1..6
    的 IK target 绑定；否则关节角在变、link 却完全不动。
4. 如果模型自带步态 simulation script，请先禁用/删除，否则它会和本脚本
   争夺关节控制权。
5. 先运行一次查看关节路径：
       python src/stage2_control/kinematic_gait.py --inspect
   --inspect 是严格只读模式：只连接、读对象、打印路径，不会
   startSimulation/stopSimulation/step/setJointPosition/setJointMode，
   也不会让机器人动。它支持任意数量关节。
   如果自动识别不对，把打印出的路径填入下面的 JOINT_PATHS。

腿与关节顺序约定
----------------
6 条腿：
    FL = Front-Left   （前左）
    ML = Middle-Left  （中左）
    RL = Rear-Left    （后左）
    FR = Front-Right  （前右）
    MR = Middle-Right （中右）
    RR = Rear-Right   （后右）

每条腿 3 个关节，按 Ant 模型的实际顺序解释：
    0 = joint1Leg / yaw / 前后摆（主推进）
    1 = joint2Leg / pitch / 抬腿
    2 = joint3Leg / knee / 小腿弯曲
如果你的模型顺序不同（例如 [髋, 膝, 踝]），需要调整 STAND_POSE
和 _apply_gait() 中 joint 1/2/3 的用法。脚本会自动打印识别结果，
第一次运行请务必核对。

作者备注
--------
不同六足模型的关节零位、正方向、减速比都不一样，所以 STAND_POSE
和步态幅值只是“通用起点”。如果六足站立时趴下/劈叉，优先只改
STAND_POSE 里的三个数，再调 YAW_SWING / LIFT_AMOUNT / KNEE_BEND。
最稳妥的做法是保持 CAPTURE_STAND_POSE=True，让脚本捕获模型当前
站立姿态；确认后再把捕获值复制到 STAND_POSE 固定下来。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List

import zmq

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 coppeliasim_zmqremoteapi_client。请先安装：\n"
        "  pip install coppeliasim-zmqremoteapi-client"
    ) from exc


# ===========================================================================
# 1. 可调参数
# ===========================================================================
LEGS = ("FL", "ML", "RL", "FR", "MR", "RR")

# Ant 模型的实际关节角色（按每条腿列表中的顺序）：
#   joint1Leg -> 竖直轴 yaw，前后摆（protraction/retraction）
#   joint2Leg -> 水平轴 pitch，抬腿/落腿
#   joint3Leg -> knee，小腿弯曲
JOINT_ROLE = ("joint1_yaw", "joint2_pitch", "joint3_knee")

# 运动学动画模式：
#   - 所有 shape 设为 Static（防止重力下坠）；
#   - 但 Ant 的 targetLeg1..6 是 IK target dummy（linktype=5），会把腿绑住，
#     导致关节角变化但 link 不动。必须把它们的 dummytype/linktype 解除，
#     static link 才会在仿真运行中跟随 setJointPosition。
# shutdown() 时恢复原 shape/dummy 属性。
MODEL_ROOT = "/Ant"
RELEASE_IK_TARGETS = True

# 左右腿分组（用于转向差速和 roll 倾斜）
LEFT_LEGS = ("FL", "ML", "RL")
RIGHT_LEGS = ("FR", "MR", "RR")

# 三角步态的两组支撑腿：
#   Group A: FL + MR + RL，相位 0
#   Group B: FR + ML + RR，相位 π（相差半个周期）
# 这样任意时刻总有一组三角形着地，另一组抬起前摆，稳定性最高。
TRIPOD_A = ("FL", "MR", "RL")
TRIPOD_B = ("FR", "ML", "RR")

# CoppeliaSim 自带 Ant 模型的真实关节路径。
#
# 腿编号与方位（根据世界坐标和头部位置推断）：
#   [0] = FL 前左    [1] = ML 中左    [2] = RL 后左
#   [5] = FR 前右    [4] = MR 中右    [3] = RR 后右
#
# 每条腿的三个关节：
#   joint1Leg = 竖直轴 yaw（前后摆）
#   joint2Leg = 水平轴 pitch（抬腿/落腿）
#   joint3Leg = knee（小腿弯曲）
#
# 注意：方括号 [i] 必须原样保留在字符串里，sim.getObject() 会把它当作
# 同名对象的重名索引。
JOINT_PATHS = {
    "FL": [
        "/Ant/joint1Leg[0]",
        "/Ant/joint1Leg[0]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[0]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
    "ML": [
        "/Ant/joint1Leg[1]",
        "/Ant/joint1Leg[1]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[1]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
    "RL": [
        "/Ant/joint1Leg[2]",
        "/Ant/joint1Leg[2]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[2]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
    "FR": [
        "/Ant/joint1Leg[5]",
        "/Ant/joint1Leg[5]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[5]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
    "MR": [
        "/Ant/joint1Leg[4]",
        "/Ant/joint1Leg[4]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[4]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
    "RR": [
        "/Ant/joint1Leg[3]",
        "/Ant/joint1Leg[3]/link1RespondableLeg/joint2Leg",
        "/Ant/joint1Leg[3]/link1RespondableLeg/joint2Leg/link2RespondableLeg/joint3Leg",
    ],
}

# --inspect 的只读检查列表。这里生成 18 个占位符：
#   /Hexapod/leg_FL/joint_1 ... /Hexapod/leg_RR/joint_3
# 运行 --inspect 时，预期它们大多会显示 MISS；紧接着脚本会列出
# 场景里所有真实关节的 handle / alias / 完整 path，你从终端复制即可。
INSPECT_JOINT_PATHS = [
    f"/Hexapod/leg_{leg}/joint_{idx}"
    for leg in LEGS
    for idx in (1, 2, 3)
]

# Ant 站立姿态（弧度），顺序 [joint1_yaw, joint2_pitch, joint3_knee]。
#
# 下面这组值是在 Ant 上实测并抬高本体 3 cm 后的站立姿态：
#   [0] FL = (-0.5236, +0.0058, +0.4014)
#   [1] ML = ( 0.0000, +0.0058, +0.4014)
#   [2] RL = (+0.5236, +0.0058, +0.4014)
#   [5] FR = (+0.5236, -0.0058, -0.4014)
#   [4] MR = ( 0.0000, -0.0058, -0.4014)
#   [3] RR = (-0.5236, -0.0058, -0.4014)
#
# 配套场景：scenes/my_robot_debugged.ttt，本体 z = 0.0768，六足端 z ≈ 0。
# 注意左右两侧的 joint2/joint3 符号是镜像的，不要把它们统一成同一个符号。
# 如果模型初始姿态不同，保持 CAPTURE_STAND_POSE=True 让脚本重新捕获。
STAND_POSE = {
    "FL": (-0.5236, +0.0058, +0.4014),
    "ML": (0.0000, +0.0058, +0.4014),
    "RL": (+0.5236, +0.0058, +0.4014),
    "FR": (+0.5236, -0.0058, -0.4014),
    "MR": (0.0000, -0.0058, -0.4014),
    "RR": (-0.5236, -0.0058, -0.4014),
}

# 默认启动时把场景中模型的“当前关节位置”作为站立姿态（模型加载时通常就是
# 设计好的站立姿态），这比猜 18 个角度可靠得多。
# 如果你想强制使用上面 STAND_POSE 的固定角度，改成 False。
# 运行 --stand-only 会打印捕获到的角度，可复制回 STAND_POSE 固定下来。
CAPTURE_STAND_POSE = True

# 三角步态相位：Group A = 0，Group B = π。
LEG_PHASE = {
    **{leg: 0.0 for leg in TRIPOD_A},
    **{leg: math.pi for leg in TRIPOD_B},
}

# 六足 Yaw 三角步态幅值。六条腿同时动，幅值比四足略小，视觉上更稳。
GAIT_FREQ = 1.2          # 步态频率 Hz
YAW_SWING = 0.28         # joint1 yaw 前后摆幅（rad）；整体走反了就取负
LIFT_AMOUNT = 0.15       # joint2 pitch 抬腿量（rad）；抬不起来就增大
KNEE_BEND = 0.10         # joint3 knee 配合弯曲量（rad）；拖地就增大
LIFT_PHASE_OFFSET = 0.0  # 抬腿相位偏移（rad）；拖地可试 ±π/2
TURN_STRIDE_GAIN = 0.35  # 转向时左右腿 yaw 摆幅差（0.35 = 35%）
STOP_DURATION = 0.5      # 停止时平滑回站立的时长（s）

# 仿真时间步长兜底（通常从 sim.getSimulationTimeStep() 读取）
DEFAULT_DT = 0.05


# ===========================================================================
# 2. 关节名称解析
# ===========================================================================
LEG_TOKEN_MAP = {
    "FL": {"fl", "lf"},
    "ML": {"ml", "lm"},
    "RL": {"rl", "lr", "hl", "lh"},
    "FR": {"fr", "rf"},
    "MR": {"mr", "rm"},
    "RR": {"rr", "hr", "rh"},
}

LEG_REGEX_MAP = {
    "fl": "FL", "lf": "FL",
    "ml": "ML", "lm": "ML",
    "rl": "RL", "lr": "RL", "hl": "RL", "lh": "RL",
    "fr": "FR", "rf": "FR",
    "mr": "MR", "rm": "MR",
    "rr": "RR", "hr": "RR", "rh": "RR",
}

JOINT_INDEX_TOKENS = {
    "hip": 0,
    "abduction": 0,
    "roll": 0,
    "shoulder": 0,
    "thigh": 1,
    "upper": 1,
    "femur": 1,
    "pitch": 1,
    "knee": 2,
    "calf": 2,
    "shin": 2,
    "lower": 2,
}


def _normalize_name(text: str) -> str:
    """把 'joint-FL-1' / 'joint FL 1' 统一成 'joint_fl_1'。"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _parse_joint_name(text: str):
    """
    从关节 alias/path 中解析 (leg, joint_index)。

    Returns
    -------
    leg : str | None
        'FL' / 'ML' / 'RL' / 'FR' / 'MR' / 'RR'，解析失败为 None。
    idx : int | None
        0/1/2，解析失败为 None。
    """
    s = _normalize_name(text)
    tokens = [t for t in s.split("_") if t]
    token_set = set(tokens)

    leg = None
    middle_tokens = {"middle", "mid", "center", "central"}

    # 2.1 先处理 front_left / middle_right / rear_right 这类拆开的词
    if ("front" in token_set and "left" in token_set) or "frontleft" in token_set:
        leg = "FL"
    elif (bool(token_set & middle_tokens) and "left" in token_set) \
            or "middleleft" in token_set or "midleft" in token_set:
        leg = "ML"
    elif (("rear" in token_set or "back" in token_set) and "left" in token_set) \
            or "rearleft" in token_set or "backleft" in token_set:
        leg = "RL"
    elif ("front" in token_set and "right" in token_set) or "frontright" in token_set:
        leg = "FR"
    elif (bool(token_set & middle_tokens) and "right" in token_set) \
            or "middleright" in token_set or "midright" in token_set:
        leg = "MR"
    elif (("rear" in token_set or "back" in token_set) and "right" in token_set) \
            or "rearright" in token_set or "backright" in token_set:
        leg = "RR"

    # 2.2 再处理独立的 fl/ml/rl/fr/mr/rr token
    if leg is None:
        for key, variants in LEG_TOKEN_MAP.items():
            if token_set & variants:
                leg = key
                break

    # 2.3 最后处理 jointFL1 / FLJ1 / MLJ2 这种连写形式
    if leg is None:
        m = re.search(
            r"(fl|lf|ml|lm|rl|lr|hl|lh|fr|rf|mr|rm|rr|hr|rh)[a-z]*?([123])",
            s,
        )
        if m:
            leg = LEG_REGEX_MAP[m.group(1)]
            idx = int(m.group(2)) - 1
            return leg, idx

    # 2.4 解析关节序号
    idx = None
    m = re.search(r"(?:^|_)([123])(?:$|_)", s)
    if m:
        idx = int(m.group(1)) - 1
    else:
        for token, value in JOINT_INDEX_TOKENS.items():
            if token in token_set:
                idx = value
                break

    return leg, idx


def _natural_key(text: str):
    """让 joint_2 排在 joint_10 前面（自然排序）。"""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


# ===========================================================================
# 3. 控制器
# ===========================================================================
class RobotController:
    """CoppeliaSim 六足机器人运动学控制器（18 关节，三角步态）。"""

    def __init__(self, host: str = "localhost", port: int = 23000,
                 joint_paths=None, stand_pose=None):
        # 不显式传 joint_paths 时，默认使用脚本顶部填好的 JOINT_PATHS。
        # 这样直接 RobotController() 也不会去场景里错误地抓 23 个 joint。
        if joint_paths is None:
            joint_paths = JOINT_PATHS
        print("=" * 72)
        print("CoppeliaSim hexapod kinematic controller (tripod gait)")
        print("=" * 72)

        # ---- 3.1 连接 ZMQ Remote API ----
        try:
            self.client = RemoteAPIClient(host=host, port=port)
            self.sim = self.client.require("sim")
        except Exception as exc:
            raise RuntimeError(
                f"无法连接 CoppeliaSim ZMQ Remote API ({host}:{port})。\n"
                "请确认：\n"
                "  1) CoppeliaSim 已经启动；\n"
                "  2) 场景已经加载；\n"
                "  3) ZMQ Remote API 服务端口是 23000（或改 --port）。"
            ) from exc

        print(f"[init] connected to {host}:{port}")

        # ---- 3.2 清掉可能正在运行的旧仿真，开启步进模式 ----
        self._stop_existing_simulation()
        self._reset_stepping()
        self.sim.setStepping(True)
        print("[init] stepping mode enabled: simulation only advances on step()")

        # ---- 3.3 读取仿真步长 ----
        try:
            dt = float(self.sim.getSimulationTimeStep())
            self.dt = dt if dt > 1e-6 else DEFAULT_DT
        except Exception:
            self.dt = DEFAULT_DT
        print(f"[init] simulation time step = {self.dt:.4f} s")

        # ---- 3.4 获取 18 个关节句柄（6 腿 × 3） ----
        self.joints = self._resolve_joints(joint_paths)

        # ---- 3.5 站立姿态 ----
        if stand_pose is not None:
            self.stand_pose = {leg: tuple(float(v) for v in stand_pose[leg])
                               for leg in LEGS}
            print("[init] using explicit stand_pose.")
        elif CAPTURE_STAND_POSE:
            # 读取模型当前姿态（仿真停止时读到的是场景保存的姿态）
            try:
                self.stand_pose = {
                    leg: tuple(float(self.sim.getJointPosition(h))
                               for h in self.joints[leg])
                    for leg in LEGS
                }
                print("[init] captured current joint positions as stand pose:")
                for leg in LEGS:
                    q = ", ".join(f"{v:+.4f}" for v in self.stand_pose[leg])
                    print(f"       {leg}: ({q})")
            except Exception as exc:
                print(f"[init][warn] 无法读取当前姿态 ({exc})，改用 STAND_POSE。")
                self.stand_pose = {leg: tuple(float(v) for v in STAND_POSE[leg])
                                   for leg in LEGS}
        else:
            self.stand_pose = {leg: tuple(float(v) for v in STAND_POSE[leg])
                               for leg in LEGS}
            print("[init] using STAND_POSE.")

        for leg in LEGS:
            if len(self.stand_pose[leg]) != 3:
                raise ValueError(f"stand_pose['{leg}'] 必须是 3 个数")

        # 关节模式最后切换，避免个别版本切换模式时复位关节位置
        self._force_kinematic_mode()

        # 运动学动画模式：static shapes + 解除 targetLeg IK target 绑定，
        # 既防止重力下坠，又允许 setJointPosition 真正带动腿。
        self._prepare_kinematic_model()

        self.stop_duration = STOP_DURATION
        self._gait_time = 0.0       # 内部步态时钟（秒）
        self._sim_started = False

        print("[init] ready. Joint mapping:")
        for leg in LEGS:
            handles = ", ".join(str(h) for h in self.joints[leg])
            print(f"       {leg}: [{handles}]")
        print(f"[init] joint role order: {JOINT_ROLE}")

    # ------------------------------------------------------------------
    # 3.6 连接/仿真状态辅助
    # ------------------------------------------------------------------
    def _get_const(self, *names):
        """兼容不同 CoppeliaSim 版本的常量名。"""
        for name in names:
            try:
                return getattr(self.sim, name)
            except Exception:
                continue
        raise AttributeError(f"CoppeliaSim API 中找不到常量: {names}")

    def _stop_existing_simulation(self) -> None:
        """如果场景已经在运行，先停掉，保证从确定状态开始。"""
        try:
            stopped = int(self._get_const("simulation_stopped"))
            state = int(self.sim.getSimulationState())
            if state == stopped:
                return
            print("[init] simulation is already running; stopping it first...")
            self.sim.stopSimulation()
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if int(self.sim.getSimulationState()) == stopped:
                    break
                time.sleep(0.02)
        except Exception as exc:
            print(f"[init][warn] could not stop existing simulation: {exc}")

    def _reset_stepping(self) -> None:
        """setStepping 是可累加的；先清掉之前残留的层级。"""
        for _ in range(20):
            try:
                previous = self.sim.setStepping(False)
            except Exception:
                break
            if not previous:
                break

    # ------------------------------------------------------------------
    # 3.7 关节发现
    # ------------------------------------------------------------------
    def _safe_alias(self, handle: int, option: int) -> str:
        """读取 alias/path；不同版本不支持 option 时回退。"""
        try:
            value = self.sim.getObjectAlias(handle, option)
        except Exception:
            try:
                value = self.sim.getObjectAlias(handle)
            except Exception:
                value = f"<handle {handle}>"
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        return str(value)

    def _joint_type_constant(self):
        return self._get_const("sceneobject_joint", "object_joint_type")

    def _resolve_joints(self, joint_paths=None) -> Dict[str, List[int]]:
        """优先使用显式路径；否则自动发现。"""
        if joint_paths:
            return self._resolve_explicit_joints(joint_paths)
        return self._discover_joints()

    def _resolve_explicit_joints(self, joint_paths) -> Dict[str, List[int]]:
        """把 JOINT_PATHS（路径或 handle）解析成 {leg: [h0,h1,h2]}。"""
        if isinstance(joint_paths, dict):
            mapping = {}
            for leg in LEGS:
                if leg not in joint_paths:
                    raise ValueError(f"JOINT_PATHS 缺少腿 '{leg}'")
                mapping[leg] = list(joint_paths[leg])
        else:
            items = list(joint_paths)
            if len(items) != 18:
                raise ValueError(f"JOINT_PATHS 平铺列表需要 18 个元素，收到 {len(items)}")
            mapping = {leg: items[i * 3:(i + 1) * 3] for i, leg in enumerate(LEGS)}

        result: Dict[str, List[int]] = {}
        print("[joints] using explicit JOINT_PATHS:")
        for leg in LEGS:
            if len(mapping[leg]) != 3:
                raise ValueError(f"JOINT_PATHS['{leg}'] 需要 3 个关节，收到 {len(mapping[leg])}")
            handles = []
            for item in mapping[leg]:
                if isinstance(item, str):
                    try:
                        handle = int(self.sim.getObject(item))
                    except Exception as exc:
                        raise ValueError(f"找不到关节路径 '{item}': {exc}") from exc
                else:
                    handle = int(item)
                if handle < 0:
                    raise ValueError(f"找不到关节: {item}")
                handles.append(handle)
            result[leg] = handles
            print(f"       {leg}: {handles}")
        return result

    def _discover_joints(self) -> Dict[str, List[int]]:
        """从场景中获取所有 joint，按名称解析成 6×3。"""
        joint_type = self._joint_type_constant()
        handles = self.sim.getObjectsInTree(
            self.sim.handle_scene, joint_type, 0
        )
        if handles is None:
            handles = []
        elif isinstance(handles, int):
            handles = [handles]
        else:
            handles = list(handles)

        records = []
        for handle in handles:
            handle = int(handle)
            if handle < 0:
                continue
            records.append({
                "handle": handle,
                "alias": self._safe_alias(handle, -1),
                "path": self._safe_alias(handle, 2),
            })

        print(f"[joints] found {len(records)} joint(s) in scene:")
        for i, rec in enumerate(records):
            print(f"       [{i:02d}] handle={rec['handle']:<5} "
                  f"alias={rec['alias']:<24} path={rec['path']}")

        # 按名称解析
        parsed: Dict[str, Dict[int, int]] = {leg: {} for leg in LEGS}
        for rec in records:
            # 优先解析 alias；alias 信息不全时再看完整 path
            leg, idx = _parse_joint_name(rec["alias"])
            if leg is None or idx is None:
                leg_path, idx_path = _parse_joint_name(rec["path"])
                if leg is None:
                    leg = leg_path
                if idx is None:
                    idx = idx_path
            if leg is None or idx is None:
                continue
            if idx not in parsed[leg]:
                parsed[leg][idx] = rec["handle"]
            else:
                print(f"[joints][warn] duplicate {leg}[{idx}] "
                      f"({rec['alias']}); ignored")

        if all(len(parsed[leg]) == 3 for leg in LEGS):
            print("[joints] auto-detected 6 legs x 3 joints by name.")
            return {leg: [parsed[leg][i] for i in range(3)] for leg in LEGS}

        # 兜底：场景恰好 18 个 joint，就按完整路径排序后每 3 个一组
        if len(records) == 18:
            ordered = sorted(records, key=lambda r: _natural_key(r["path"]))
            print("[joints][warn] 名称解析不完整，但场景恰好有 18 个 joint；")
            print("               按路径排序每 3 个一组，请核对打印的映射！")
            return {
                leg: [ordered[i * 3 + j]["handle"] for j in range(3)]
                for i, leg in enumerate(LEGS)
            }

        raise RuntimeError(
            "自动识别失败：需要 6 条腿 × 3 个关节 = 18 个 joint。\n"
            f"场景中检测到 {len(records)} 个 joint。\n"
            "请从上面的 handle/alias/path 列表中挑选 18 个关节，\n"
            "填入脚本顶部的 JOINT_PATHS 字典后重试。\n"
            "示例：\n"
            "JOINT_PATHS = {\n"
            "  'FL': ['/Hexapod/leg_FL/joint_1', "
            "'/Hexapod/leg_FL/joint_2', '/Hexapod/leg_FL/joint_3'],\n"
            "  ...\n"
            "}"
        )

    def _force_kinematic_mode(self) -> None:
        """尝试把 18 个关节都切成 kinematic 模式。"""
        mode = self._get_const("jointmode_kinematic")
        failed = []
        for leg in LEGS:
            for idx, handle in enumerate(self.joints[leg]):
                try:
                    self.sim.setJointMode(handle, mode)
                except Exception:
                    try:
                        # 兼容旧版 API 的三参数形式
                        self.sim.setJointMode(handle, mode, 0)
                    except Exception as exc:
                        failed.append((leg, idx, handle, str(exc)))

        if failed:
            print("[mode][warn] 以下关节无法自动切换为 kinematic：")
            for leg, idx, handle, error in failed:
                print(f"       {leg}[{idx}] handle={handle}: {error}")
            print("       请在 CoppeliaSim 中双击关节 -> Joint -> "
                  "Mode 选择 Kinematic。")
        else:
            print("[mode] all 18 joints set to kinematic.")

    def _prepare_kinematic_model(self) -> None:
        """
        为 Yaw 三角步态准备“运动学动画”模式。

        关键发现
        --------
        Ant 的 /Ant/legBase/targetLeg1..6 是 **IK target dummy**（linktype=5）。
        它们会把腿的 link 绑在 IK 目标上：
          - setJointPosition() 能改关节角；
          - 但 link/足端完全不动，像被“钉住”。
        把 targetLeg1..6 的 dummytype/linktype 归零后，静态 link 就能
        在仿真运行中正常跟随 setJointPosition。

        因此本方法：
          1. 把所有 shape 设为 Static（防重力下坠）；
          2. 解除 targetLeg1..6 的 IK target 绑定；
          3. 记录原属性，shutdown() 时恢复。
        """
        self._kinematic_active = False
        self._saved_shape_flags = []
        self._saved_dummy_flags = []

        try:
            root = int(self.sim.getObject(MODEL_ROOT))
            shape_type = self._get_const("sceneobject_shape", "object_shape_type")
            static_param = self._get_const("shapeintparam_static")
            dummy_param = self._get_const("dummyintparam_dummytype")
            link_param = self._get_const("dummyintparam_link_type")

            shapes = self.sim.getObjectsInTree(root, shape_type, 0)
            if shapes is None:
                shapes = []
            elif isinstance(shapes, int):
                shapes = [shapes]
            else:
                shapes = list(shapes)

            for handle in shapes:
                handle = int(handle)
                try:
                    static_old = int(self.sim.getObjectInt32Param(
                        handle, static_param))
                except Exception:
                    static_old = 1
                self._saved_shape_flags.append((handle, static_old))
                try:
                    self.sim.setObjectInt32Param(handle, static_param, 1)
                except Exception:
                    pass

            released = 0
            if RELEASE_IK_TARGETS:
                dummies = self.sim.getObjectsInTree(root, 4, 0)
                if dummies is None:
                    dummies = []
                elif isinstance(dummies, int):
                    dummies = [dummies]
                for dummy in dummies:
                    dummy = int(dummy)
                    alias = self._safe_alias(dummy, -1).lower()
                    if "targetleg" not in alias:
                        continue
                    try:
                        dtype_old = int(self.sim.getObjectInt32Param(
                            dummy, dummy_param))
                    except Exception:
                        dtype_old = 5
                    try:
                        ltype_old = int(self.sim.getObjectInt32Param(
                            dummy, link_param))
                    except Exception:
                        ltype_old = 5
                    self._saved_dummy_flags.append(
                        (dummy, dtype_old, ltype_old))
                    try:
                        self.sim.setObjectInt32Param(dummy, dummy_param, 0)
                        self.sim.setObjectInt32Param(dummy, link_param, 0)
                        released += 1
                    except Exception:
                        pass

            self._kinematic_active = True
            print(f"[kinematic] static {len(shapes)} shapes under {MODEL_ROOT}; "
                  f"released {released} IK target dummy(s).")
        except Exception as exc:
            print(f"[kinematic][warn] 无法准备 {MODEL_ROOT}: "
                  f"{type(exc).__name__}: {exc}")
            print("           如果模型不是 /Ant，请修改 MODEL_ROOT。")

    def _restore_kinematic_model(self) -> None:
        """恢复 shape 的 static 属性与 IK target dummy 类型。"""
        if not getattr(self, "_kinematic_active", False):
            return
        try:
            static_param = self._get_const("shapeintparam_static")
            dummy_param = self._get_const("dummyintparam_dummytype")
            link_param = self._get_const("dummyintparam_link_type")
            for handle, static_old in self._saved_shape_flags:
                try:
                    self.sim.setObjectInt32Param(
                        handle, static_param, static_old)
                except Exception:
                    pass
            for dummy, dtype_old, ltype_old in self._saved_dummy_flags:
                try:
                    self.sim.setObjectInt32Param(
                        dummy, dummy_param, dtype_old)
                    self.sim.setObjectInt32Param(
                        dummy, link_param, ltype_old)
                except Exception:
                    pass
            print("[kinematic] restored shape/static and IK targets.")
        except Exception as exc:
            print(f"[kinematic][warn] restore failed: "
                  f"{type(exc).__name__}: {exc}")
        finally:
            self._kinematic_active = False

    # ------------------------------------------------------------------
    # 3.8 运动学控制
    # ------------------------------------------------------------------
    def _set_leg(self, leg: str, q0: float, q1: float, q2: float) -> None:
        """设置一条腿的 3 个关节角。"""
        h0, h1, h2 = self.joints[leg]
        self.sim.setJointPosition(h0, float(q0))
        self.sim.setJointPosition(h1, float(q1))
        self.sim.setJointPosition(h2, float(q2))

    def stand(self) -> None:
        """保持站立姿态（所有 18 个关节设为预设角度）。"""
        for leg in LEGS:
            q = self.stand_pose[leg]
            self._set_leg(leg, q[0], q[1], q[2])
        self._gait_time = 0.0

    # ------------------------------------------------------------------
    # 3.9 底层排查：单关节暴力测试 / 限位与约束检查
    # ------------------------------------------------------------------
    def diagnose_joint(self, leg: str = "FL", index: int = 0) -> int:
        """打印指定关节的限位、模式、依赖、父子层级和模型 dummy。"""
        if leg not in self.joints:
            raise ValueError(f"leg 必须是 {LEGS}，收到 {leg}")
        if index not in (0, 1, 2):
            raise ValueError("index 必须是 0/1/2")
        handle = self.joints[leg][index]
        alias = self._safe_alias(handle, -1)
        path = self._safe_alias(handle, 2)
        q_before = None
        interval_before = None
        cyclic_before = True
        try:
            q_before = float(self.sim.getJointPosition(handle))
            interval_before = self.sim.getJointInterval(handle)
            cyclic_before = bool(interval_before[0])
        except Exception:
            pass

        print("\n" + "=" * 72)
        print(f"[diag] {leg} joint{index + 1}  handle={handle}")
        print(f"[diag] alias={alias}")
        print(f"[diag] path={path}")
        for label, fn in [
            ("joint mode", lambda: self.sim.getJointMode(handle)),
            ("joint interval", lambda: self.sim.getJointInterval(handle)),
            ("joint dependency", lambda: self.sim.getJointDependency(handle)),
            ("dynamically enabled", lambda: self.sim.isDynamicallyEnabled(handle)),
            ("target position", lambda: self.sim.getJointTargetPosition(handle)),
            ("current position", lambda: self.sim.getJointPosition(handle)),
        ]:
            try:
                print(f"[diag] {label:<20}: {fn()}")
            except Exception as exc:
                print(f"[diag] {label:<20}: ERR {type(exc).__name__}: {exc}")

        try:
            parent = self.sim.getObjectParent(handle)
            print(f"[diag] parent: h={parent} "
                  f"alias={self._safe_alias(parent, -1)} "
                  f"path={self._safe_alias(parent, 2)}")
            gp = self.sim.getObjectParent(parent)
            print(f"[diag] grandparent: h={gp} "
                  f"alias={self._safe_alias(gp, -1)}")
        except Exception as exc:
            print(f"[diag] parent: ERR {exc}")

        try:
            children = self.sim.getObjectsInTree(
                handle, self.sim.handle_all, 0)
            print(f"[diag] subtree ({len(children)} objects):")
            for h in children:
                print(f"         h={h:<5} type={self.sim.getObjectType(h):<2} "
                      f"alias={self._safe_alias(h, -1):<22} "
                      f"path={self._safe_alias(h, 2)}")
        except Exception as exc:
            print(f"[diag] subtree: ERR {exc}")

        # List dummies under the model: targetLeg1..6 / legBase are common in Ant.
        try:
            root = int(self.sim.getObject(MODEL_ROOT))
            dummies = self.sim.getObjectsInTree(root, 4, 0)  # 4 = dummy
            print(f"[diag] dummies under {MODEL_ROOT}: {len(dummies)}")
            loop_count = 0
            for h in dummies:
                a = self._safe_alias(h, -1)
                try:
                    dtype = self.sim.getObjectInt32Param(
                        h, self.sim.dummyintparam_dummytype)
                except Exception:
                    dtype = "?"
                try:
                    ltype = self.sim.getObjectInt32Param(
                        h, self.sim.dummyintparam_link_type)
                except Exception:
                    ltype = "?"
                try:
                    linked = self.sim.getLinkDummy(h)
                except Exception:
                    linked = "?"
                # 真正的动态闭链必须由 getLinkDummy() 指向另一个 dummy。
                # 注意：dummytype_dynloopclosure 的常量值是 0，不能只看 0。
                is_loop = (linked not in ("?", -1))
                if is_loop:
                    loop_count += 1
                if ("target" in a.lower() or "legbase" in a.lower()
                        or "base" in a.lower() or is_loop):
                    print(f"         h={h:<5} alias={a:<20} "
                          f"dummytype={dtype} linktype={ltype} "
                          f"linked={linked} "
                          f"{'[LOOP-CLOSURE]' if is_loop else ''}")
            print(f"[diag] dynamic loop closures found: {loop_count}")
            saved = getattr(self, "_saved_dummy_flags", [])
            if saved:
                print("[diag] IK targets released this session "
                      "(original dummytype/linktype):")
                for dummy, dtype_old, ltype_old in saved:
                    print(f"         h={dummy:<5} "
                          f"alias={self._safe_alias(dummy, -1):<20} "
                          f"dummytype={dtype_old} linktype={ltype_old}")
        except Exception as exc:
            print(f"[diag] dummies: ERR {exc}")

        # Try widening/removing joint interval (cyclic = no limits).
        if cyclic_before:
            print("[unseal] joint interval is already cyclic; "
                  "no limit change needed.")
        else:
            try:
                print("[unseal] trying setJointInterval(cyclic=True)...")
                self.sim.setJointInterval(handle, True, [0.0, 0.0])
                print(f"[unseal] after: {self.sim.getJointInterval(handle)}")
            except Exception as exc:
                print(f"[unseal] setJointInterval failed: {exc}")

        # Re-assert kinematic mode (3-arg form is required by 4.7).
        try:
            print("[unseal] setJointMode(kinematic, 0)...")
            try:
                self.sim.setJointMode(
                    handle, self.sim.jointmode_kinematic, 0)
            except Exception:
                self.sim.setJointMode(
                    handle, self.sim.jointmode_kinematic)
            print(f"[unseal] mode now: {self.sim.getJointMode(handle)}")
        except Exception as exc:
            print(f"[unseal] setJointMode failed: {exc}")

        try:
            print(f"[unseal] setJointPosition(0) -> "
                  f"{self.sim.setJointPosition(handle, 0.0)}")
            print(f"[unseal] actual position: "
                  f"{self.sim.getJointPosition(handle)}")
        except Exception as exc:
            print(f"[unseal] setJointPosition failed: {exc}")

        # 调试结束后恢复原角度/限位，避免污染下一次 CAPTURE_STAND_POSE。
        try:
            if q_before is not None:
                self.sim.setJointPosition(handle, q_before)
            if interval_before is not None:
                self.sim.setJointInterval(
                    handle, cyclic_before, list(interval_before[1]))
            print(f"[unseal] restored position={self.sim.getJointPosition(handle)} "
                  f"interval={self.sim.getJointInterval(handle)}")
        except Exception as exc:
            print(f"[unseal] restore failed: {exc}")

        return handle

    def test_motion(self, leg: str = "FL", index: int = 0,
                    amplitude_deg: float = 45.0, period: float = 1.0,
                    duration: float = 6.0) -> None:
        """
        绕过正弦步态，直接让一个关节在 ±amplitude_deg 之间来回摆。

        该函数只调用 sim.setJointPosition()，用于验证：
          - 关节是否能被写动；
          - 足端/link 是否跟随；
          - GUI 是否真的刷新。
        """
        handle = self.diagnose_joint(leg, index)
        amplitude = math.radians(float(amplitude_deg))
        period = max(float(period), 0.2)
        duration = max(float(duration), 1.0)

        print("\n" + "=" * 72)
        print(f"[test-motion] {leg} joint{index + 1}: "
              f"±{amplitude_deg:.1f} deg, period={period:.2f}s, "
              f"duration={duration:.1f}s")
        print("[test-motion] 如果关节角在变、GUI 不动，问题在 shape/约束层；"
              "如果关节角都不变，问题在关节模式/限位/约束。")

        # 找该腿足端，方便同时打印 tip 位置
        leg_index = {"FL": 0, "ML": 1, "RL": 2,
                     "FR": 5, "MR": 4, "RR": 3}[leg]
        tip = self.sim.getObject(
            f"/Ant/joint1Leg[{leg_index}]/link1RespondableLeg/joint2Leg/"
            f"link2RespondableLeg/joint3Leg/link3RespondableLeg/link3Leg/"
            f"tipLeg{leg_index + 1}"
        )

        t0 = self.sim_time()
        wall0 = time.time()
        last_report = -1.0
        mismatches = 0

        while self.sim_time() - t0 < duration:
            t = self.sim_time() - t0
            q = amplitude * math.sin(2.0 * math.pi * t / period)
            self.sim.setJointPosition(handle, q)
            self.sim.step()

            actual = float(self.sim.getJointPosition(handle))
            if abs(actual - q) > math.radians(5.0):
                mismatches += 1

            if t - last_report >= 0.5:
                last_report = t
                tip_pos = self.sim.getObjectPosition(tip, -1)
                print(f"[test-motion] t={t:5.2f}s  cmd={q:+.3f}  "
                      f"actual={actual:+.3f}  "
                      f"tip=({tip_pos[0]:+.3f},{tip_pos[1]:+.3f},{tip_pos[2]:+.3f})",
                      flush=True)

            # 让 GUI 肉眼可见地刷新（约 20 Hz）
            time.sleep(0.03)

            if time.time() - wall0 > duration * 3.0 + 5.0:
                print("[test-motion][warn] wall-clock timeout; break.")
                break

        print(f"[test-motion] done. command/actual mismatch frames: {mismatches}")
        if mismatches > 3:
            print("[test-motion][FAIL] 关节角没有跟上命令：检查 Limits / "
                  "joint mode / dependency / 闭链约束。")
        else:
            print("[test-motion] 关节角已跟随命令；如果 GUI 仍不动，"
                  "检查 shape 是否 Static / 被约束覆盖。")

    def walk_forward_steps(self, steps: int = 5,
                           body_step: float = 0.04,
                           rest: float = 1.5,
                           delay: float = 0.02) -> None:
        """
        向前走 N 个完整步态周期，然后保持站立。

        说明
        ----
        - 关节仍由 `_apply_gait()` 驱动；
        - 本体是 Static，这里用 `setObjectPosition()` 做同步的视觉前进
          （每周期前进 body_step 米），不是物理摩擦推进；
        - delay 用于让 GUI 肉眼可见，约 20~50 Hz。
        """
        steps = int(steps)
        if steps <= 0:
            return
        cycle = 1.0 / max(float(GAIT_FREQ), 1e-6)
        try:
            root = int(self.sim.getObject(MODEL_ROOT))
        except Exception:
            root = None
            body_step = 0.0

        self.stand()
        self.step()
        print(f"[walk] forward {steps} cycle(s), "
              f"cycle={cycle:.3f}s, body_step={body_step:.3f}m")

        for n in range(steps):
            t0 = self.sim_time()
            wall0 = time.time()
            while self.sim_time() - t0 < cycle:
                self.forward()
                self.step()
                if root is not None and body_step:
                    p = self.sim.getObjectPosition(root, -1)
                    p[0] -= body_step * self.dt / cycle
                    self.sim.setObjectPosition(root, -1, p)
                if delay > 0:
                    time.sleep(delay)
                if time.time() - wall0 > cycle * 10.0 + 5.0:
                    print("[walk][warn] simulation time not advancing; break.")
                    break
            if root is not None:
                p = self.sim.getObjectPosition(root, -1)
                print(f"[walk] cycle {n + 1}/{steps}  "
                      f"body=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})",
                      flush=True)

        # 走完后保持站立
        rest = max(float(rest), 0.0)
        t_rest = self.sim_time() + rest
        wall1 = time.time()
        while self.sim_time() < t_rest:
            self.stand()
            self.step()
            if delay > 0:
                time.sleep(delay)
            if time.time() - wall1 > rest * 10.0 + 5.0:
                print("[walk][warn] stand hold timeout; break.")
                break
        print("[walk] done; holding stand pose.")

    def _phase(self, t=None) -> float:
        """返回步态相位时间；不传 t 时使用内部时钟。"""
        return self._gait_time if t is None else float(t)

    def _apply_gait(self, t: float, turn: float = 0.0) -> None:
        """
        Ant 专用六足 Yaw 三角步态（Tripod Gait）。

        相位逻辑
        --------
        phase = 2π · GAIT_FREQ · t

        Group A = FL, MR, RL：leg_phase = phase + 0
        Group B = FR, ML, RR：leg_phase = phase + π

        两组相差半个周期（0.5 cycle）：
          - Group A 的 cos(leg_phase) > 0 时，A 组抬腿；
          - 此时 Group B 的 cos(leg_phase + π) < 0，B 组落地支撑；
          - 半个周期后互换。
        FL+MR+RL 与 FR+ML+RR 各自构成三角形支撑面，
        所以任意时刻至少有一组三角形着地。

        关节轨迹（Ant）
        ----------------
        joint1 (yaw)   : 前后扫动，是主要前进动力
                         yaw = base_yaw - side · YAW_SWING · scale · sin(phase)
        joint2 (pitch) : 腿向前摆时抬起，向后时落下
                         lift = max(0, cos(phase + LIFT_PHASE_OFFSET))
                         pitch = base_pitch - side · LIFT_AMOUNT · scale · lift
        joint3 (knee)  : 抬腿时配合弯曲，避免小腿拖地
                         knee = base_knee - side · KNEE_BEND · scale · lift

        其中 side = +1（左腿 FL/ML/RL），-1（右腿 FR/MR/RR）。
        左右 yaw 轴朝向相反，所以 side 让两侧同向发力；
        前面的负号让 sin(phase)=+1 时，左右腿都朝 -x（Ant 前方）迈。

        转向
        ----
        turn > 0（左转）：右侧三条腿 yaw 步幅增大，左侧减小；
        turn < 0（右转）：左侧三条腿 yaw 步幅增大，右侧减小。
        左右差速通过 scale 同时作用于 yaw / lift / knee。

        调参
        ----
        - 整体走反：YAW_SWING 取负；
        - 腿抬不起来/拖地：增大 LIFT_AMOUNT / KNEE_BEND，
          或把 LIFT_PHASE_OFFSET 调成 ±π/2；
        - 抬太高/抽搐：减小 LIFT_AMOUNT / KNEE_BEND；
        - 转向反了：对调 TURN_STRIDE_GAIN 的左右分配，或交换 turn 正负。
        """
        phase = 2.0 * math.pi * GAIT_FREQ * float(t)

        if turn > 0:            # 左转：右侧腿步幅增大
            left_scale = 1.0 - TURN_STRIDE_GAIN
            right_scale = 1.0 + TURN_STRIDE_GAIN
        elif turn < 0:          # 右转：左侧腿步幅增大
            left_scale = 1.0 + TURN_STRIDE_GAIN
            right_scale = 1.0 - TURN_STRIDE_GAIN
        else:
            left_scale = right_scale = 1.0

        for leg in LEGS:
            is_left = leg in LEFT_LEGS
            side = 1.0 if is_left else -1.0
            scale = left_scale if is_left else right_scale
            leg_phase = phase + LEG_PHASE[leg]

            base_yaw, base_pitch, base_knee = self.stand_pose[leg]

            # joint1 yaw：sin 前后扫动；右侧取反，保证左右同向。
            yaw = base_yaw - side * YAW_SWING * scale * math.sin(leg_phase)

            # joint2 pitch：cos 正半周抬腿；右侧取反。
            lift = max(0.0, math.cos(leg_phase + LIFT_PHASE_OFFSET))
            pitch = base_pitch - side * LIFT_AMOUNT * scale * lift

            # joint3 knee：抬腿时小腿配合弯曲；右侧取反。
            knee = base_knee - side * KNEE_BEND * scale * lift

            self._set_leg(leg, yaw, pitch, knee)

        self._gait_time = float(t)

    def forward(self, t=None) -> None:
        """前进步态。"""
        self._apply_gait(self._phase(t), turn=0.0)

    def turn_left(self, t=None) -> None:
        """左转：右侧三条腿（FR/MR/RR）yaw 步幅增大，左侧减小。"""
        self._apply_gait(self._phase(t), turn=+1.0)

    def turn_right(self, t=None) -> None:
        """右转：左侧三条腿（FL/ML/RL）yaw 步幅增大，右侧减小。"""
        self._apply_gait(self._phase(t), turn=-1.0)

    def stop(self) -> None:
        """
        停止步态，平滑回到站立姿态。
        注意：stop() 内部会主动推进仿真（平滑插值需要时间），
        所以只在演示结束时调用，不要在每个控制周期调用。
        """
        if not self._sim_started:
            return

        print("[stop] smoothly returning to stand pose...")
        current = {
            leg: [float(self.sim.getJointPosition(h)) for h in self.joints[leg]]
            for leg in LEGS
        }
        steps = max(1, int(round(self.stop_duration / max(self.dt, 1e-6))))
        for k in range(1, steps + 1):
            alpha = k / steps
            for leg in LEGS:
                for i, handle in enumerate(self.joints[leg]):
                    q = (1.0 - alpha) * current[leg][i] + alpha * self.stand_pose[leg][i]
                    self.sim.setJointPosition(handle, float(q))
            self.sim.step()
        self._gait_time = 0.0
        print("[stop] stand pose reached.")

    def step(self) -> None:
        """推进一个仿真步（供外部主循环调用）。"""
        self.sim.step()
        self._gait_time += self.dt

    def sim_time(self) -> float:
        """当前仿真时间（秒）。"""
        return float(self.sim.getSimulationTime())

    def start(self) -> None:
        """启动仿真；随后进入站立姿态并走一步。"""
        if self._sim_started:
            return
        print("[start] starting simulation (stepping mode)...")
        self.sim.startSimulation()
        self._sim_started = True
        self._gait_time = 0.0
        self.stand()
        self.step()

    def shutdown(self) -> None:
        """停止仿真并关闭步进模式。"""
        print("[shutdown] stopping simulation...")
        try:
            if self._sim_started:
                self.sim.stopSimulation()
        except Exception as exc:
            print(f"[shutdown][warn] stopSimulation: {exc}")
        finally:
            self._sim_started = False
        try:
            self.sim.setStepping(False)
        except Exception:
            pass
        self._restore_kinematic_model()
        print("[shutdown] done.")


# ===========================================================================
# 4. 演示主循环
# ===========================================================================
def run_phase(controller: RobotController, duration: float, action,
              name: str) -> None:
    """按仿真时间运行一个阶段；action 在每个仿真步前调用。"""
    print(f"\n[demo] {name} for {duration:.1f} s")
    t0 = controller.sim_time()
    wall_t0 = time.time()
    while controller.sim_time() - t0 < duration:
        action()
        controller.step()

        # 防止 stepping 没生效导致死循环
        if time.time() - wall_t0 > max(10.0, duration * 10.0):
            print("[demo][warn] simulation time is not advancing; break.")
            break


# ===========================================================================
# 5. --inspect：严格只读的关节路径检查
# ===========================================================================
def _get_client_version() -> str:
    try:
        return version("coppeliasim-zmqremoteapi-client")
    except PackageNotFoundError:
        return "NOT INSTALLED"


def _format_program_version(raw) -> str:
    """把 40700 格式化成 4.7.0；已经是字符串版本则原样返回。"""
    text = str(raw).strip()
    if text.isdigit() and len(text) >= 5:
        value = int(text)
        return f"{value // 10000}.{(value // 100) % 100}.{value % 100} (raw {text})"
    return text


def _get_server_version(sim) -> str:
    try:
        return str(sim.getStringParam(sim.stringparam_program_version))
    except Exception:
        pass
    try:
        return str(sim.getInt32Param(sim.intparam_program_version))
    except Exception:
        pass
    return "unknown"


def _read_alias(sim, handle: int, option: int = -1) -> str:
    """读取 alias（option=-1）或完整 path（option=2），失败时回退。"""
    try:
        value = sim.getObjectAlias(handle, option)
    except Exception:
        try:
            value = sim.getObjectAlias(handle)
        except Exception:
            value = f"<handle {handle}>"
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value)


def _as_handle_list(value) -> List[int]:
    """把 getObjectsInTree 的返回值统一成 int 列表。"""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _flatten_joint_paths(joint_paths) -> List[str]:
    """把 JOINT_PATHS（dict 或平铺 list）展开成路径列表。"""
    if not joint_paths:
        return []
    if isinstance(joint_paths, dict):
        paths = []
        for leg in LEGS:
            for item in joint_paths.get(leg, []):
                paths.append(str(item))
        return paths
    return [str(item) for item in joint_paths]


def inspect_joints(args) -> int:
    """
    只读检查：连接 + 读对象 + 打印 handle/alias/path。

    严禁在本函数中调用：
      sim.startSimulation / sim.stopSimulation / sim.step /
      sim.setStepping / sim.setJointPosition / sim.setJointMode
    因此机器人不会因为 --inspect 而移动。
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    timeout = float(getattr(args, "timeout", 10.0))
    print("=" * 72)
    print("CoppeliaSim --inspect (READ-ONLY joint path checker)")
    print("=" * 72)
    print(f"[env] python : {sys.executable}")
    print(f"[env] client : coppeliasim-zmqremoteapi-client {_get_client_version()}")
    print(f"[env] target : {args.host}:{args.port}")
    print("[inspect] READ-ONLY: no start/stop/step/joint writes in this mode.")

    # 建立连接（只读；不 stop 当前仿真）
    try:
        client = RemoteAPIClient(host=args.host, port=args.port)
        client.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        client.socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        sim = client.require("sim")
    except Exception as exc:
        print(f"\n[inspect][FAIL] 无法连接/require('sim'): "
              f"{type(exc).__name__}: {exc}")
        print("请确认 CoppeliaSim 已启动、ZMQ 端口正确。")
        return 2

    try:
        print(f"[sim] program version: "
              f"{_format_program_version(_get_server_version(sim))}")
        state = int(sim.getSimulationState())
        print(f"[sim] simulation state: {state} "
              f"(read-only; inspect 不会 stop/start 它)")
        if state != 0:
            print("[sim][warn] 当前仿真在运行；inspect 不会停它，"
                  "模型自己的脚本仍可能让关节动。想要完全静止，"
                  "请先在界面点 Stop。")
        print(f"[sim] simulation time : {sim.getSimulationTime():.4f} s")
        print(f"[sim] time step       : {sim.getSimulationTimeStep():.4f} s")
        try:
            scene_name = str(sim.getStringParam(
                sim.stringparam_scene_path_and_name))
            print(f"[sim] scene           : "
                  f"{scene_name if scene_name else '(untitled / empty scene)'}")
            if not scene_name:
                print("[sim][warn] 当前场景没有路径/文件名。若你已经在另一个窗口"
                      "打开了 ant.ttt/hexapod.ttt，说明 Python 连到了另一个"
                      " CoppeliaSim 实例（空场景）。请关闭多余实例，只保留"
                      "加载了模型的这个实例。")
        except Exception:
            pass
        try:
            all_objects = _as_handle_list(sim.getObjectsInTree(
                sim.handle_scene, sim.handle_all, 0))
            print(f"[sim] scene objects   : {len(all_objects)}")
        except Exception:
            pass
    except Exception as exc:
        print(f"[sim][warn] 读取仿真信息失败: {exc}")

    # 1) 检查路径列表：--paths > 已填写的 JOINT_PATHS > 占位符
    paths_text = getattr(args, "paths", None)
    if paths_text:
        paths = [p.strip() for p in paths_text.split(",") if p.strip()]
        source = "--paths"
    elif JOINT_PATHS:
        paths = _flatten_joint_paths(JOINT_PATHS)
        source = "JOINT_PATHS (already filled)"
    else:
        paths = list(INSPECT_JOINT_PATHS)
        source = "INSPECT_JOINT_PATHS (placeholders)"

    print(f"\n[inspect] checking {len(paths)} path(s) from {source}:")
    found = []
    missing = []
    for path in paths:
        try:
            handle = int(sim.getObject(path))
            error = ""
        except Exception as exc:
            handle = -1
            error = f"{type(exc).__name__}: {exc}"

        if handle >= 0:
            alias = _read_alias(sim, handle, -1)
            full_path = _read_alias(sim, handle, 2)
            print(f"  [OK]   {path}")
            print(f"         handle={handle}  alias={alias}")
            print(f"         full_path={full_path}")
            found.append((path, handle, alias, full_path))
        else:
            print(f"  [MISS] {path}" + (f"  ({error})" if error else "  (returned -1)"))
            missing.append(path)

    # 2) 如果 JOINT_PATHS 已填，直接按 6 组打印解析到的 handle
    joint_paths_complete = False
    if isinstance(JOINT_PATHS, dict) and JOINT_PATHS:
        print("\n[inspect] JOINT_PATHS mapping (resolved read-only):")
        all_ok = True
        for leg in LEGS:
            handles = []
            for path in JOINT_PATHS.get(leg, []):
                try:
                    handle = int(sim.getObject(str(path)))
                except Exception:
                    handle = -1
                handles.append(handle)
                if handle < 0:
                    all_ok = False
            print(f"  {leg}: {handles}")
        joint_paths_complete = all_ok
        if all_ok:
            print("[inspect] JOINT_PATHS complete: 6 legs x 3 = 18 handles.")
        else:
            print("[inspect][warn] JOINT_PATHS 中存在找不到的 path（handle=-1）。")

    # 3) 列出场景中所有关节，方便复制真实 path
    print("\n[inspect] all joints currently in the scene:")
    all_joints = []
    try:
        joint_type = getattr(sim, "sceneobject_joint", None)
        if joint_type is None:
            joint_type = getattr(sim, "object_joint_type")
        handles = _as_handle_list(
            sim.getObjectsInTree(sim.handle_scene, joint_type, 0)
        )
        for handle in handles:
            alias = _read_alias(sim, handle, -1)
            full_path = _read_alias(sim, handle, 2)
            all_joints.append((handle, alias, full_path))

        if all_joints:
            for handle, alias, full_path in all_joints:
                print(f"  handle={handle:<6} alias={alias:<26} path={full_path}")
            print(f"\n[inspect] total: {len(all_joints)} joints")
        else:
            print("  (场景里没有关节；确认模型已经加载，且不是只打开空场景)")
    except Exception as exc:
        print(f"  [warn] 无法枚举场景关节: {type(exc).__name__}: {exc}")

    # 3) 按 6 条腿 × 3 个关节打印解析映射（只读）
    parsed: Dict[str, Dict[int, int]] = {leg: {} for leg in LEGS}
    for handle, alias, full_path in all_joints:
        leg, idx = _parse_joint_name(alias)
        if leg is None or idx is None:
            leg_path, idx_path = _parse_joint_name(full_path)
            if leg is None:
                leg = leg_path
            if idx is None:
                idx = idx_path
        if leg is not None and idx is not None:
            parsed[leg].setdefault(idx, handle)

    print("\n[inspect] parsed 6 legs x 3 joints mapping:")
    complete = True
    for leg in LEGS:
        parts = []
        for idx in range(3):
            handle = parsed[leg].get(idx)
            if handle is None:
                parts.append(f"{JOINT_ROLE[idx]}=MISSING")
                complete = False
            else:
                parts.append(f"{JOINT_ROLE[idx]}=h{handle}")
        print(f"  {leg}: " + " | ".join(parts))
    if complete:
        print("[inspect] name-based mapping complete: 6 legs x 3 = 18 joints.")
    elif joint_paths_complete:
        print("[inspect] Ant 的 joint1Leg/joint2Leg/joint3Leg 名称本身不含"
              "腿编号；名称解析失败是正常的。以上面的 "
              "'JOINT_PATHS mapping' 为准，18 个 handle 已全部找到。")
    else:
        print("[inspect][warn] mapping incomplete; 请根据上面真实 path "
              "填写 JOINT_PATHS（或确认关节命名可被解析）。")

    # 4) 汇总
    print("\n" + "-" * 72)
    print(f"[inspect] preset paths: {len(found)} found, {len(missing)} missing")
    if missing:
        print("[inspect] missing paths:")
        for path in missing:
            print(f"  - {path}")
    print("\n[inspect] 下一步：")
    if joint_paths_complete:
        print("  1. JOINT_PATHS 已完整（18/18），无需再改路径；")
        print("  2. 先运行 --stand-only 调站立姿态；")
        print("  3. 站立稳定后，可运行完整 forward 演示（Yaw 三角步态）。")
    else:
        print("  1. 从上面的 'all joints' 列表复制真实 path；")
        print("  2. 填到 kinematic_gait.py 顶部的 JOINT_PATHS；")
        print("  3. 再次运行 --inspect，确认所有路径都是 [OK]。")
    print("  提示：--inspect 不会让机器人动；它不调用任何 joint 写操作。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoppeliaSim 六足机器人运动学控制演示（18 关节三角步态）"
    )
    parser.add_argument("--host", default="localhost",
                        help="CoppeliaSim ZMQ host（默认 localhost）")
    parser.add_argument("--port", type=int, default=23000,
                        help="CoppeliaSim ZMQ port（默认 23000）")
    parser.add_argument("--inspect", action="store_true",
                        help="只读检查关节路径：打印 handle/alias/path，"
                             "不启动/停止仿真，不写关节")
    parser.add_argument("--paths", default=None,
                        help="配合 --inspect：逗号分隔的关节路径；"
                             "不填则使用脚本顶部的 INSPECT_JOINT_PATHS")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="--inspect 的 ZMQ 收发超时秒数（默认 10）")
    parser.add_argument("--stand-only", action="store_true",
                        help="只启动仿真并保持站立，便于调 STAND_POSE（Ctrl+C 结束）")
    parser.add_argument("--test-motion", action="store_true",
                        help="单关节暴力测试：让指定关节 ±振幅来回摆")
    parser.add_argument("--unseal", action="store_true",
                        help="打印/尝试解除关节限位，并检查 dummy/闭链")
    parser.add_argument("--test-leg", default="FL", choices=list(LEGS),
                        help="--test-motion 的腿（默认 FL）")
    parser.add_argument("--test-index", type=int, default=0, choices=[0, 1, 2],
                        help="--test-motion 的关节索引：0=yaw, 1=pitch, 2=knee")
    parser.add_argument("--test-amplitude", type=float, default=45.0,
                        help="--test-motion 振幅（度，默认 45）")
    parser.add_argument("--test-period", type=float, default=1.0,
                        help="--test-motion 周期（秒，默认 1.0）")
    parser.add_argument("--test-duration", type=float, default=6.0,
                        help="--test-motion 持续时间（秒，默认 6）")
    parser.add_argument("--walk-steps", type=int, default=0,
                        help="向前走 N 个完整步态周期，然后保持站立")
    parser.add_argument("--walk-body-step", type=float, default=0.04,
                        help="每个步态周期本体的视觉前进量（米，默认 0.04）")
    parser.add_argument("--turn-right", action="store_true",
                        help="把演示中的左转改成右转")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    # --inspect 必须在创建 RobotController 之前分流：
    # RobotController.__init__ 会 stop 旧仿真、切换 stepping、强制 kinematic、
    # 捕获站立姿态；这些都不应该在“只看路径”的 inspect 模式里发生。
    if args.inspect:
        return inspect_joints(args)

    try:
        controller = RobotController(
            host=args.host,
            port=args.port,
            joint_paths=JOINT_PATHS,
        )
    except Exception as exc:
        print(f"\n[main][error] 初始化失败: {exc}")
        traceback.print_exc()
        return 1

    if args.test_motion or args.unseal:
        print("\n[test] 进入单关节暴力诊断模式。")
        try:
            if args.unseal:
                controller.diagnose_joint(args.test_leg, args.test_index)
            if args.test_motion:
                controller.start()
                controller.test_motion(
                    leg=args.test_leg,
                    index=args.test_index,
                    amplitude_deg=args.test_amplitude,
                    period=args.test_period,
                    duration=args.test_duration,
                )
        except KeyboardInterrupt:
            print("\n[test] KeyboardInterrupt")
        except Exception as exc:
            print(f"\n[test][error] {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            try:
                controller.stop()
            except Exception:
                pass
            controller.shutdown()
        return 0

    if args.walk_steps:
        print(f"\n[walk] 前进 {args.walk_steps} 个步态周期后站立。")
        try:
            controller.start()
            controller.walk_forward_steps(
                steps=args.walk_steps,
                body_step=args.walk_body_step,
            )
            controller.stop()
        except KeyboardInterrupt:
            print("\n[walk] KeyboardInterrupt")
        except Exception as exc:
            print(f"\n[walk][error] {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            try:
                controller.shutdown()
            except Exception:
                pass
        return 0

    if args.stand_only:
        print("\n[stand-only] 保持站立姿态；按 Ctrl+C 结束。")
        print("如果姿态不对，修改脚本顶部的 STAND_POSE 后重新运行。")
        try:
            controller.start()
            while True:
                controller.stand()
                controller.step()
        except KeyboardInterrupt:
            print("\n[stand-only] KeyboardInterrupt")
        finally:
            try:
                controller.stop()
            except KeyboardInterrupt:
                print("\n[stand-only] stop interrupted.")
            except Exception as exc:
                print(f"[stand-only][warn] stop(): {exc}")
            finally:
                controller.shutdown()
        return 0

    ok = True
    try:
        controller.start()

        # 需求规定的演示顺序：站立 -> 前进 -> 左转 -> 站立
        run_phase(controller, 2.0, controller.stand, "stand")
        run_phase(controller, 3.0, controller.forward, "forward")
        if args.turn_right:
            run_phase(controller, 2.0, controller.turn_right, "turn right")
        else:
            run_phase(controller, 2.0, controller.turn_left, "turn left")
        run_phase(controller, 1.0, controller.stand, "stand")

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt: stopping...")
    except Exception as exc:
        ok = False
        print(f"\n[main][error] {exc}")
        traceback.print_exc()
    finally:
        try:
            controller.stop()
        except KeyboardInterrupt:
            print("\n[main] stop interrupted.")
        except Exception as exc:
            print(f"[main][warn] stop(): {exc}")
        finally:
            controller.shutdown()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
