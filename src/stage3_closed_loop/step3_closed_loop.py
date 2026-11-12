# -*- coding: utf-8 -*-
"""
step3_closed_loop.py
====================
阶段 3：SSVEP 解码 + 防抖 + 六足机器人控制的闭环联调。

这个脚本不接真实脑电硬件，而是把 S1.mat 的 trial 当成“实时数据流”：
每 1.5 秒取一个 trial，现场跑一次 CCA，得到 (label, confidence)。

链路
----
producer thread
    读取 S1.mat -> MNE 预处理 -> 每 1.5 s 解码一个 trial
    -> 10% 随机翻转（模拟误识别）
    -> 15% 低置信度（模拟走神）
    -> Debouncer（置信度过滤 / 5 帧投票 / 1.5 s 冷却）
    -> queue.Queue

consumer thread
    50 Hz 从队列取最新指令
    -> RobotController.stand / forward / turn_left / turn_right
    -> 收到 stop 后平滑停止并 shutdown()

为什么用队列解耦
----------------
解码是“低频、重计算、时间抖动大”的任务；控制是“高频、轻量、
要求稳定节拍”的任务。如果放在同一个循环里，每次 CCA 都会阻塞
50 Hz 控制，导致步态停顿或关节抽搐。queue.Queue 把两条时间线分开：
producer 只生产防抖后的最终指令，consumer 只消费最新指令；队列吸收
解码耗时抖动，双方不需要共享动作状态，也不互相阻塞。

三层防抖的物理意义
------------------
1) 置信度过滤：CCA 最大相关系数低 = 当前没有稳定 SSVEP（走神、
   眨眼、电极噪声），此时应输出 No-control，而不是强行 argmax；
2) 滑动窗口投票：单帧误识别是孤立事件，最近 5 帧中至少 3 帧一致
   才说明用户意图稳定，能显著降低误触发；
3) 冷却锁定：确认一个动作后给机器人 1.5 s 完成步态/转向，避免同一
   指令被连续重复下发导致机械抖动和队列拥塞。

运行
----
    conda activate bci-quadruped
    cd D:\bci-quadruped-project
    python src\stage3_closed_loop\step3_closed_loop.py

只验证调度和防抖、不启动 CoppeliaSim：
    python src\stage3_closed_loop\step3_closed_loop.py --dry-run --max-seconds 20

注意
----
- 默认每个 trial 间隔 1.5 s，与真实 BCI 的决策周期一致；
- 默认最多跑 90 s，防止没有出现 stop 时无限运行；
- 只有 Debouncer 确认后的指令才会进入队列，原始解码不会直接驱动机器人。
"""

from __future__ import annotations

import argparse
import queue
import random
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# 保证从 src/ 直接运行也能导入同目录模块
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stage1_decoders.cca_decoder as s1  # noqa: E402
from stage2_control.kinematic_gait import RobotController  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 指令映射与防抖
# ---------------------------------------------------------------------------
LABEL_TO_ACTION = {
    1: "forward",      # 8.0 Hz  -> 前进
    6: "turn_left",    # 13.0 Hz -> 左转
    11: "turn_right",  # 10.2 Hz -> 右转
    16: "stop",        # 15.2 Hz -> 停止
}
ACTION_NAMES = {
    "forward": "前进",
    "turn_left": "左转",
    "turn_right": "右转",
    "stop": "停止",
    "stand": "站立",
}
END = "__END__"


@dataclass
class Decision:
    accepted: bool
    command: Optional[str]
    reason: str


class Debouncer:
    """
    三层防抖：置信度过滤 -> 滑动窗口投票 -> 冷却锁定。

    为什么不能把 CCA 的 argmax 直接发给机器人？
    ------------------------------------------------
    CCA 每个 1.5 s 窗口只给出“这 4 个频率里最像哪一个”，它并不知道
    用户是否真的在注视刺激。一次眨眼、电极接触变差、注意力漂移，都可能
    让某个错误频率的相关系数偶然排到第一。直接 argmax 驱动机器人会把
    这种单帧噪声变成真实动作，在六足机器人上表现为突然转向、抽搐、
    甚至原地反复切换。

    三层防抖分别处理三种不同时间尺度的不确定性：
      1) 置信度过滤：处理“这一帧是否可信”（信号质量/注意力）；
      2) 滑动窗口投票：处理“最近几帧是否一致”（意图稳定性）；
      3) 冷却锁定：处理“动作是否应该重复执行”（执行器/交互节奏）。

    三层顺序不能随意交换：
      - 先做置信度过滤，低质量帧不进入投票，避免污染窗口；
      - 再做冷却判断，执行器忙时不再积累新票；
      - 最后才投票，保证窗口里都是可用的有效帧。
    """

    def __init__(self, confidence_threshold: float = 0.30,
                 window_size: int = 5, majority: int = 3,
                 cooldown_s: float = 1.5):
        if window_size <= 0:
            raise ValueError("window_size 必须 > 0")
        if not 1 <= majority <= window_size:
            raise ValueError("majority 必须在 1..window_size 之间")
        self.confidence_threshold = float(confidence_threshold)
        self.window_size = int(window_size)
        self.majority = int(majority)
        self.cooldown_s = float(cooldown_s)
        self.window = deque(maxlen=self.window_size)
        self.cooldown_until = 0.0

    def feed(self, label: int, confidence: float,
             now: Optional[float] = None) -> Decision:
        now = time.monotonic() if now is None else float(now)
        action = LABEL_TO_ACTION.get(int(label))

        # 第 1 层：置信度过滤（No-control 门控）
        # confidence 是 4 个参考频率中最大的典型相关系数，范围 [0, 1]。
        # 它不是“分类概率”，而是“EEG 与某个 sin/cos 模板的线性相关强度”。
        # 低于阈值通常意味着：用户走神、眨眼、电极阻抗变差、没有注视刺激。
        # 这时最安全的动作是“不动作”，而不是把 argmax 的随机噪声当成指令。
        if confidence < self.confidence_threshold:
            return Decision(
                False, None,
                f"忽略（低置信度 {confidence:.2f}）",
            )

        # 第 2 层：冷却锁定
        # 指令一旦被确认，机器人需要时间完成一个步态周期/转向；如果下一帧
        # 立刻又确认同一指令，队列会被重复命令淹没，机械结构也会抖动。
        # 1.5 s 冷却既覆盖一个决策周期，也给用户“松手/换意图”的缓冲。
        # 1 ms 容差用于处理 1.5 s 决策周期刚好落在冷却边界的情况，
        # 避免日志出现没有意义的“冷却中 0.0s”。
        if now + 1e-3 < self.cooldown_until:
            remaining = max(0.0, self.cooldown_until - now)
            return Decision(
                False, None,
                f"冷却中（{remaining:.1f}s）",
            )

        if action is None:
            return Decision(False, None, "忽略（未知标签）")

        # 第 3 层：滑动窗口多数投票
        # 只把“置信度合格且不在冷却期”的帧放进窗口。单帧误识别（10%
        # 翻转）是孤立事件，连续 5 帧里要出现 3 次同一标签才确认；
        # 这等价于要求用户维持约 3 个决策周期（约 3~4.5 s）的稳定意图。
        # 对 4 分类来说，随机噪声连续 3 次碰巧同类的概率远低于单帧错误，
        # 因而能显著降低误触发，而代价只是增加约 2 个窗口的确认延迟。
        self.window.append(int(label))
        counts = Counter(self.window)
        top_label, top_count = counts.most_common(1)[0]

        if top_count >= self.majority:
            command = LABEL_TO_ACTION[top_label]
            # 确认后清空窗口：下一次输出必须重新积累 3 票，
            # 避免同一个意图在冷却结束后被旧票立刻重复确认。
            self.window.clear()
            self.cooldown_until = now + self.cooldown_s
            return Decision(
                True, command,
                f"确认（{ACTION_NAMES[command]} {top_count}/{self.window_size}）",
            )

        top_action = ACTION_NAMES.get(
            LABEL_TO_ACTION.get(top_label, ""), "未知")
        return Decision(
            False, None,
            f"投票中（{len(self.window)}/{self.window_size}，"
            f"{top_action}×{top_count}）",
        )


# ---------------------------------------------------------------------------
# 2. 数据 / 解码
# ---------------------------------------------------------------------------
def load_s1_epochs(data_path: Path):
    """加载 S1.mat，做一次预处理，返回可逐 trial 解码的 epoch 数组。"""
    data = s1.load_mat(data_path)
    trials, y = s1.extract_trials(data, s1.LABELS)
    raw = s1.build_raw(trials, y)
    _epochs, epoch_data = s1.build_epochs(
        raw, y, trials.shape[-1],
        tmin=s1.EPOCH_TMIN,
        tmax=s1.EPOCH_TMAX,
        verbose=True,
    )
    return epoch_data, y


def decode_one(epoch: np.ndarray, rng: random.Random) -> tuple[int, float]:
    """对一个 trial 跑 CCA，并模拟误识别 / 走神。"""
    label, scores = s1.cca_classify(
        epoch,
        s1.FREQS,
        s1.FS,
        labels=s1.LABELS,
        n_harmonics=s1.N_HARMONICS,
        return_scores=True,
    )
    # confidence 不是分类概率，而是 4 个参考频率里最高的典型相关系数：
    # 越接近 1 说明 EEG 与某个 sin/cos 模板越同步；越低越像噪声/走神。
    confidence = float(np.max(scores))

    # 15% 概率输出低置信度（模拟走神/眨眼）：
    # 强制压到阈值 0.30 以下，让 Debouncer 的第一层真正被触发。
    if rng.random() < 0.15:
        confidence = rng.uniform(0.10, 0.29)

    # 10% 概率随机翻转标签（模拟误识别）：
    # 模拟“信号很像但类别判错”的情况，用来验证投票层能否纠正单帧错误。
    if rng.random() < 0.10:
        candidates = [x for x in s1.LABELS if x != int(label)]
        label = rng.choice(candidates)

    return int(label), confidence


def format_decision(label: int, confidence: float,
                    decision: Decision) -> str:
    raw_name = ACTION_NAMES.get(LABEL_TO_ACTION.get(int(label)), "未知")
    issued = ACTION_NAMES.get(decision.command, "—") if decision.accepted else "—"
    return (f"[原始解码] {raw_name} ({confidence:.2f}) "
            f"-> [防抖状态] {decision.reason} "
            f"-> [下发指令] {issued}")


# ---------------------------------------------------------------------------
# 3. 生产者 / 消费者
# ---------------------------------------------------------------------------
class ClosedLoop:
    """
    闭环运行时容器：把共享状态、线程和队列收在一个对象里。

    这里刻意不让 producer/consumer 直接互相调用：
      - 共享状态只有 queue.Queue + threading.Event；
      - 任何一方异常都不会直接改对方的局部变量；
      - main 只负责启动、等待和收尾，逻辑边界清晰。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.command_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.print_lock = threading.Lock()
        self.errors: list[str] = []
        self.debouncer = Debouncer(
            confidence_threshold=args.confidence_threshold,
            window_size=args.window,
            majority=args.majority,
            cooldown_s=args.cooldown,
        )
        self.rng = random.Random(args.seed)

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message, flush=True)

    # ---- producer ----
    def producer_loop(self) -> None:
        """
        生产线程：模拟“脑电数据流 -> CCA -> 防抖 -> 队列”。

        为什么单独开线程
        ----------------
        MNE 预处理和 CCA 是重计算，单个 trial 的耗时可能从几十毫秒到
        几百毫秒不等；而机器人控制需要稳定的 50 Hz 节拍。如果解码和
        控制写在同一个循环里，每次 CCA 都会让机器人卡一下，表现为
        步态停顿、关节抖动，严重时队列/仿真时间错位。

        为什么用队列，而不是共享变量
        ----------------------------
        解码频率约 0.67 Hz（1/1.5 s），控制频率 50 Hz，两者相差 75 倍。
        queue.Queue 是单向缓冲：
          - 解码线程只管“生产已确认的指令”，不关心机器人当前在做什么；
          - 控制线程只管“消费最新指令”，不会被 CCA 阻塞；
          - 队列天然吸收两边的时钟抖动和偶发计算延迟；
          - 不需要加锁保护共享动作状态，减少竞态条件。
        队列里只放 Debouncer 确认后的指令，原始 label/confidence 不会
        进入控制链路，保证“未经防抖的噪声不能驱动机器人”。

        时间基准
        --------
        使用 time.monotonic()（单调时钟）而不是 time.time()：
        系统时间被 NTP 校准/手动修改时，monotonic 不会倒退，
        适合做 1.5 s 决策周期和 1.5 s 冷却计时。
        """
        try:
            # 数据只加载/预处理一次；如果每个 trial 都重新 build_raw，
            # 1.5 s 的实时性会被 MNE 滤波完全破坏。
            self.log(f"[producer] loading {self.args.data} ...")
            epoch_data, y = load_s1_epochs(Path(self.args.data))
            self.log(f"[producer] ready: {len(epoch_data)} trials; "
                     f"labels={sorted(set(int(v) for v in y))}")
            self.log("[producer] 每 %.1f s 解码一个 trial；"
                     "10%% 翻转，15%% 低置信度" % self.args.trial_interval)

            next_tick = time.monotonic()
            start = time.monotonic()
            trial_idx = 0

            while not self.stop_event.is_set():
                now = time.monotonic()
                if now < next_tick:
                    # Event.wait(timeout) 比 sleep 更好：main 或 consumer
                    # 触发 stop_event 时能立刻醒来，不必等完 1.5 s。
                    if self.stop_event.wait(next_tick - now):
                        break
                if time.monotonic() - start >= self.args.max_seconds:
                    # 安全阀：即使数据里一直没有出现 stop，也不会无限跑。
                    self.log(f"[producer] 达到 --max-seconds "
                             f"{self.args.max_seconds:.1f}s，结束。")
                    break

                # 从预加载的 epoch 里顺序取 trial；到尾部后循环，
                # 模拟在线数据流持续输入。epoch 已经是 8 个后枕区通道、
                # 1.5 s 窗口，正好是 CCA 的输入。
                epoch = epoch_data[trial_idx % len(epoch_data)]
                label, confidence = decode_one(epoch, self.rng)

                # 防抖是唯一允许产生“机器人指令”的地方；
                # 原始解码只用于终端可视化，不直接进入队列。
                decision = self.debouncer.feed(label, confidence)
                self.log(format_decision(label, confidence, decision))

                if decision.accepted:
                    # 队列里只放已经过三层防抖的最终动作。
                    self.command_queue.put(decision.command)
                    if decision.command == "stop":
                        # 不在生产线程里调 controller.stop()：
                        # 机器人对象的生命周期属于 consumer 线程。
                        self.log("[producer] 防抖确认 stop，"
                                 "交给消费者平滑停止。")
                        break

                trial_idx += 1
                # 固定节拍累加，而不是每次 now + interval，
                # 避免 CCA 耗时导致的累计漂移；如果解码偶尔超时，
                # 下一轮会立即补上，而不是把周期越拖越长。
                next_tick += self.args.trial_interval

        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"producer: {type(exc).__name__}: {exc}")
            self.stop_event.set()
        finally:
            # END 是生命周期哨兵，不是动作；无论正常结束、异常还是
            # max-seconds 超时，都要让 consumer 有机会退出 50Hz 循环。
            self.command_queue.put(END)

    # ---- consumer ----
    def _drain(self):
        """
        非阻塞地排空队列，只保留“最新一条”动作指令。

        为什么不用 queue.get() 阻塞等待？
        ---------------------------------
        控制线程必须每 20 ms 执行一次 controller.step()，即使这一拍没有
        新指令，也要维持 stand/forward 的关节轨迹。如果阻塞在 get()，
        没有新指令时机器人会停步，仿真时间也会停住。

        为什么丢弃旧指令？
        ------------------
        BCI 指令是“状态语义”（当前想前进/左转），不是需要逐条执行的
        事件队列。如果消费者偶尔慢了一拍，堆在队列里的 forward、
        forward、left 已经过时；直接跳到最新 left 才能跟用户当前意图。
        返回的 end 是生命周期信号，和动作指令分开处理。
        """
        latest = None
        end = False
        while True:
            try:
                item = self.command_queue.get_nowait()
            except queue.Empty:
                break
            if item == END:
                end = True
            else:
                latest = item
        return latest, end

    def consumer_loop(self) -> None:
        """
        消费线程：50 Hz 驱动机器人，不参与任何 EEG/CCA 计算。

        为什么是 50 Hz？
        ----------------
        控制循环要持续调用 controller.forward()/turn_*/stand() 并
        controller.step()，才能让三角步态的相位连续推进。time.sleep(0.02)
        给出约 50 Hz 的墙钟节拍；真正的仿真时间步长由 RobotController
        内部 dt（当前 0.05 s）决定，二者共同决定视觉速度。

        为什么这里绝对不能跑 CCA？
        -------------------------
        CCA/MNE 是低频重计算（约 0.67 Hz 决策一次），控制是高频轻量
        动作。把解码放在本线程会让 20 ms 节拍被几十~几百毫秒的 CCA
        阻塞，机器人出现停顿/抽搐。解码放在 producer，本线程只读队列，
        这是“时间解耦”的核心。

        指令生命周期
        ------------
        - 没有新指令：保持 current_action（默认 stand），继续 step；
        - 新指令：切换动作，下一条 tick 开始执行；
        - stop：先 controller.stop() 平滑回站姿，再由 finally shutdown()
          恢复场景；stop 不在 producer 里执行，避免跨线程操作机器人。
        """
        controller = None
        stopped = False
        try:
            if self.args.dry_run:
                self.log("[consumer][dry-run] 不连接 CoppeliaSim，只模拟 50Hz。")
            else:
                self.log("[consumer] 初始化 RobotController ...")
                controller = RobotController(host=self.args.host,
                                             port=self.args.port)
                controller.start()
                self.log("[consumer] RobotController ready; 50Hz loop.")

            # 默认动作是站立；收到第一个确认指令前，机器人保持稳定。
            current_action = "stand"
            while not self.stop_event.is_set():
                # 非阻塞地拿最新指令：没有新指令也要继续 step，
                # 否则步态会停在半路。
                latest, end = self._drain()

                if latest == "stop":
                    self.log("[consumer] 收到 stop，平滑停止并退出。")
                    self.stop_event.set()
                    if controller is not None:
                        controller.stop()
                        stopped = True
                    break

                if end:
                    self.log("[consumer] 收到队列结束信号，退出。")
                    break

                if latest and latest != current_action:
                    current_action = latest
                    self.log(f"[机器人] 切换动作 -> "
                             f"{ACTION_NAMES[current_action]}")

                # 每个 tick 都按“当前动作”重算一次关节角。
                # 不能只在动作切换时调用一次 forward()：三角步态是
                # 连续相位轨迹，必须在每个控制周期重新计算并 step。
                if self.args.dry_run:
                    # dry-run 只验证调度/日志，不做仿真步进
                    pass
                elif current_action == "forward":
                    controller.forward()
                elif current_action == "turn_left":
                    controller.turn_left()
                elif current_action == "turn_right":
                    controller.turn_right()
                elif current_action == "stand":
                    controller.stand()

                if controller is not None:
                    # stepping 模式下，只有调用 step() 仿真时间才前进；
                    # 这一步同时把本 tick 的关节角提交给 CoppeliaSim。
                    controller.step()
                time.sleep(0.02)

        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"consumer: {type(exc).__name__}: {exc}")
            self.stop_event.set()
        finally:
            if controller is not None:
                if not stopped:
                    try:
                        controller.stop()
                    except Exception as exc:  # noqa: BLE001
                        self.errors.append(f"stop: {type(exc).__name__}: {exc}")
                try:
                    controller.shutdown()
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"shutdown: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SSVEP 解码 + 防抖 + 六足机器人闭环联调"
    )
    parser.add_argument("--data", default=str(s1.ROOT / "data" / "S1.mat"),
                        help="S1.mat 路径")
    parser.add_argument("--host", default="localhost",
                        help="CoppeliaSim host")
    parser.add_argument("--port", type=int, default=23000,
                        help="CoppeliaSim port")
    parser.add_argument("--trial-interval", type=float, default=1.5,
                        help="每个 trial 的间隔秒数（默认 1.5）")
    parser.add_argument("--max-seconds", type=float, default=90.0,
                        help="最长运行秒数（默认 90，防止没有 stop）")
    parser.add_argument("--confidence-threshold", type=float, default=0.30,
                        help="置信度阈值（默认 0.30）")
    parser.add_argument("--window", type=int, default=5,
                        help="投票窗口大小（默认 5）")
    parser.add_argument("--majority", type=int, default=3,
                        help="确认所需票数（默认 3）")
    parser.add_argument("--cooldown", type=float, default=1.5,
                        help="确认后的冷却秒数（默认 1.5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="误识别/走神模拟的随机种子")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连接 CoppeliaSim，只验证线程/队列/防抖")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    loop = ClosedLoop(args)

    producer = threading.Thread(target=loop.producer_loop,
                                name="bci-producer", daemon=True)
    consumer = threading.Thread(target=loop.consumer_loop,
                                name="robot-consumer", daemon=True)

    print("=" * 72)
    print("Stage 3: SSVEP -> Debouncer -> Ant closed loop")
    print(f"data={args.data}")
    print(f"trial_interval={args.trial_interval:.2f}s  "
          f"threshold={args.confidence_threshold:.2f}  "
          f"window={args.window}  majority={args.majority}  "
          f"cooldown={args.cooldown:.1f}s")
    print(f"mode={'DRY-RUN' if args.dry_run else 'COPPELIASIM'}")
    print("=" * 72)

    producer.start()
    consumer.start()

    try:
        while producer.is_alive() or consumer.is_alive():
            producer.join(timeout=0.2)
            consumer.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt; stopping...")
        loop.stop_event.set()
        loop.command_queue.put(END)

    loop.stop_event.set()
    loop.command_queue.put(END)
    producer.join(timeout=5.0)
    consumer.join(timeout=5.0)

    if loop.errors:
        print("\n[main] errors:")
        for err in loop.errors:
            print("  -", err)
        return 1
    print("\n[main] closed loop finished cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
