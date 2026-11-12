# -*- coding: utf-8 -*-
"""
streaming_loop.py
=================
V2.0 Stage 3：250 Hz 流式 SSVEP 闭环（3-band FBCCA + 时间窗防抖）。

这个文件不是“离线跑 48 个 trial”，而是模拟真实采集设备的连续数据流：

    S1.mat 的 24 个 trial 顺序拼成连续 64 通道 Raw
        ↓  每次 10 个采样点（40 ms）推入环形缓冲区
    collections.deque(maxlen=375)   # 最近 1.5 s
        ↓  每 50 个采样点（200 ms）取一次滑动窗口
    scipy.signal 实时滤波（notch + 3 子带）
        ↓
    3-band FBCCA -> (label, confidence)
        ↓
    时间窗防抖（1000 ms，>60%，平均置信度>0.3，锁定 1.5 s）
        ↓
    queue.Queue（只放确认后的指令）
        ↓
    Consumer 50 Hz：RobotController stand/forward/turn/stop

为什么必须是“三线程 + 队列”
---------------------------
- Stream 线程：严格按 250 Hz 推样本；它不能被 CCA 阻塞，否则采样时间戳会乱。
- Decode 线程：对每个 200 ms 窗口做滤波 + FBCCA；单次计算约 20 ms。
- Consumer 线程：严格 50 Hz 驱动仿真；它只读队列，绝不跑 CCA。
三者通过有界队列解耦：decode 偶尔慢一拍，stream 仍持续采样，consumer
仍按 20 ms 节拍执行动作。这就是“实时流式”与“离线批处理”的本质区别。
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# 保证从 src/stage3_closed_loop/ 直接运行也能 import 到 stage1/stage2
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage1_decoders import cca_decoder as cca  # noqa: E402
from stage1_decoders import fbcca_decoder as fbcca  # noqa: E402
from stage2_control.kinematic_gait import RobotController  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 流式参数 / 滤波器
# ---------------------------------------------------------------------------
FS = float(cca.FS)                       # 250 Hz
WINDOW_SAMPLES = int(round(1.5 * FS))    # 375
DECODE_EVERY = int(round(0.2 * FS))      # 50
DEFAULT_PACKET = 10                      # 10 samples = 40 ms
BANDCOUNT = 3
WEIGHTS = np.asarray(fbcca.weights_for(BANDCOUNT, "standard"), dtype=np.float64)

# 3-band FBCCA 子带；与 fbcca_decoder.FBCCA_BANDS_3 对齐
STREAM_BANDS = tuple(fbcca.FBCCA_BANDS_3)

# 50 Hz notch：质量因子 Q=30，250 Hz 采样，8 通道
NOTCH_B, NOTCH_A = iirnotch(50.0, 30.0, fs=FS)

# 预先设计好 3 个子带 SOS，避免每个窗口重复 butter()
SUB_BAND_SOS = tuple(
    butter(
        4, [lo, hi],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    for lo, hi in STREAM_BANDS
)


def decode_window(window: np.ndarray) -> tuple[int, float]:
    """
    对一个 (8, 375) 滑动窗口做实时滤波 + 3-band FBCCA。

    参数
    ----
    window : ndarray
        shape=(n_channels, n_times)，单位为原始 EEG（µV 或 V 均可，
        CCA 对整体缩放不敏感）。

    返回
    ----
    label : int
    confidence : float
       3 个子带加权并归一化后的最大得分，范围约 [0, 1]。

    为什么用 filtfilt / sosfiltfilt：
    ---------------------------------
    每个 200 ms 决策窗口都要重新滤波，不能依赖 MNE 的离线流水线。
    filtfilt/sosfiltfilt 是零相位滤波，不会像单次 lfilter 那样引入
    明显相位延迟；对 375 点的小片段来说，scipy 的实现足够快。
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"window 必须是二维，实际 {x.shape}")
    if x.shape[0] != len(cca.CCA_CHANNELS):
        raise ValueError(
            f"window 通道数应为 {len(cca.CCA_CHANNELS)}，实际 {x.shape[0]}"
        )

    # 1) 50 Hz 工频陷波（对 8 通道同时做）
    x = filtfilt(NOTCH_B, NOTCH_A, x, axis=1)

    scores = []
    for sos in SUB_BAND_SOS:
        # 2) 子带滤波：每个 subband 独立，保留 6-90 / 14-90 / 22-90 Hz
        band = sosfiltfilt(sos, x, axis=1)
        # 3) 该子带跑 CCA：4 个候选频率的典型相关系数
        _label, band_scores = cca.cca_classify(
            band,
            cca.FREQS,
            cca.FS,
            labels=cca.LABELS,
            n_harmonics=cca.N_HARMONICS,
            return_scores=True,
        )
        scores.append(band_scores)

    # 4) 标准 FBCCA 权重融合；归一化不改变 argmax，只把置信度缩放回 [0,1]
    stack = np.stack(scores, axis=0)                 # (3, 4)
    weighted = np.tensordot(WEIGHTS, stack, axes=(0, 0))
    weighted = weighted / float(WEIGHTS.sum())

    best = int(np.argmax(weighted))
    return int(cca.LABELS[best]), float(weighted[best])


# ---------------------------------------------------------------------------
# 2. 基于时间的防抖
# ---------------------------------------------------------------------------
@dataclass
class DebounceDecision:
    command: Optional[str]
    reason: str


class TimeDebouncer:
    """
    基于时间的防抖，替代 V1.0 的“按次数投票”。

    V1.0：每 1.5 s 一个结果，5 帧投票、3 票确认。
    V2.0：每 200 ms 一个结果，5 帧只覆盖 1 s；如果继续按“5 帧中 3 票”，
          确认会过快、且对一次误识别过于敏感。

    规则
    ----
    - 维护最近 1000 ms 的 (timestamp, label, confidence)；
    - 某标签数量占窗口总数 > 60%；
    - 该标签的平均置信度 > 0.30；
    - 确认后清空窗口并锁定 1.5 s；锁定期间不接收新结果。

    为什么用时间而不是次数：
    ------------------------
    真实设备的包大小、丢包、调度抖动都会改变“每 200 ms 一帧”的实际
    到达时间。按时间窗口统计，能保证“最近 1 s 的意图”这个物理含义
    不会因为帧数变化而漂移。
    """

    def __init__(self, window_s: float = 1.0, ratio: float = 0.60,
                 min_confidence: float = 0.30, cooldown_s: float = 1.5,
                 min_samples: int = 5):
        self.window_s = float(window_s)
        self.ratio = float(ratio)
        self.min_confidence = float(min_confidence)
        self.cooldown_s = float(cooldown_s)
        self.min_samples = int(min_samples)
        self.samples: deque[tuple[float, int, float]] = deque()
        self.locked_until = 0.0

    def update(self, label: int, confidence: float,
               now: Optional[float] = None) -> DebounceDecision:
        now = time.monotonic() if now is None else float(now)

        # 锁定期间不接收、不积累；1.5 s 后重新从干净窗口开始。
        # 1 ms 容差用于避免刚好落在边界时打印 LOCKED 0.0s。
        if now + 1e-3 < self.locked_until:
            remain = max(0.0, self.locked_until - now)
            return DebounceDecision(
                None, f"LOCKED {remain:.1f}s"
            )

        cutoff = now - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        self.samples.append((now, int(label), float(confidence)))

        total = len(self.samples)
        if total == 0:
            return DebounceDecision(None, "WINDOW 0")

        # 启动阶段 / 冷却刚结束时只有 1~2 帧；不能因为“1 帧 100%”就确认。
        # 200 ms 一帧时，5 帧约覆盖 0.8~1.0 s，符合 1000 ms 窗口的物理意义。
        if total < self.min_samples:
            return DebounceDecision(
                None, f"WINDOW {total}/{self.min_samples} (warming)"
            )

        counts = Counter(label for _t, label, _c in self.samples)
        top_label, top_count = counts.most_common(1)[0]
        top_ratio = top_count / total
        top_confs = [
            c for _t, label, c in self.samples if label == top_label
        ]
        top_mean = float(np.mean(top_confs)) if top_confs else 0.0

        # 严格 “超过 60%”，且该标签平均置信度 > 0.3
        if top_ratio > self.ratio and top_mean > self.min_confidence:
            command = ACTION_BY_LABEL.get(top_label)
            if command is not None:
                self.samples.clear()
                self.locked_until = now + self.cooldown_s
                name = ACTION_NAMES.get(command, command)
                return DebounceDecision(
                    command,
                    f"CONFIRM {name} {top_ratio:.0%} avg={top_mean:.2f}",
                )

        name = ACTION_NAMES.get(ACTION_BY_LABEL.get(top_label), "未知")
        return DebounceDecision(
            None,
            f"WINDOW {top_count}/{total} ({top_ratio:.0%}) "
            f"{name} avg={top_mean:.2f}",
        )


# 动作映射：流式模块只依赖 cca_decoder 的 LABEL 约定
ACTION_BY_LABEL = {
    1: "forward",
    6: "turn_left",
    11: "turn_right",
    16: "stop",
}
ACTION_NAMES = {
    "forward": "前进",
    "turn_left": "左转",
    "turn_right": "右转",
    "stop": "停止",
    "stand": "站立",
}


# ---------------------------------------------------------------------------
# 3. 流式运行时
# ---------------------------------------------------------------------------
class StreamingLoop:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_event = threading.Event()
        self.print_lock = threading.Lock()
        self.errors: list[str] = []

        # 环形缓冲区：maxlen=375，自动丢弃 1.5 s 之前的旧样本
        self.buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)
        # 有界队列：解码慢时只保留最新窗口，绝不无限堆积
        self.snapshot_queue: queue.Queue = queue.Queue(maxsize=1)
        self.command_queue: queue.Queue = queue.Queue()
        self.debouncer = TimeDebouncer(
            window_s=args.window_ms / 1000.0,
            ratio=args.ratio,
            min_confidence=args.min_conf,
            cooldown_s=args.cooldown,
            min_samples=args.min_samples,
        )

        self.stream_data: Optional[np.ndarray] = None
        self.samples_seen = 0

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message, flush=True)

    # ---- 数据 preload ----
    def prepare_stream(self) -> None:
        """
        把 S1.mat 的 24 个 trial 按时间顺序拼成连续 Raw。

        注意：这里不预先切 1.5 s 窗，也不做 MNE 离线滤波；
        后续的 notch / 子带滤波都在每个滑动窗口上现场完成。
        """
        self.log(f"[stream] loading {self.args.data} ...")
        data = cca.load_mat(Path(self.args.data), var="data")
        trials, _y = cca.extract_trials(data, cca.LABELS)

        # (trials, channels, samples) -> (channels, trials*samples)
        continuous = trials.transpose(1, 0, 2).reshape(
            trials.shape[1], -1
        )

        # 只保留 8 个后枕区通道；流式滤波在 8 通道上做，省 8 倍计算。
        ch_idx = [cca.CH_NAMES.index(ch) for ch in cca.CCA_CHANNELS]
        self.stream_data = np.ascontiguousarray(
            continuous[ch_idx], dtype=np.float32
        )
        self.log(
            f"[stream] continuous data: {self.stream_data.shape} "
            f"({self.stream_data.shape[1] / FS:.1f} s, 8 posterior channels)"
        )
        self.log(
            f"[stream] ring buffer={WINDOW_SAMPLES} samples (1.5 s); "
            f"decode every {DECODE_EVERY} samples (200 ms)"
        )

        # 关键：先在正式流开始前暖机一次。
        # 第一次 sklearn CCA / scipy 滤波会触发大量 import、BLAS 初始化和
        # 线程池创建，可能耗时 1~2 s；如果放在实时流里，首个窗口会明显
        # 迟到。暖机不计入实时日志，但能保证后续窗口稳定在 ~20 ms。
        if self.stream_data.shape[1] >= WINDOW_SAMPLES:
            warm = self.stream_data[:, :WINDOW_SAMPLES]
            t0 = time.perf_counter()
            decode_window(warm)
            warm_ms = (time.perf_counter() - t0) * 1000.0
            self.log(f"[stream] warmup decode: {warm_ms:.1f} ms "
                     f"(excluded from real-time latency)")

    # ---- 250 Hz 模拟 ----
    def stream_loop(self) -> None:
        """
        模拟设备：每 40 ms 推入 10 个采样点（250 Hz 等效）。

        也可以改成 1 个点/4 ms，但 Windows 的 sleep 分辨率不稳定；
        10 点/40 ms 的包模式更接近 OpenBCI 的 BLE/WiFi 包，也更容易
        保证“每 50 个点触发一次解码”的对齐。
        """
        if self.stream_data is None:
            return
        n_samples = self.stream_data.shape[1]
        packet = int(self.args.packet_size)
        decode_every = int(self.args.decode_every)
        next_decode = WINDOW_SAMPLES
        idx = 0
        wall0 = time.monotonic()

        try:
            while not self.stop_event.is_set():
                if time.monotonic() - wall0 >= self.args.max_seconds:
                    self.log(f"[stream] max-seconds={self.args.max_seconds:.1f}s, stop.")
                    break

                # 连续数据循环播放；包跨尾部时拼接
                end = idx + packet
                if end <= n_samples:
                    block = self.stream_data[:, idx:end]
                else:
                    block = np.concatenate(
                        [self.stream_data[:, idx:],
                         self.stream_data[:, :end - n_samples]],
                        axis=1,
                    )
                idx = end % n_samples

                for j in range(block.shape[1]):
                    self.buffer.append(block[:, j])
                    self.samples_seen += 1

                # buffer 满 1.5 s 后，按 50 点节拍触发；next_decode 递增
                if (self.samples_seen >= next_decode
                        and len(self.buffer) >= WINDOW_SAMPLES):
                    snapshot = np.asarray(self.buffer, dtype=np.float32).T
                    sim_t = self.samples_seen / FS
                    item = (snapshot, sim_t)
                    try:
                        self.snapshot_queue.put_nowait(item)
                    except queue.Full:
                        # 只保留最新窗口；旧窗口已经过时
                        try:
                            self.snapshot_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.snapshot_queue.put_nowait(item)
                        except queue.Full:
                            pass
                    next_decode += decode_every

                time.sleep(packet / FS)

        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"stream: {type(exc).__name__}: {exc}")
            self.stop_event.set()
        finally:
            # 用 None 作为 decoder 的结束哨兵
            try:
                self.snapshot_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.snapshot_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.snapshot_queue.put_nowait(None)
                except queue.Full:
                    pass

    # ---- 解码线程 ----
    def decode_loop(self) -> None:
        """
        只做两件事：滤波 + CCA，然后交给时间防抖。

        这个线程绝不碰 RobotController；即使单次解码慢到几十毫秒，
        consumer 的 50 Hz 节拍仍然独立运行。
        """
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.snapshot_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break

                window, sim_t = item
                t0 = time.perf_counter()
                label, confidence = decode_window(window)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                decision = self.debouncer.update(label, confidence)
                command = decision.command
                cmd_text = ACTION_NAMES.get(command, "—") if command else "—"
                label_text = ACTION_NAMES.get(
                    ACTION_BY_LABEL.get(label), "未知"
                )
                self.log(
                    f"[流式解码] t={sim_t:6.2f}s label={label:2d}({label_text}) "
                    f"conf={confidence:.2f} latency={latency_ms:5.1f}ms "
                    f"-> [时间防抖] {decision.reason} -> [下发] {cmd_text}"
                )

                if command:
                    self.command_queue.put(command)
                    if command == "stop":
                        # 由 consumer 平滑停止；这里只负责把 stop 放到队列。
                        self.log("[decode] 防抖确认 stop，交给 consumer 平滑停止。")

        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"decode: {type(exc).__name__}: {exc}")
            self.stop_event.set()
        finally:
            self.command_queue.put("__END__")

    # ---- 50 Hz 消费 / 机器人 ----
    def _drain_commands(self):
        latest = None
        end = False
        while True:
            try:
                item = self.command_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__END__":
                end = True
            else:
                latest = item
        return latest, end

    def consumer_loop(self) -> None:
        """
        50 Hz 控制循环：只消费队列，不跑滤波/CCA。

        每 20 ms 执行一次动作并 step()，保证步态相位连续；没有新指令
        时保持当前动作（默认 stand）。
        """
        controller = None
        stopped = False
        try:
            if self.args.dry_run:
                self.log("[consumer][dry-run] no CoppeliaSim; 50 Hz loop only.")
            else:
                self.log("[consumer] initializing RobotController ...")
                controller = RobotController(host=self.args.host,
                                             port=self.args.port)
                controller.start()
                self.log("[consumer] RobotController ready; 50 Hz loop.")

            current_action = "stand"
            while not self.stop_event.is_set():
                latest, end = self._drain_commands()

                if latest == "stop":
                    self.log("[consumer] received stop; smooth stop and exit.")
                    self.stop_event.set()
                    if controller is not None:
                        controller.stop()
                        stopped = True
                    break
                if end:
                    self.log("[consumer] queue END; exit.")
                    break

                if latest and latest != current_action:
                    current_action = latest
                    self.log(f"[机器人] 切换动作 -> "
                             f"{ACTION_NAMES[current_action]}")

                if controller is not None:
                    if current_action == "forward":
                        controller.forward()
                    elif current_action == "turn_left":
                        controller.turn_left()
                    elif current_action == "turn_right":
                        controller.turn_right()
                    else:
                        controller.stand()
                    # stepping 模式下，step() 才推进仿真时间；
                    # 同时提交本 tick 的关节角。
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
        description="250 Hz 流式 SSVEP 闭环（3-band FBCCA + time debounce）"
    )
    parser.add_argument(
        "--data", default=str(cca.ROOT / "data" / "S1.mat"),
        help="S1.mat 路径",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--packet-size", type=int, default=DEFAULT_PACKET,
                        help="每个模拟数据包的采样点数（默认 10=40ms）")
    parser.add_argument("--decode-every", type=int, default=DECODE_EVERY,
                        help="每多少个采样点解码一次（默认 50=200ms）")
    parser.add_argument("--window-ms", type=float, default=1000.0,
                        help="时间防抖窗口（ms，默认 1000）")
    parser.add_argument("--ratio", type=float, default=0.60,
                        help="确认所需标签占比（默认 >0.60）")
    parser.add_argument("--min-conf", type=float, default=0.30,
                        help="平均置信度阈值（默认 >0.30）")
    parser.add_argument("--cooldown", type=float, default=1.5,
                        help="确认后的锁定秒数（默认 1.5）")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="开始确认所需的最少窗口结果数（默认 5）")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="最长运行秒数（默认 60，防止没有 stop）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连接 CoppeliaSim，只验证流式/防抖/线程")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    loop = StreamingLoop(args)
    try:
        loop.prepare_stream()
    except Exception as exc:
        print(f"[main][error] prepare_stream: {type(exc).__name__}: {exc}")
        return 1

    print("=" * 78)
    print("V2.0 Stage 3: 250 Hz streaming closed loop")
    print(f"data={args.data}")
    print(f"window={WINDOW_SAMPLES} samples / {WINDOW_SAMPLES / FS:.1f}s; "
          f"decode every {args.decode_every} samples / "
          f"{args.decode_every / FS * 1000:.0f}ms")
    print(f"bands={STREAM_BANDS} weights={[round(float(w), 4) for w in WEIGHTS]}")
    print(f"debounce window={args.window_ms:.0f}ms ratio>{args.ratio:.2f} "
          f"avg_conf>{args.min_conf:.2f} min_samples={args.min_samples} "
          f"cooldown={args.cooldown:.1f}s")
    print(f"mode={'DRY-RUN' if args.dry_run else 'COPPELIASIM'}")
    print("=" * 78)

    stream = threading.Thread(target=loop.stream_loop, name="stream", daemon=True)
    decode = threading.Thread(target=loop.decode_loop, name="decode", daemon=True)
    consumer = threading.Thread(target=loop.consumer_loop, name="consumer", daemon=True)
    stream.start()
    decode.start()
    consumer.start()

    try:
        while stream.is_alive() or decode.is_alive() or consumer.is_alive():
            stream.join(timeout=0.2)
            decode.join(timeout=0.2)
            consumer.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt; stopping...")
        loop.stop_event.set()

    loop.stop_event.set()
    stream.join(timeout=5.0)
    decode.join(timeout=5.0)
    consumer.join(timeout=5.0)

    if loop.errors:
        print("\n[main] errors:")
        for err in loop.errors:
            print("  -", err)
        return 1
    print("\n[main] streaming loop finished cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
