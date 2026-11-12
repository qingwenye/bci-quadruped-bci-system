# -*- coding: utf-8 -*-
"""
cca_decoder.py
==============
Subject 1 + Subject 2 的 SSVEP 四分类解码（纯手写 CCA，不依赖 MoABB）。

完整流程
--------
1. 循环读取 ../data/S1.mat、../data/S2.mat，兼容 MATLAB v5/v7 与 v7.3。
2. 每个文件筛出标签 1、6、11、16，各 6 个 block；两个被试共 48 trial。
3. 用 MNE 建立 64 通道 / 250 Hz 的 Info，做 50 Hz 陷波 + 6-16 Hz 带通。
4. 以“刺激起点”为事件做 Epoching：刺激后 0.2-1.7 s，窗长 1.5 s。
5. 手写正弦/余弦参考信号，用 sklearn 的 CCA 计算典型相关系数并分类。
6. 拼接两个被试的 epoch，输出逐 trial 结果、准确率、混淆矩阵。

配套脚本
--------
V2.0 的 compare_acc.py 将复用本文件的预处理与 CCA 函数，
对比 CCA / FBCCA 的准确率与耗时。

关于标签与频率的重要更正
------------------------
这个数据是 Wang et al. (2016) 的 40-target benchmark。40 个 target 在
刺激界面上按 5 行 × 8 列排列，因此官方 event id 与频率的对应关系是：

    id 1-8   -> 8, 9, 10, 11, 12, 13, 14, 15 Hz
    id 9-16  -> 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2 Hz
    id 17-24 -> 8.4, 9.4, ..., 15.4 Hz
    id 25-32 -> 8.6, 9.6, ..., 15.6 Hz
    id 33-40 -> 8.8, 9.8, ..., 15.8 Hz

所以标签 1 / 6 / 11 / 16 实际对应 **8.0 / 13.0 / 10.2 / 15.2 Hz**，
并不是 8 / 9 / 10 / 11 Hz（那是标签 1 / 2 / 3 / 4）。这一点与 MOABB
Wang2016 的 _events 映射以及本数据的频谱扫描结果一致。

因此本脚本默认：
    LABELS   = [1, 6, 11, 16]
    FREQS    = [8.0, 13.0, 10.2, 15.2]
    BANDPASS = (6, 16)   # 6-14 Hz 会把 15.2 Hz 的 label 16 衰减掉

如果你确实只想要 8/9/10/11 Hz 四个相邻频率，把下面配置改成：
    LABELS   = [1, 2, 3, 4]
    BANDPASS = (6.0, 14.0)
其余代码不用动。

另外，每个 trial 的 1500 个采样点并不是从刺激起点开始的：官方数据是
[-0.5 s, 5.5 s]（500 ms 刺激前基线 + 5 s 刺激 + 0.5 s 刺激后空白）。
因此本脚本把事件放在 trial 内的 0.5 s 处，再用 tmin=0.2 / tmax=1.7，
得到的正是“刺激后 0.2-1.7 s”。

运行环境
--------
Python 3.10 + numpy + scipy + h5py + mne + scikit-learn + matplotlib。
在项目根目录或 src/ 下运行均可：
    python src/stage1_decoders/cca_decoder.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径与实验配置
# ---------------------------------------------------------------------------
# 以脚本位置为基准：src/stage1_decoders/cca_decoder.py -> 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent

# 默认读取 S1 与 S2；也可用环境变量覆盖：
#   SSVEP_DATA_PATHS="D:\...\S1.mat;D:\...\S2.mat"
# 或只跑一个文件：
#   SSVEP_DATA_PATH="D:\...\S1.mat"
DEFAULT_DATA_PATHS = [ROOT / "data" / "S1.mat", ROOT / "data" / "S2.mat"]
if os.environ.get("SSVEP_DATA_PATHS"):
    DATA_PATHS = [
        Path(p) for p in os.environ["SSVEP_DATA_PATHS"].split(os.pathsep) if p.strip()
    ]
elif os.environ.get("SSVEP_DATA_PATH"):
    DATA_PATHS = [Path(os.environ["SSVEP_DATA_PATH"])]
else:
    DATA_PATHS = DEFAULT_DATA_PATHS

OUTPUT_PATH = Path(os.environ.get(
    "SSVEP_OUTPUT_PATH",
    ROOT / "docs" / "step_0.analysis" / "confusion_matrix.png",
))

FS = 250.0                 # 采样率 (Hz)
N_CHANNELS = 64            # 通道数
N_SAMPLES = 1500           # 每个 trial 的采样点数 (6 s)
N_BLOCKS = 6               # 每个 target 的 block 数

# 数据文件的 trial 起点是刺激前 0.5 s
STIM_ONSET_S = 0.5

# 需求：刺激后 0.2 s 开始，窗口 1.5 s
EPOCH_TMIN = 0.2
EPOCH_TMAX = 1.7

# 参考信号谐波数（SSVEP 通常 2-5 个谐波）
N_HARMONICS = 3

# 要解码的 4 个官方 event id
LABELS = [1, 6, 11, 16]

# 带通范围：为了保留 label 16 的 15.2 Hz 基频，放宽到 6-16 Hz
BANDPASS = (6.0, 16.0)
NOTCH_FREQ = 50.0

# CCA 使用的后枕区通道（8 通道）。
# 说明：64 通道全部做 6-16 Hz 窄带滤波后，通道间高度共线，协方差矩阵
# 条件数可达 1e11，sklearn CCA 会给出数值上退化的相关系数（全部 ≈1.0）。
# 因此 CCA 只在后枕区通道上进行；MNE 的滤波/分段仍然使用全部 64 通道。
CCA_CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

# Wang2016 官方 64 通道顺序（来自 MOABB Wang2016._ch_names，去掉 stim）
CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]


def wang2016_event_freq(event_id: int) -> float:
    """Wang2016 官方 5×8 排列：event id (1-40) -> 刺激频率 (Hz)。"""
    if not 1 <= int(event_id) <= 40:
        raise ValueError(f"event_id 必须在 1-40 之间，收到 {event_id}")
    row, col = divmod(int(event_id) - 1, 8)
    return 8.0 + 0.2 * row + col


FREQS = [wang2016_event_freq(label) for label in LABELS]


# ---------------------------------------------------------------------------
# 1. 数据加载（兼容 v5/v7 与 v7.3/HDF5）
# ---------------------------------------------------------------------------
def _to_eeg_layout(arr: np.ndarray) -> np.ndarray:
    """把各种存储顺序统一成 (n_channels, n_samples, n_targets, n_blocks)。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(
            f"EEG 张量应为 4 维 (64, 1500, 40, 6)，实际 shape={arr.shape}"
        )

    # MATLAB v5/v7: (64, 1500, 40, 6)
    if arr.shape[0] == N_CHANNELS and arr.shape[1] == N_SAMPLES:
        return arr

    # MATLAB v7.3 / h5py 通常把 MATLAB 的维度反过来存: (6, 40, 1500, 64)
    if arr.shape[3] == N_CHANNELS and arr.shape[2] == N_SAMPLES:
        return arr.transpose(3, 2, 1, 0)

    # 有些工具会存成 (40, 6, 64, 1500)
    if arr.shape[2] == N_CHANNELS and arr.shape[3] == N_SAMPLES:
        return arr.transpose(2, 3, 0, 1)

    # 兜底：找 64 和 1500 所在的轴
    ch_axis = int(np.argmax([s == N_CHANNELS for s in arr.shape]))
    t_axis = next(i for i, s in enumerate(arr.shape) if s == N_SAMPLES and i != ch_axis)
    other = [i for i in range(4) if i not in (ch_axis, t_axis)]
    # other 的两个轴分别是 target/block；按 (target, block) 处理
    arr = np.transpose(arr, (ch_axis, t_axis, other[0], other[1]))
    if arr.shape[0] != N_CHANNELS or arr.shape[1] != N_SAMPLES:
        raise ValueError(f"无法把 shape={arr.shape} 解释为 (64,1500,40,6)")
    return arr


def _load_hdf5(path: Path, var: str) -> np.ndarray:
    """读取 MATLAB v7.3 (HDF5) 文件中的变量。"""
    import h5py

    with h5py.File(path, "r") as f:
        if var in f and isinstance(f[var], h5py.Dataset):
            return np.asarray(f[var][()], dtype=np.float64)

        # 没有同名变量时，找最大的 4 维数值 dataset
        candidates: list[tuple[str, int]] = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                candidates.append((name, int(obj.size)))

        f.visititems(visit)
        if not candidates:
            raise KeyError(
                f"{path} 中找不到变量 '{var}'，也没有 4 维数值 dataset；"
                f"根节点: {list(f.keys())}"
            )
        name = max(candidates, key=lambda item: item[1])[0]
        return np.asarray(f[name][()], dtype=np.float64)


def load_mat(path: Path, var: str = "data") -> np.ndarray:
    """加载 .mat 中的 EEG 张量，自动识别 v5/v7 与 v7.3。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {path}\n"
            f"请确认 S1.mat / S2.mat 位于 {path.parent} 下，或用环境变量 "
            f"SSVEP_DATA_PATHS 指定完整路径。"
        )

    with path.open("rb") as fh:
        magic = fh.read(8)

    if magic.startswith(b"\x89HDF"):
        print(f"[load] MATLAB v7.3 (HDF5): {path}")
        arr = _load_hdf5(path, var)
    else:
        import scipy.io as sio

        print(f"[load] MATLAB v5/v7 (scipy): {path}")
        mat = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
        if var not in mat:
            keys = [k for k in mat if not k.startswith("__")]
            raise KeyError(f"{path} 中找不到变量 '{var}'；可用变量: {keys}")
        arr = mat[var]

    arr = _to_eeg_layout(arr)
    print(f"[load] EEG shape = {arr.shape}  "
          f"(channels, samples, targets, blocks)")
    return arr


def extract_trials(data: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """按 label 筛出 trial，返回 (n_trials, n_channels, n_samples), labels。"""
    n_targets = data.shape[2]
    for label in labels:
        if not 1 <= label <= n_targets:
            raise ValueError(f"label {label} 超出范围 1-{n_targets}")

    # (ch, time, target, block) -> (target, block, ch, time)
    x = data[:, :, [label - 1 for label in labels], :].transpose(2, 3, 0, 1)
    x = np.ascontiguousarray(x.reshape(-1, data.shape[0], data.shape[1]))
    y = np.repeat(np.asarray(labels, dtype=int), data.shape[3])
    print(f"[trials] {x.shape[0]} trials × {x.shape[1]} channels × "
          f"{x.shape[2]} samples; labels = {labels}, 每类 {data.shape[3]} 个")
    return x, y


# ---------------------------------------------------------------------------
# 2. MNE 预处理：Info / 陷波 / 带通 / 事件分段
# ---------------------------------------------------------------------------
def build_raw(trials: np.ndarray, y: np.ndarray,
              apply_bandpass: bool = True, bandpass=None):
    """
    把 trial 拼成连续 Raw，做 50 Hz 陷波，可选带通。

    默认保持 V1.0 的 CCA 行为：50 Hz notch + 6-16 Hz bandpass。
    FBCCA 需要保留 6-90 Hz 的宽频信息，因此传入
    ``apply_bandpass=False``，只做 notch，再在 fbcca_decoder.py 里
    对 Raw 做多个子带滤波。
    """
    import mne

    mne.set_log_level("ERROR")

    n_trials, n_channels, n_samples = trials.shape
    if n_channels != N_CHANNELS:
        raise ValueError(f"期望 {N_CHANNELS} 通道，实际 {n_channels}")
    if len(y) != n_trials:
        raise ValueError(f"trial 数 {n_trials} 与标签数 {len(y)} 不一致")

    # 官方数据单位是 µV，MNE 约定用 V
    raw_data = np.ascontiguousarray(
        trials.transpose(1, 0, 2).reshape(n_channels, n_trials * n_samples) * 1e-6
    )

    info = mne.create_info(CH_NAMES, FS, ch_types="eeg", verbose=False)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        info.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception as exc:  # pragma: no cover - 仅影响电极位置的可视化
        print(f"[warn] 标准电极位置未设置: {exc}")

    raw = mne.io.RawArray(raw_data, info, verbose=False)

    # 50 Hz 陷波（IIR，零相位）
    raw.notch_filter(
        freqs=NOTCH_FREQ,
        picks="eeg",
        method="iir",
        iir_params=dict(order=2, ftype="butter"),
        verbose=False,
    )
    # 默认的 CCA 带通；FBCCA 会跳过这一步，保留宽频 Raw
    band = BANDPASS if bandpass is None else tuple(bandpass)
    if apply_bandpass:
        raw.filter(
            band[0],
            band[1],
            picks="eeg",
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )
        band_text = f"{band[0]:.0f}-{band[1]:.0f} Hz"
    else:
        band_text = "none (broadband for FBCCA)"

    print(f"[mne] Raw: {raw.info['nchan']} ch, {raw.info['sfreq']:.0f} Hz, "
          f"{raw.n_times / FS:.1f} s; notch={NOTCH_FREQ:.0f} Hz, "
          f"band={band_text}")
    return raw


def build_epochs(raw, y: np.ndarray, n_samples: int, tmin: float | None = None,
                 tmax: float | None = None, labels: list[int] | None = None,
                 verbose: bool = True):
    """
    以刺激起点为事件做 Epoching。

    Parameters
    ----------
    tmin, tmax : float, optional
        相对刺激起点的时间（秒）。默认 0.2 / 1.7，即 1.5 s 窗。
    labels : list[int], optional
        参与分段的标签，默认全局 LABELS。
    verbose : bool
        是否打印分段信息。
    """
    import mne

    tmin = EPOCH_TMIN if tmin is None else float(tmin)
    tmax = EPOCH_TMAX if tmax is None else float(tmax)
    labels = LABELS if labels is None else list(labels)
    if tmax <= tmin:
        raise ValueError(f"tmax ({tmax}) 必须大于 tmin ({tmin})")

    events = np.array(
        [
            [int(round(i * n_samples + STIM_ONSET_S * FS)), 0, int(label)]
            for i, label in enumerate(y)
        ],
        dtype=int,
    )
    event_id = {str(label): int(label) for label in labels}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,       # 不做基线校正；带通已去掉直流
        preload=True,
        verbose=False,
    )
    if len(epochs) != len(y):
        raise RuntimeError(f"Epoching 后 trial 数 {len(epochs)} != 原始 {len(y)}")

    # CCA 只用后枕区通道；MNE 的 64 通道数据保留在 raw/epochs 中
    epochs.pick(CCA_CHANNELS)
    data = epochs.get_data(copy=True)
    if verbose:
        print(f"[mne] Epochs: {data.shape} (trials, channels, samples), "
              f"t = {epochs.times[0]:.3f} .. {epochs.times[-1]:.3f} s")
    return epochs, data


# ---------------------------------------------------------------------------
# 3. 手写 CCA
# ---------------------------------------------------------------------------
def get_reference_signals(freqs, t, fs, n_harmonics):
    """
    生成 SSVEP 正弦/余弦参考信号。

    Parameters
    ----------
    freqs : array-like
        刺激频率列表，例如 [8.0, 13.0, 10.2, 15.2]。
    t : array-like
        时间向量（秒），长度必须与 EEG epoch 的采样点数一致。
    fs : float
        采样率（Hz）。本函数用 t 直接计算，fs 仅用于 Nyquist 校验。
    n_harmonics : int
        谐波数。

    Returns
    -------
    refs : ndarray, shape (n_freqs, 2 * n_harmonics, n_samples)
        每个频率对应一个 (2H, N) 矩阵：sin/cos 成对排列，覆盖基频到 H 次谐波。
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    n_harmonics = int(n_harmonics)
    if t.size == 0:
        raise ValueError("t 不能为空")
    if n_harmonics < 1:
        raise ValueError("n_harmonics 必须 >= 1")
    if np.any(freqs <= 0):
        raise ValueError(f"频率必须为正数，收到 {freqs}")
    if np.any(freqs * n_harmonics >= fs / 2.0):
        warnings.warn(
            f"最高谐波 {freqs.max() * n_harmonics:.1f} Hz 已达到/超过 Nyquist "
            f"{fs / 2:.1f} Hz，请减少谐波数或检查频率。",
            RuntimeWarning,
        )

    refs = np.empty((freqs.size, 2 * n_harmonics, t.size), dtype=np.float64)
    for i, freq in enumerate(freqs):
        for h in range(1, n_harmonics + 1):
            omega_t = 2.0 * np.pi * h * freq * t
            refs[i, 2 * (h - 1), :] = np.sin(omega_t)
            refs[i, 2 * (h - 1) + 1, :] = np.cos(omega_t)
    return refs


def cca_classify(eeg_epoch, freqs, fs, labels=None, n_harmonics=N_HARMONICS,
                 return_scores=False):
    """
    对一个 EEG epoch 做 CCA 分类。

    Parameters
    ----------
    eeg_epoch : ndarray, shape (n_channels, n_samples)
        单个 trial 的 EEG。也接受转置后的 (n_samples, n_channels)。
    freqs : array-like
        候选频率，例如 [8.0, 13.0, 10.2, 15.2]。
    fs : float
        采样率。
    labels : array-like or None
        与 freqs 一一对应的标签。默认使用全局 LABELS（长度匹配时）。
    n_harmonics : int
        参考信号谐波数。
    return_scores : bool
        为 True 时额外返回每个频率的典型相关系数。

    Returns
    -------
    best_label : int
        最大典型相关系数对应的标签。
    scores : ndarray, optional
        仅当 return_scores=True 时返回，shape=(n_freqs,)。
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.exceptions import ConvergenceWarning

    x = np.asarray(eeg_epoch, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"eeg_epoch 应为二维，实际 shape={x.shape}")
    # 约定 (channels, samples)；若明显是 (samples, channels) 则转置
    if x.shape[0] > x.shape[1]:
        x = x.T

    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    n_samples = x.shape[1]
    t = np.arange(n_samples, dtype=np.float64) / fs
    refs = get_reference_signals(freqs, t, fs, n_harmonics)

    if labels is None:
        labels = LABELS if len(freqs) == len(LABELS) else list(range(1, len(freqs) + 1))
    labels = list(labels)
    if len(labels) != len(freqs):
        raise ValueError("labels 与 freqs 长度必须一致")

    scores = np.zeros(len(freqs), dtype=np.float64)
    for i, ref in enumerate(refs):
        cca = CCA(n_components=1, scale=True, max_iter=1000, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cca.fit(x.T, ref.T)
        x_score, y_score = cca.transform(x.T, ref.T)
        r = np.corrcoef(x_score[:, 0], y_score[:, 0])[0, 1]
        scores[i] = abs(float(r)) if np.isfinite(r) else 0.0

    best_idx = int(np.argmax(scores))
    best_label = int(labels[best_idx])
    if return_scores:
        return best_label, scores
    return best_label


# ---------------------------------------------------------------------------
# 4. 评估与可视化
# ---------------------------------------------------------------------------
def predict_epochs(epoch_data: np.ndarray, freqs=None, labels=None,
                   n_harmonics=None):
    """对 (n_trials, n_channels, n_samples) 批量预测，返回 preds, scores。"""
    freqs = FREQS if freqs is None else list(freqs)
    labels = LABELS if labels is None else list(labels)
    n_harmonics = N_HARMONICS if n_harmonics is None else int(n_harmonics)

    preds = np.zeros(len(epoch_data), dtype=int)
    scores = np.zeros((len(epoch_data), len(freqs)), dtype=np.float64)
    for i, epoch in enumerate(epoch_data):
        label, corr = cca_classify(
            epoch, freqs, FS, labels=labels,
            n_harmonics=n_harmonics, return_scores=True,
        )
        preds[i] = label
        scores[i] = corr
    return preds, scores


def evaluate(epochs, data: np.ndarray, freqs=None, labels=None):
    """遍历所有 epoch 预测，返回真实标签、预测标签、相关系数矩阵。"""
    y_true = epochs.events[:, 2].astype(int)
    preds, scores = predict_epochs(data, freqs=freqs, labels=labels)
    return y_true, preds, scores


def print_results(y_true, preds, scores, labels=None, freqs=None):
    """打印逐 trial 结果、每类准确率、相关系数与混淆矩阵。"""
    labels = LABELS if labels is None else list(labels)
    freqs = FREQS if freqs is None else list(freqs)

    header = (f"{'trial':>5} {'true':>4} {'pred':>4}  " +
              "  ".join(f"{f:>8.1f}Hz" for f in freqs))
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for i, (true, pred, corr) in enumerate(zip(y_true, preds, scores)):
        mark = "OK " if true == pred else "ERR"
        print(f"{i:5d} {true:4d} {pred:4d}  " +
              "  ".join(f"{v:10.3f}" for v in corr) + f"  {mark}")

    acc = float(np.mean(preds == y_true))
    n_correct = int(np.sum(preds == y_true))
    print(f"\nOverall accuracy: {acc * 100:.2f}%  ({n_correct}/{len(y_true)})")

    print("\nPer-class accuracy:")
    for label, freq in zip(labels, freqs):
        mask = y_true == label
        class_acc = float(np.mean(preds[mask] == label)) if mask.any() else float("nan")
        print(f"  target {label:2d} ({freq:5.1f} Hz): "
              f"{class_acc * 100:6.2f}%  (n={int(mask.sum())})")

    print("\nMean CCA correlation (rows=true class, cols=reference freq):")
    print("true\\ref " + "  ".join(f"{f:>8.1f}" for f in freqs))
    for label, freq in zip(labels, freqs):
        mask = y_true == label
        mean_row = scores[mask].mean(axis=0)
        print(f"{label:>3d} ({freq:5.1f}) " +
              "  ".join(f"{v:10.3f}" for v in mean_row))

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, preds, labels=labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("        " + "".join(f"{label:>6}" for label in labels))
    for label, row in zip(labels, cm):
        print(f"{label:>6}  " + "".join(f"{int(v):>6}" for v in row))

    print("\nNote: CCA is training-free (references are fixed sin/cos signals).")
    print("Each trial is predicted independently, so leave-one-out gives the")
    print("same result as full-sample resubstitution; no train/test split is needed.")
    return acc, cm


def plot_confusion_matrix(cm: np.ndarray, acc: float, out_path: Path,
                          labels=None, freqs=None, title: str | None = None):
    """画混淆矩阵并保存为 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = LABELS if labels is None else list(labels)
    freqs = FREQS if freqs is None else list(freqs)
    if title is None:
        title = f"CCA confusion matrix (acc = {acc * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0,
                   vmax=max(int(cm.max()), 1))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Trial count")

    tick_labels = [f"{label}\n{freq:g} Hz" for label, freq in zip(labels, freqs)]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=12)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[saved] {out_path}")


# ---------------------------------------------------------------------------
# 5. 主流程：循环 S1/S2 -> 拼接 -> 评估
# ---------------------------------------------------------------------------
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 72)
    print("SSVEP CCA decoding - multi-subject (S1 + S2)")
    print("=" * 72)
    print(f"Data files : {[str(p) for p in DATA_PATHS]}")
    print(f"Output     : {OUTPUT_PATH}")
    print(f"Labels     : {LABELS}")
    for label, freq in zip(LABELS, FREQS):
        print(f"  target {label:2d} -> {freq:.1f} Hz")
    print(f"Epoch      : stimulus + {EPOCH_TMIN:.1f} .. + {EPOCH_TMAX:.1f} s "
          f"({EPOCH_TMAX - EPOCH_TMIN:.1f} s window)")
    print(f"Band-pass  : {BANDPASS[0]:.1f} - {BANDPASS[1]:.1f} Hz, "
          f"notch {NOTCH_FREQ:.0f} Hz")
    print(f"CCA channels ({len(CCA_CHANNELS)}): {CCA_CHANNELS}")

    all_epochs = []
    all_y = []
    all_subject = []

    for path in DATA_PATHS:
        print("\n" + "-" * 72)
        print(f"Subject file: {path.name}")
        print("-" * 72)
        data = load_mat(path, var="data")
        trials, y = extract_trials(data, LABELS)
        raw = build_raw(trials, y)
        epochs, epoch_data = build_epochs(raw, y, n_samples=trials.shape[-1])

        all_epochs.append(epoch_data)
        all_y.append(epochs.events[:, 2].astype(int))
        all_subject.append(np.full(len(epochs), path.stem))

    epoch_data = np.concatenate(all_epochs, axis=0)
    y_true = np.concatenate(all_y)
    subject_ids = np.concatenate(all_subject)
    print("\n" + "=" * 72)
    print(f"[combined] {epoch_data.shape[0]} trials × {epoch_data.shape[1]} channels "
          f"× {epoch_data.shape[2]} samples from {len(DATA_PATHS)} subject(s)")
    print("=" * 72)

    # CCA 不需要训练；逐个 trial 独立预测，等价于 leave-one-out。
    preds, scores = predict_epochs(epoch_data)
    acc, cm = print_results(y_true, preds, scores)

    print("\nPer-subject accuracy:")
    for sid in dict.fromkeys(subject_ids):
        mask = subject_ids == sid
        subj_acc = float(np.mean(preds[mask] == y_true[mask]))
        print(f"  {sid}: {subj_acc * 100:6.2f}%  ({int(np.sum(preds[mask] == y_true[mask]))}"
              f"/{int(mask.sum())})")

    subject_names = "+".join(path.stem for path in DATA_PATHS)
    plot_confusion_matrix(
        cm, acc, OUTPUT_PATH,
        title=f"CCA confusion matrix - {subject_names} (acc = {acc * 100:.1f}%)",
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
