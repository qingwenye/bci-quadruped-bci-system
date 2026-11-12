# -*- coding: utf-8 -*-
"""
fbcca_decoder.py
================
V2.0 Stage 1：Filter Bank CCA（FBCCA）SSVEP 解码。

与 CCA 的区别
-------------
CCA 只使用一个 6-16 Hz 的宽带（预处理阶段已经带通），而 FBCCA 在
Raw 状态保留更宽的频带（6-90 Hz），然后用一组递进的高通子带分别做 CCA：

    [6, 90], [14, 90], [22, 90], [30, 90], [38, 90] Hz

每个子带都会给出 4 个候选频率的典型相关系数。低次子带包含基频，
高次子带更强调谐波；按标准 FBCCA 权重

    w_h = h^(-1.25) + 0.25,  h = 1..5

加权求和后取最大类别。低次子带权重更高，因为它们保留更多 SSVEP
基频能量；高次子带只做辅助。

消融
----
默认保留标准 5 带；可用 ``--bands 3`` 砍掉 (30,90)、(38,90) 两个
最高频子带，或用 ``--weight-mode low`` 对 5 带施加更强的低频权重。
该数据上 3 带 + 标准权重取得了更高的准确率和更低延迟，但 5 带仍作为
可复现基线保留。

注意
----
- 复用 cca_decoder.load_mat / extract_trials / build_raw；
- 不复用 cca_decoder.build_epochs：FBCCA 必须在 Raw 上先做子带滤波，
  再截取 0.2-1.7 s 的 Epoch；
- 本文件可以独立运行，默认跑 S1+S2 48 trial，输出混淆矩阵和准确率。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 兼容两种运行方式：
#   python src/stage1_decoders/fbcca_decoder.py
#   python -c "from stage1_decoders import fbcca_decoder"
try:
    from . import cca_decoder as cca
except ImportError:  # 直接作为脚本运行
    import cca_decoder as cca


# ---------------------------------------------------------------------------
# 1. FBCCA 配置
# ---------------------------------------------------------------------------
# 5 个递进子带；高频上限 90 Hz，低于 250 Hz 采样率的 Nyquist(125 Hz)。
FBCCA_BANDS = (
    (6.0, 90.0),
    (14.0, 90.0),
    (22.0, 90.0),
    (30.0, 90.0),
    (38.0, 90.0),
)

# 3-band ablation：砍掉最高频的 (30,90)、(38,90)
FBCCA_BANDS_3 = FBCCA_BANDS[:3]


def weights_for(n_bands: int, mode: str = "standard"):
    """
    FBCCA 子带权重。

    standard:
        w_h = h^(-1.25) + 0.25
    low:
        更强压制高频子带的启发式权重；
        5 带为 (2.0, 1.0, 0.5, 0.25, 0.15)。
    """
    if mode == "standard":
        return tuple(
            1.0 / (h ** 1.25) + 0.25
            for h in range(1, int(n_bands) + 1)
        )
    if mode == "low":
        base = (2.0, 1.0, 0.5, 0.25, 0.15)
        if not 1 <= int(n_bands) <= len(base):
            raise ValueError(f"low 权重最多支持 {len(base)} 个子带")
        return base[:int(n_bands)]
    raise ValueError(f"未知 weight mode: {mode!r}（可选 standard / low）")


# 标准 5 带权重，保留原常量名以兼容旧调用
FBCCA_WEIGHTS = weights_for(len(FBCCA_BANDS), "standard")


# ---------------------------------------------------------------------------
# 2. 子带 Raw / Epoch
# ---------------------------------------------------------------------------
def build_wideband_raw(trials: np.ndarray, y: np.ndarray):
    """
    构造只做 50 Hz notch、不做 6-16 Hz 带通的宽频 Raw。

    CCA 的 build_raw 默认会带通到 6-16 Hz；FBCCA 需要 6-90 Hz 的
    宽频信息，所以显式关闭带通，但仍复用 build_raw 的 Raw 拼接、
    Info、notch 逻辑。
    """
    return cca.build_raw(trials, y, apply_bandpass=False)


def build_subband_epochs(raw, y: np.ndarray, n_samples: int,
                         bands=FBCCA_BANDS,
                         tmin: float | None = None,
                         tmax: float | None = None):
    """
    在 Raw 上逐个子带滤波，然后再做 Epoch。

    返回
    ----
    subband_data : list[ndarray]
        长度 = 子带数；每个元素 shape=(n_trials, n_cca_channels, n_times)。

    为什么不在 build_epochs 之后做子带滤波？
    ----------------------------------------
    CCA 的 build_epochs 已经做了 6-16 Hz 带通，会把 16 Hz 以上的
    SSVEP 谐波提前丢掉；FBCCA 必须在滤波之前保留宽频 Raw。
    """
    import mne

    tmin = cca.EPOCH_TMIN if tmin is None else float(tmin)
    tmax = cca.EPOCH_TMAX if tmax is None else float(tmax)
    labels = list(cca.LABELS)

    events = np.array(
        [
            [int(round(i * n_samples + cca.STIM_ONSET_S * cca.FS)), 0, int(label)]
            for i, label in enumerate(y)
        ],
        dtype=int,
    )
    event_id = {str(label): int(label) for label in labels}

    out = []
    for lo, hi in bands:
        print(f"[fbcca] sub-band {lo:.0f}-{hi:.0f} Hz ...", flush=True)
        band_raw = raw.copy().filter(
            lo, hi,
            picks="eeg",
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )
        epochs = mne.Epochs(
            band_raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        if len(epochs) != len(y):
            raise RuntimeError(
                f"sub-band {lo}-{hi}: epochs={len(epochs)} != y={len(y)}"
            )
        epochs.pick(cca.CCA_CHANNELS)
        data = epochs.get_data(copy=True)
        out.append(data)

    return out


# ---------------------------------------------------------------------------
# 3. FBCCA 预测
# ---------------------------------------------------------------------------
def predict_fbcca(subband_data, freqs=None, labels=None,
                  n_harmonics=None, weights=None):
    """
    对每个子带分别跑 CCA，再按 FBCCA 权重融合。

    Parameters
    ----------
    subband_data : list[ndarray]
        每个子带的 (n_trials, n_channels, n_samples) 数据。
    freqs, labels, n_harmonics :
        与 CCA 相同；默认使用 cca_decoder 的全局配置。
    weights :
        子带权重；默认按子带数量使用 standard 权重。
        3 带 / 5 带 / low 权重实验都可显式传入。

    Returns
    -------
    preds : ndarray, shape=(n_trials,)
    scores : ndarray, shape=(n_trials, n_classes)
        加权并归一化后的融合得分，取值范围近似 [0, 1]。
    """
    if not subband_data:
        raise ValueError("subband_data 不能为空")
    if weights is None:
        weights = weights_for(len(subband_data), "standard")
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(subband_data):
        raise ValueError(
            f"权重数量 {len(weights)} 与子带数量 "
            f"{len(subband_data)} 不一致"
        )

    freqs = cca.FREQS if freqs is None else list(freqs)
    labels = cca.LABELS if labels is None else list(labels)
    n_harmonics = cca.N_HARMONICS if n_harmonics is None else int(n_harmonics)

    per_band_scores = []
    for data in subband_data:
        _preds, scores = cca.predict_epochs(
            data, freqs=freqs, labels=labels, n_harmonics=n_harmonics
        )
        per_band_scores.append(scores)

    # (n_bands, n_trials, n_classes)
    stack = np.stack(per_band_scores, axis=0)
    weighted = np.tensordot(weights, stack, axes=(0, 0))
    # 归一化到与单子带 CCA 相近的 [0,1] 尺度，便于统一置信度阈值。
    weighted = weighted / float(weights.sum())

    preds = np.asarray(labels, dtype=int)[np.argmax(weighted, axis=1)]
    return preds, weighted


# ---------------------------------------------------------------------------
# 4. 完整评估
# ---------------------------------------------------------------------------
def evaluate_fbcca(data_paths, output_path: Path, plot: bool = True,
                   bands=FBCCA_BANDS, weight_mode: str = "standard"):
    """加载 S1/S2 -> 宽频 Raw -> 子带 Epoch -> FBCCA -> 混淆矩阵。"""
    bands = tuple(bands)
    weights = weights_for(len(bands), weight_mode)
    print("=" * 72)
    print("SSVEP FBCCA decoding - multi-subject")
    print("=" * 72)
    print(f"Data files : {[str(p) for p in data_paths]}")
    print(f"Output     : {output_path}")
    print(f"Labels     : {cca.LABELS}")
    for label, freq in zip(cca.LABELS, cca.FREQS):
        print(f"  target {label:2d} -> {freq:.1f} Hz")
    print(f"Sub-bands  : {bands}")
    print(f"Weights    : {[round(w, 4) for w in weights]} "
          f"(mode={weight_mode})")
    print(f"Epoch      : stimulus + {cca.EPOCH_TMIN:.1f} .. + "
          f"{cca.EPOCH_TMAX:.1f} s")

    all_subbands = []
    all_y = []
    all_subject = []

    for path in data_paths:
        print("\n" + "-" * 72)
        print(f"Subject file: {path.name}")
        print("-" * 72)
        data = cca.load_mat(path, var="data")
        trials, y = cca.extract_trials(data, cca.LABELS)
        raw = build_wideband_raw(trials, y)
        subbands = build_subband_epochs(
            raw, y, n_samples=trials.shape[-1], bands=bands
        )
        all_subbands.append(subbands)
        all_y.append(y)
        all_subject.append(np.full(len(y), path.stem))

    n_bands = len(bands)
    combined = [
        np.concatenate([subject[b] for subject in all_subbands], axis=0)
        for b in range(n_bands)
    ]
    y_true = np.concatenate(all_y)
    subject_ids = np.concatenate(all_subject)

    print("\n" + "=" * 72)
    print(f"[combined] {combined[0].shape[0]} trials × "
          f"{combined[0].shape[1]} channels × {combined[0].shape[2]} samples")
    print("=" * 72)

    preds, scores = predict_fbcca(combined, weights=weights)
    acc, cm = cca.print_results(y_true, preds, scores)

    print("\nPer-subject accuracy:")
    for sid in dict.fromkeys(subject_ids):
        mask = subject_ids == sid
        n_ok = int(np.sum(preds[mask] == y_true[mask]))
        print(f"  {sid}: {n_ok / int(mask.sum()) * 100:6.2f}%  "
              f"({n_ok}/{int(mask.sum())})")

    if plot:
        subject_names = "+".join(path.stem for path in data_paths)
        cca.plot_confusion_matrix(
            cm, acc, output_path,
            title=f"FBCCA ({len(bands)} bands, {weight_mode}) - "
                  f"{subject_names} (acc = {acc * 100:.1f}%)",
        )

    print("\nDone.")
    return acc, cm


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBCCA SSVEP 解码（默认 S1+S2，4 类）"
    )
    parser.add_argument(
        "--data", nargs="+",
        default=[str(p) for p in cca.DATA_PATHS],
        help="一个或多个 .mat 文件；默认 data/S1.mat data/S2.mat",
    )
    parser.add_argument(
        "--bands", type=int, choices=[3, 5], default=5,
        help="子带数量：5=标准 FBCCA，3=砍掉最高频子带的消融",
    )
    parser.add_argument(
        "--weight-mode", choices=["standard", "low"], default="standard",
        help="standard=w_h=h^-1.25+0.25；low=更强低频权重",
    )
    parser.add_argument(
        "--output", default=None,
        help="混淆矩阵输出路径；默认按 bands/weight-mode 自动命名",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="只打印准确率，不生成 PNG",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    paths = [Path(p) for p in args.data]
    if args.output:
        output = Path(args.output)
    elif args.bands == 5 and args.weight_mode == "standard":
        output = cca.ROOT / "docs" / "step_0.analysis" / \
            "fbcca_confusion_matrix.png"
    else:
        output = cca.ROOT / "docs" / "step_0.analysis" / \
            f"fbcca_confusion_matrix_{args.bands}bands_{args.weight_mode}.png"

    bands = FBCCA_BANDS if args.bands == 5 else FBCCA_BANDS_3
    evaluate_fbcca(
        paths,
        output,
        plot=not args.no_plot,
        bands=bands,
        weight_mode=args.weight_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
