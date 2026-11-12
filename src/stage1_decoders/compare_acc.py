# -*- coding: utf-8 -*-
"""
compare_acc.py
==============
V2.0 Stage 1：CCA vs FBCCA 的准确率 / 耗时对比。

对比四组：
    CCA
    FBCCA（5 子带，标准权重）
    FBCCA（3 子带，去掉最高频）
    FBCCA（5 子带，low-weight）

所有方法都使用同一批 S1 + S2 trial（4 类 × 2 被试 × 6 block = 48 trial），
但预处理路径不同：

    CCA   : 50 Hz notch + 6-16 Hz band-pass -> Epoch -> 单子带 CCA
    FBCCA : 50 Hz notch -> 子带 Raw 滤波 -> 各自 Epoch -> 加权融合

耗时只统计“预测函数”本身（不含数据加载、MNE 预处理、子带滤波），
用 time.perf_counter() 计时，最后除以 48 得到单 trial 平均预测耗时。

运行：
    python src/stage1_decoders/compare_acc.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import cca_decoder as cca
    from . import fbcca_decoder as fbcca
except ImportError:  # 直接作为脚本运行
    import cca_decoder as cca
    import fbcca_decoder as fbcca


def prepare_data(data_paths):
    """
    一次性准备两套输入：

    - cca_epoch_data : (48, 8, 376)，6-16 Hz 预处理后的 CCA Epoch
    - fb_subband_data: 5 × (48, 8, 376)，宽频 Raw 的 5 个子带 Epoch
    - y_true        : 48 个真实标签
    """
    cca_epochs = []
    fb_subjects = []
    y_all = []
    subject_ids = []

    for path in data_paths:
        print("\n" + "-" * 72)
        print(f"Subject file: {path.name}")
        print("-" * 72)

        data = cca.load_mat(path, var="data")
        trials, y = cca.extract_trials(data, cca.LABELS)

        # CCA：沿用 V1.0 的 6-16 Hz 预处理
        raw_cca = cca.build_raw(trials, y)
        _epochs_cca, epoch_data = cca.build_epochs(
            raw_cca, y, n_samples=trials.shape[-1], verbose=False
        )
        cca_epochs.append(epoch_data)

        # FBCCA：宽频 Raw -> 5 个子带 Epoch
        raw_wide = fbcca.build_wideband_raw(trials, y)
        fb_subjects.append(
            fbcca.build_subband_epochs(
                raw_wide, y, n_samples=trials.shape[-1]
            )
        )
        y_all.append(y)
        subject_ids.append(np.full(len(y), path.stem))

    cca_data = np.concatenate(cca_epochs, axis=0)
    n_bands = len(fbcca.FBCCA_BANDS)
    fb_data = [
        np.concatenate([subject[b] for subject in fb_subjects], axis=0)
        for b in range(n_bands)
    ]
    y_true = np.concatenate(y_all)
    subject_ids = np.concatenate(subject_ids)
    return cca_data, fb_data, y_true, subject_ids


def warmup(predict_fn, data, n_trials: int = 8, **kwargs):
    """
    预测前先跑一小批 warm-up。

    为什么需要：
    - 第一次调用 sklearn CCA 会触发 BLAS/线程池和 Python 导入开销；
    - 如果只测第一次完整预测，CCA 会被“首帧开销”拖慢，
      与已经预热过的 FBCCA 比较不公平。
    warm-up 不计入最终表格。
    """
    n = min(max(int(n_trials), 1), len(data) if isinstance(data, np.ndarray) else len(data[0]))
    if isinstance(data, list):
        small = [band[:n] for band in data]
    else:
        small = data[:n]
    predict_fn(small, **kwargs)


def run_timed(predict_fn, data, **kwargs):
    """用 perf_counter 测一次完整预测，返回 preds、耗时秒数。"""
    t0 = time.perf_counter()
    preds, _scores = predict_fn(data, **kwargs)
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def print_table(rows, n_trials: int):
    """打印 Markdown 风格对比表。"""
    print("\n" + "=" * 78)
    print("CCA vs FBCCA (S1 + S2, 48 trials)")
    print("=" * 78)
    print("| Method | Accuracy | Correct/Total | Avg prediction/trial | Total prediction |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['name']} | {row['acc'] * 100:.2f}% | "
              f"{row['ok']}/{n_trials} | "
              f"{row['per_trial_ms']:.2f} ms | {row['total_s']:.4f} s |")
    print("=" * 78)

    base = rows[0]["total_s"] if rows and rows[0]["total_s"] else float("nan")
    for row in rows[1:]:
        ratio = row["total_s"] / base if base == base else float("nan")
        print(f"{row['name']} / CCA prediction-time ratio: {ratio:.2f}×")
    print("注：耗时只包含预测函数本身，不含数据加载/MNE 预处理/子带滤波。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCA vs FBCCA 准确率和耗时对比（默认 S1+S2）"
    )
    parser.add_argument(
        "--data", nargs="+",
        default=[str(p) for p in cca.DATA_PATHS],
        help="一个或多个 .mat 文件；默认 data/S1.mat data/S2.mat",
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

    print("=" * 78)
    print("Stage 1 comparison: CCA vs FBCCA")
    print("=" * 78)
    print(f"Data files : {[str(p) for p in paths]}")
    print(f"Labels     : {cca.LABELS}")
    print(f"FBCCA bands: {fbcca.FBCCA_BANDS}")
    print(f"FBCCA 3-band: {fbcca.FBCCA_BANDS_3}")
    print(f"FBCCA w_h  : {[round(w, 4) for w in fbcca.FBCCA_WEIGHTS]}")

    cca_data, fb_data, y_true, subject_ids = prepare_data(paths)
    n_trials = len(y_true)
    if n_trials != 48:
        print(f"[warn] 预期 48 trial，实际 {n_trials}；仍继续。")
    print(f"\n[data] CCA epochs={cca_data.shape}, "
          f"FBCCA sub-bands={len(fb_data)} × {fb_data[0].shape}")

    fb_data_3 = fb_data[:len(fbcca.FBCCA_BANDS_3)]
    methods = [
        ("CCA", cca.predict_epochs, cca_data, {}),
        ("FBCCA (5 bands, standard)", fbcca.predict_fbcca, fb_data,
         {"weights": fbcca.weights_for(5, "standard")}),
        ("FBCCA (3 bands)", fbcca.predict_fbcca, fb_data_3,
         {"weights": fbcca.weights_for(3, "standard")}),
        ("FBCCA (5 bands, low-weight)", fbcca.predict_fbcca, fb_data,
         {"weights": fbcca.weights_for(5, "low")}),
    ]

    # 先 warm-up，排除 sklearn/BLAS 首次调用开销；warm-up 不计入表格。
    print("[timing] warm-up (8 trials per method) ...")
    for _name, fn, data, kwargs in methods:
        warmup(fn, data, **kwargs)

    rows = []
    predictions = {}
    for name, fn, data, kwargs in methods:
        preds, elapsed = run_timed(fn, data, **kwargs)
        predictions[name] = preds
        ok = int(np.sum(preds == y_true))
        rows.append({
            "name": name,
            "acc": float(np.mean(preds == y_true)),
            "ok": ok,
            "total_s": elapsed,
            "per_trial_ms": elapsed / n_trials * 1000.0,
        })

    print_table(rows, n_trials)

    print("\nPer-subject accuracy:")
    print("| Method | S1 | S2 |")
    print("|---|---:|---:|")
    for name, preds in predictions.items():
        cells = []
        for sid in dict.fromkeys(subject_ids):
            mask = subject_ids == sid
            acc = float(np.mean(preds[mask] == y_true[mask]))
            cells.append(f"{acc * 100:.2f}%")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
