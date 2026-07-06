# -*- coding: utf-8 -*-
"""
S05: Run frozen commercial model on ALL splits, collect errors.

Output: {artifact_dir}/commercial_results_{train,valid,test}.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time
import numpy as np
import pandas as pd

from s01_model import OldLivenessModel, CommercialStage1Gate, extract_8_commercial_features, advance_stage1_gate
from s01_model import commercial_model_manifest
from s01_model import FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC, STAGE1_FS, STAGE1_PRIMITIVE_SEC, STAGE1_GATE_K
from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample, downsample_to_5hz, validate_h5_file
from s04_data import load_splits, multiprocessing_context_from_env, resolve_n_workers

SKIP_INITIAL = 3
MIN_AUTO_PARALLEL_SAMPLES = 32


def _format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _print_progress(split_name, done, total, start_time, rows_count):
    if total <= 0:
        return
    elapsed = max(1e-9, time.time() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = 100.0 * done / total
    print(
        f"[{split_name}] {done}/{total} ({pct:5.1f}%) "
        f"speed={rate:.2f} samples/s eta={_format_duration(eta)} rows={rows_count}",
        flush=True,
    )


def _progress_interval(total):
    if total <= 20:
        return 1
    return max(1, total // 20)


def _safe_rate(count, elapsed):
    return round(float(count) / float(elapsed), 6) if elapsed > 0 else None


def _write_commercial_manifest(artifact_dir):
    with open(os.path.join(artifact_dir, "commercial_model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(commercial_model_manifest(), f, indent=2, ensure_ascii=False)


def _print_timing_summary(rows):
    print("\n[S05 TIMING] split summary")
    for row in rows:
        print(
            f"  {row['split']:<5} samples={row['samples']:>5} rows={row['rows']:>6} "
            f"errors={row['errors']:>6} elapsed={row['elapsed_sec']:>7.1f}s "
            f"speed={row['samples_per_sec'] or 0:.2f} samples/s"
        )


def _cascade_sample_worker(args):
    idx, sample, dc_threshold = args
    return idx, run_sample(sample, OldLivenessModel(), dc_threshold)


def _resolve_s05_workers(n_workers, total):
    if n_workers is None:
        if total < MIN_AUTO_PARALLEL_SAMPLES:
            return 1
        return resolve_n_workers(None, n_items=total)
    return resolve_n_workers(n_workers, n_items=total)


def _iter_sample_results(name, samples, model, dc_threshold, n_workers, split_t0):
    total = len(samples)
    interval = _progress_interval(total)
    n_workers = _resolve_s05_workers(n_workers, total)
    if n_workers <= 1 or total <= 1:
        for i, sample in enumerate(samples, start=1):
            yield i - 1, run_sample(sample, model, dc_threshold)
            if i == 1 or i == total or i % interval == 0:
                yield "progress", i
        return

    pool_kwargs = {"max_workers": n_workers}
    mp_ctx = multiprocessing_context_from_env()
    if mp_ctx is not None:
        pool_kwargs["mp_context"] = mp_ctx
    print(f"[{name}] parallel workers={n_workers}", flush=True)
    ordered = [None] * total
    done = 0
    with ProcessPoolExecutor(**pool_kwargs) as executor:
        futures = [
            executor.submit(_cascade_sample_worker, (idx, sample, dc_threshold))
            for idx, sample in enumerate(samples)
        ]
        for future in as_completed(futures):
            idx, rows = future.result()
            ordered[idx] = rows
            done += 1
            if done == 1 or done == total or done % interval == 0:
                current_rows = sum(len(part) for part in ordered if part is not None)
                _print_progress(name, done, total, split_t0, current_rows)
    for idx, rows in enumerate(ordered):
        yield idx, rows or []


def _run_split(name, samples, model, dc_threshold, artifact_dir, n_workers=1):
    rows = []
    split_t0 = time.time()
    print(f"[{name}] start: {len(samples)} samples", flush=True)
    for idx, result in _iter_sample_results(name, samples, model, dc_threshold, n_workers, split_t0):
        if idx == "progress":
            _print_progress(name, result, len(samples), split_t0, len(rows))
        else:
            rows.extend(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(artifact_dir, f"commercial_results_{name}.csv"), index=False)
    n_err = df[(df["fallback"] == False)]["is_error"].sum() if len(df) > 0 else 0
    elapsed = time.time() - split_t0
    print(f"[{name}] {len(df)} rows, {n_err} errors, elapsed={_format_duration(elapsed)}")
    return {
        "split": name,
        "samples": len(samples),
        "rows": len(df),
        "errors": int(n_err),
        "elapsed_sec": round(elapsed, 3),
        "samples_per_sec": _safe_rate(len(samples), elapsed),
        "rows_per_sec": _safe_rate(len(df), elapsed),
    }


def _to_25hz(sample, ppg, acc):
    if _is_25hz_sample(sample):
        return (np.asarray(ppg, dtype=np.float64),
                np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
    acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _prewindow_to_25hz(sample, window, window_sec):
    n = int(window.shape[0])
    if (_is_25hz_sample(sample) or n == int(round(float(window_sec) * FEATURE_FS))
            or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(window, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(window, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def _slice_acc(acc25, start, size):
    if acc25 is None or start >= len(acc25):
        return None
    return acc25[start:start + size]


def _stage1_pass(window, dc_threshold, ppg_src_fs):
    ir5 = downsample_to_5hz(window[:, 0], ppg_src_fs, STAGE1_FS)
    s1_win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    if len(ir5) < s1_win:
        return False
    gate = CommercialStage1Gate(dc_threshold, K=STAGE1_GATE_K)
    enabled = False
    for start in range(0, len(ir5) - s1_win + 1, s1_win):
        enabled = bool(gate.update(ir5[start:start + s1_win]))
    return enabled


def run_sample(sample, model, dc_threshold):
    base = {"sample_name": sample.get("sample_name", "unknown"), "target": int(sample.get("target", 0))}
    try:
        ppg, acc = load_ppg(sample), load_acc(sample)
        ok, err = validate_h5_file(sample["h5_file"], base["sample_name"])
        if not ok: raise ValueError(err)
    except Exception as exc:
        return [{**base, "window_idx": -1, "stage2_enabled": False, "score": None, "pred": 0,
                 "is_error": 1, "fallback": True, "fallback_reason": str(exc)}]
    results = []
    if is_prewindowed_signal(ppg):
        mode = detect_green_mode(ppg)
        for idx in range(SKIP_INITIAL, ppg.shape[0]):
            win25, ppg_src = _prewindow_to_25hz(sample, ppg[idx], COMMERCIAL_WIN_SEC)
            en = _stage1_pass(ppg[idx], dc_threshold, ppg_src)
            if not en:
                results.append({**base, "window_idx": idx, "stage2_enabled": False, "score": None,
                                "pred": 0, "is_error": int(base["target"] != 0), "fallback": False, "fallback_reason": None})
                continue
            try:
                ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
                acc_seg = None
                if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                    acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], COMMERCIAL_WIN_SEC)
                is_live, score, _, _ = model.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
            except Exception: score, is_live = None, 0
            results.append({**base, "window_idx": idx, "stage2_enabled": True, "score": score,
                            "pred": int(is_live), "is_error": int(int(is_live) != base["target"]),
                            "fallback": False, "fallback_reason": None})
        return results
    ppg25, acc25, ppg_src = _to_25hz(sample, ppg, acc)
    mode = detect_green_mode(ppg)
    ir5 = downsample_to_5hz(ppg[:, 0], ppg_src, STAGE1_FS)
    s1_win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    s2_win = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS))
    s2_stride = max(1, int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS)))
    n_s1, n_s2 = max(0, (len(ir5) - s1_win) // s1_win + 1), max(0, (len(ppg25) - s2_win) // s2_stride + 1)
    gate = CommercialStage1Gate(dc_threshold, K=STAGE1_GATE_K)
    last_s1 = -1
    for step in range(SKIP_INITIAL, n_s2):
        tgt = int(np.floor(step * s2_stride / FEATURE_FS + 1e-9))
        if tgt >= n_s1: break
        en, last_s1 = advance_stage1_gate(gate, ir5, s1_win, s1_win, last_s1, tgt)
        if not en:
            results.append({**base, "window_idx": step, "stage2_enabled": False, "score": None,
                            "pred": 0, "is_error": int(base["target"] != 0), "fallback": False, "fallback_reason": None})
            continue
        try:
            win = ppg25[step * s2_stride:step * s2_stride + s2_win, :]
            ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
            is_live, score, _, _ = model.predict_raw(
                extract_8_commercial_features(ir, amb, g1, g2, g3, _slice_acc(acc25, step * s2_stride, s2_win)))
        except Exception: score, is_live = None, 0
        results.append({**base, "window_idx": step, "stage2_enabled": True, "score": score,
                        "pred": int(is_live), "is_error": int(int(is_live) != base["target"]),
                        "fallback": False, "fallback_reason": None})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--splits_dir", default="artifacts")
    p.add_argument("--dc_threshold", type=float, default=0.3e6)
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    _write_commercial_manifest(args.artifact_dir)
    splits = load_splits(args.splits_dir)
    model = OldLivenessModel()
    t0 = time.time()
    timing_rows = []
    for name in ["train", "valid", "test"]:
        timing_rows.append(_run_split(name, splits[name], model, args.dc_threshold, args.artifact_dir, args.n_workers))
    total_elapsed = time.time() - t0
    timing_rows.append({
        "split": "total",
        "samples": int(sum(r["samples"] for r in timing_rows)),
        "rows": int(sum(r["rows"] for r in timing_rows)),
        "errors": int(sum(r["errors"] for r in timing_rows)),
        "elapsed_sec": round(total_elapsed, 3),
        "samples_per_sec": _safe_rate(sum(r["samples"] for r in timing_rows), total_elapsed),
        "rows_per_sec": _safe_rate(sum(r["rows"] for r in timing_rows), total_elapsed),
    })
    _print_timing_summary(timing_rows)
    print(f"Done ({total_elapsed:.1f}s / {_format_duration(total_elapsed)})")

if __name__ == "__main__":
    main()
