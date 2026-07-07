# -*- coding: utf-8 -*-
"""
S06: Extract new feature pool on commercial-positive guard candidates.

Output: {artifact_dir}/error_features_{train,valid,test}.csv
"""

import argparse, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import pandas as pd

from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample
from s02_features import extract_feature_pool_from_window, validate_h5_file
from s04_data import load_splits, multiprocessing_context_from_env, resolve_n_workers

FEATURE_FS = 25
MIN_AUTO_PARALLEL_GROUPS = 32


def _format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _progress_interval(total):
    if total <= 20:
        return 1
    return max(1, total // 20)


def _print_progress(split_name, done, total, start_time, ok, skip):
    elapsed = max(1e-9, time.time() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = 100.0 * done / total if total else 100.0
    print(
        f"[{split_name}] candidates {done}/{total} ({pct:5.1f}%) "
        f"speed={rate:.2f}/s eta={_format_duration(eta)} ok={ok} skipped={skip}",
        flush=True,
    )


def _resolve_s06_workers(n_workers, total):
    if n_workers is None:
        if total < MIN_AUTO_PARALLEL_GROUPS:
            return 1
        return resolve_n_workers(None, n_items=total)
    return resolve_n_workers(n_workers, n_items=total)


def build_hard_negative_summary(df, split):
    if df is None or len(df) == 0:
        return {
            "split": split,
            "candidate_source": "commercial_positive_stage2_enabled",
            "total_candidates": 0,
            "hard_negative_candidates": 0,
            "worn_positive_candidates": 0,
            "hard_negative_rate": 0.0,
            "unique_samples": 0,
            "hard_negative_samples": 0,
        }
    should_veto = pd.to_numeric(df.get("should_veto", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(df.get("target", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    total = int(len(df))
    hard = int(should_veto.sum())
    sample_col = df.get("sample_name", pd.Series(dtype=str))
    hard_samples = int(df.loc[should_veto == 1, "sample_name"].nunique()) if "sample_name" in df else 0
    return {
        "split": split,
        "candidate_source": "commercial_positive_stage2_enabled",
        "total_candidates": total,
        "hard_negative_candidates": hard,
        "worn_positive_candidates": int((target == 1).sum()),
        "hard_negative_rate": float(hard / total) if total else 0.0,
        "unique_samples": int(sample_col.nunique()) if len(sample_col) else 0,
        "hard_negative_samples": hard_samples,
    }


def write_hard_negative_audit(artifact_dir, split, df):
    out_dir = os.path.join(artifact_dir, "hard_negative_audit")
    os.makedirs(out_dir, exist_ok=True)
    summary = build_hard_negative_summary(df, split)
    summary_path = os.path.join(out_dir, f"hard_negative_summary_{split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"hard_negative_summary_{split}.csv"), index=False)
    candidate_path = os.path.join(out_dir, f"hard_negative_candidates_{split}.csv")
    if df is None or len(df) == 0:
        pd.DataFrame().to_csv(candidate_path, index=False)
    else:
        should_veto = pd.to_numeric(df.get("should_veto", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
        hard = df[should_veto == 1].copy()
        hard.to_csv(candidate_path, index=False)
    with open(os.path.join(out_dir, f"hard_negative_summary_{split}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Hard Negative Audit - {split}\n\n")
        f.write("Source: commercial-positive, Stage2-enabled candidates.\n\n")
        f.write(f"- total_candidates: {summary['total_candidates']}\n")
        f.write(f"- hard_negative_candidates: {summary['hard_negative_candidates']}\n")
        f.write(f"- hard_negative_rate: {summary['hard_negative_rate']:.6f}\n")
        f.write(f"- hard_negative_samples: {summary['hard_negative_samples']}\n")
    return summary


def write_skip_report(artifact_dir, split, rows):
    cols = ["split", "sample_name", "window_idx", "reason", "detail"]
    path = os.path.join(artifact_dir, f"skipped_error_features_{split}.csv")
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _prewindow_to_25hz(sample, window, window_sec):
    n = int(window.shape[0])
    if (_is_25hz_sample(sample) or n == int(round(float(window_sec) * FEATURE_FS))
            or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(window, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(window, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def _to_25hz(sample, ppg, acc):
    if _is_25hz_sample(sample):
        return (np.asarray(ppg, dtype=np.float64),
                np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS); acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _process_candidate_group(split, sn, records, sample):
    rows, skip_rows, ok, skip = [], [], 0, 0
    if sample is None:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "sample_not_found", "detail": ""})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}
    try:
        ppg, acc = load_ppg(sample), load_acc(sample)
    except Exception as exc:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "load_signal_failed", "detail": str(exc)})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}
    try:
        mode = detect_green_mode(ppg)
        prewindowed = is_prewindowed_signal(ppg)
        if not prewindowed:
            ppg25, _acc25, _ = _to_25hz(sample, ppg, acc)
            sw, ss = int(round(5.0 * FEATURE_FS)), int(round(1.0 * FEATURE_FS))
    except Exception as exc:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "sample_preprocess_failed", "detail": str(exc)})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}

    for row in records:
        widx = int(row["window_idx"])
        try:
            score = float(row["score"]) if pd.notna(row["score"]) else -2000.0
            if prewindowed:
                if widx >= ppg.shape[0]:
                    skip += 1
                    skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                                      "reason": "window_index_out_of_bounds", "detail": f"n_windows={ppg.shape[0]}"})
                    continue
                win25, _ = _prewindow_to_25hz(sample, ppg[widx], 5.0)
                ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
            else:
                if widx * ss + sw > len(ppg25):
                    skip += 1
                    skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                                      "reason": "window_slice_out_of_range", "detail": f"signal_len={len(ppg25)}"})
                    continue
                win = ppg25[widx * ss:widx * ss + sw, :]
                ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
            feats = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
            target = int(row["target"])
            out = {"sample_name": sn, "target": target, "should_veto": int(target == 0),
                   "commercial_pred": int(row["pred"]), "window_idx": widx,
                   "commercial_score": score, "is_error": int(row["is_error"])}
            out.update(feats)
            rows.append(out)
            ok += 1
        except Exception as exc:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                              "reason": "feature_extraction_failed", "detail": str(exc)})
    return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}


def _s06_group_worker(args):
    return _process_candidate_group(*args)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--splits_dir", default="artifacts")
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    splits = load_splits(args.splits_dir)
    sn_map = {}; [sn_map.update({s["sample_name"]: s}) for part in ["train","valid","test"] for s in splits[part]]
    t0 = time.time()
    for name in ["train", "valid", "test"]:
        cp = os.path.join(args.artifact_dir, f"commercial_results_{name}.csv")
        if not os.path.exists(cp): print(f"[{name}] Skipped"); continue
        comm = pd.read_csv(cp)
        errors = comm[(comm["pred"] == 1) & (comm["fallback"] == False) & (comm["stage2_enabled"] == True)]
        if len(errors) == 0:
            print(f"[{name}] No commercial-positive candidates")
            write_skip_report(args.artifact_dir, name, [])
            write_hard_negative_audit(args.artifact_dir, name, pd.DataFrame())
            continue
        rows, skip_rows, ok, skip = [], [], 0, 0
        total_candidates = len(errors)
        interval = _progress_interval(total_candidates)
        split_t0 = time.time()
        processed = 0
        groups = [
            (sn, group.to_dict("records"), sn_map.get(sn))
            for sn, group in errors.groupby("sample_name", sort=False)
        ]
        n_workers = _resolve_s06_workers(args.n_workers, len(groups))
        print(f"[{name}] candidates={total_candidates}, samples={len(groups)}, workers={n_workers}", flush=True)
        if n_workers > 1 and len(groups) > 1:
            pool_kwargs = {"max_workers": n_workers}
            mp_ctx = multiprocessing_context_from_env()
            if mp_ctx is not None:
                pool_kwargs["mp_context"] = mp_ctx
            with ProcessPoolExecutor(**pool_kwargs) as executor:
                futures = [
                    executor.submit(_s06_group_worker, (name, sn, records, sample))
                    for sn, records, sample in groups
                ]
                for future in as_completed(futures):
                    result = future.result()
                    rows.extend(result["rows"])
                    skip_rows.extend(result["skip_rows"])
                    ok += int(result["ok"])
                    skip += int(result["skip"])
                    processed += int(result["processed"])
                    if processed == total_candidates or processed % interval == 0 or processed - int(result["processed"]) == 0:
                        _print_progress(name, processed, total_candidates, split_t0, ok, skip)
        else:
            for sn, records, sample in groups:
                result = _process_candidate_group(name, sn, records, sample)
                rows.extend(result["rows"])
                skip_rows.extend(result["skip_rows"])
                ok += int(result["ok"])
                skip += int(result["skip"])
                processed += int(result["processed"])
                if processed == total_candidates or processed % interval == 0 or processed - int(result["processed"]) == 0:
                    _print_progress(name, processed, total_candidates, split_t0, ok, skip)
        print(f"[{name}] Extracted={ok} skipped={skip}")
        df = pd.DataFrame(rows)
        if rows:
            df.to_csv(os.path.join(args.artifact_dir, f"error_features_{name}.csv"), index=False)
        write_skip_report(args.artifact_dir, name, skip_rows)
        write_hard_negative_audit(args.artifact_dir, name, df)
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
