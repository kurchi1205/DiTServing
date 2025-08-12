import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Any

import pandas as pd


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_cache_suffix(k: str) -> str:
    # "p2_prompt_123_cache_5" -> "p2_prompt_123"
    return k.split("_cache_")[0] if "_cache_" in k else k


def choose_interval_key(nested: Dict[str, Any], preferred: str | None) -> str:
    """
    Choose which interval key (e.g. "6") to use across prompts.
    If `preferred` exists in >= 70% of items, use it; otherwise pick the most common key (ties -> largest int).
    """
    if not nested:
        raise ValueError("Empty nested dict; cannot choose interval key.")
    total = len(nested)
    if preferred is not None:
        have = sum(1 for v in nested.values() if isinstance(v, dict) and preferred in v)
        if total and have / total >= 0.70:
            return preferred

    counts = Counter()
    for v in nested.values():
        if isinstance(v, dict):
            counts.update(v.keys())
    if not counts:
        raise ValueError("No interval keys found in nested metrics.")

    def key_sort(item):
        k, cnt = item
        kval = int(k) if str(k).isdigit() else -1
        return (cnt, kval)

    return sorted(counts.items(), key=key_sort, reverse=True)[0][0]


def extract_interval(nested: Dict[str, Any], key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p, per in nested.items():
        if isinstance(per, dict) and key in per:
            try:
                out[p] = float(per[key])
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Find failures for cache interval 5 with CLIP-weighted rank score.")
    ap.add_argument("--clip_path", required=True, help="Path to CLIP JSON (keys like p2_prompt_X_cache_5).")
    ap.add_argument("--fid_path", required=True, help="Path to FID JSON (nested intervals).")
    ap.add_argument("--ssim_path", required=True, help="Path to SSIM JSON (nested intervals).")
    # Which interval inside FID/SSIM to use as 'cache=5'. Defaults are guesses; override if needed.
    ap.add_argument("--fid_interval_key", default="6", help='Interval key for FID (e.g. "6").')
    ap.add_argument("--ssim_interval_key", default="6", help='Interval key for SSIM (e.g. "6").')
    # Scoring knobs
    ap.add_argument("--quantile", type=float, default=0.10, help="Tail quantile (default 0.10).")
    ap.add_argument("--w_clip", type=float, default=3.0, help="Weight for CLIP badness (default 3.0).")
    ap.add_argument("--w_fid", type=float, default=1.0, help="Weight for FID badness (default 1.0).")
    ap.add_argument("--w_ssim", type=float, default=1.0, help="Weight for SSIM badness (default 1.0).")
    ap.add_argument("--gamma", type=float, default=1.5, help="Tail emphasis exponent >=1 (default 1.5).")
    ap.add_argument("--topN", type=int, default=50, help="Top-N worst to keep (default 50).")
    ap.add_argument("--out_json", default="failures_cache5.json", help="Output JSON path.")
    ap.add_argument("--stdout", action="store_true", help="Also print the JSON payload to stdout.")
    args = ap.parse_args()

    # ---- Load JSONs ----
    clip_raw = load_json(args.clip_path)
    fid_all = load_json(args.fid_path)
    ssim_all = load_json(args.ssim_path)

    # ---- CLIP: already cache_5 ----
    clip_map = {strip_cache_suffix(k): float(v) for k, v in clip_raw.items()}

    # ---- Pick interval keys for FID/SSIM ----
    fid_key = choose_interval_key(fid_all, args.fid_interval_key)
    ssim_key = choose_interval_key(ssim_all, args.ssim_interval_key)

    fid_k = extract_interval(fid_all, fid_key)
    ssim_k = extract_interval(ssim_all, ssim_key)

    # ---- Merge by prompt ----
    df = (
        pd.DataFrame({"prompt": list(clip_map.keys()), "clip": list(clip_map.values())})
        .merge(pd.DataFrame({"prompt": list(fid_k.keys()), "fid": list(fid_k.values())}),
               on="prompt", how="inner")
        .merge(pd.DataFrame({"prompt": list(ssim_k.keys()), "ssim": list(ssim_k.values())}),
               on="prompt", how="inner")
        .dropna(subset=["clip", "fid", "ssim"])
        .reset_index(drop=True)
    )
    if df.empty:
        raise SystemExit("No overlapping prompts across CLIP/FID/SSIM after filtering.")

    # ---- Rank-percentile badness (monotonic & robust) ----
    # Higher clip -> better -> LOWER badness; higher fid -> worse -> HIGHER badness; higher ssim -> better -> LOWER badness
    clip_rank = df["clip"].rank(pct=True, method="average")
    fid_rank = df["fid"].rank(pct=True, method="average")
    ssim_rank = df["ssim"].rank(pct=True, method="average")

    eps = 1e-9
    df["clip_bad"] = (1.0 - clip_rank).clip(eps, 1 - eps)
    df["fid_bad"] = fid_rank.clip(eps, 1 - eps)
    df["ssim_bad"] = (1.0 - ssim_rank).clip(eps, 1 - eps)

    g = max(1.0, float(args.gamma))
    df["clip_bad_g"] = df["clip_bad"] ** g
    df["fid_bad_g"] = df["fid_bad"] ** g
    df["ssim_bad_g"] = df["ssim_bad"] ** g

    w_clip, w_fid, w_ssim = float(args.w_clip), float(args.w_fid), float(args.w_ssim)
    w_sum = w_clip + w_fid + w_ssim
    if w_sum <= 0:
        raise ValueError("At least one of the weights must be positive.")

    df["failure_score"] = (w_clip * df["clip_bad_g"] +
                           w_fid * df["fid_bad_g"] +
                           w_ssim * df["ssim_bad_g"]) / w_sum

    # ---- Tail flags (simple rules) ----
    q = float(args.quantile)
    df["clip_tail_fail"] = df["clip"] <= df["clip"].quantile(q)
    df["fid_tail_fail"] = df["fid"] >= df["fid"].quantile(1 - q)
    df["ssim_tail_fail"] = df["ssim"] <= df["ssim"].quantile(q)
    df["is_fail_tail"] = df["clip_tail_fail"] | df["fid_tail_fail"] | df["ssim_tail_fail"]

    # ---- Assemble outputs ----
    worst_overall = df.sort_values("failure_score", ascending=False).head(args.topN)
    tail_union = df[df["is_fail_tail"]].sort_values("failure_score", ascending=False)
    tail_intersection = df[df["clip_tail_fail"] & df["fid_tail_fail"] & df["ssim_tail_fail"]] \
        .sort_values("failure_score", ascending=False)

    def rows(dfi: pd.DataFrame):
        cols = [
            "prompt", "clip", "fid", "ssim",
            "clip_bad", "fid_bad", "ssim_bad",
            "clip_bad_g", "fid_bad_g", "ssim_bad_g",
            "failure_score",
            "clip_tail_fail", "fid_tail_fail", "ssim_tail_fail", "is_fail_tail"
        ]
        return dfi[cols].to_dict(orient="records")

    payload = {
        "config": {
            "fid_interval_key_used": fid_key,
            "ssim_interval_key_used": ssim_key,
            "quantile": q,
            "weights": {"clip": w_clip, "fid": w_fid, "ssim": w_ssim},
            "gamma": g,
            "topN": int(args.topN)
        },
        "counts": {
            "total_joined": int(len(df)),
            "tail_union": int(len(tail_union)),
            "tail_intersection": int(len(tail_intersection))
        },
        "lists": {
            "worst_overall": rows(worst_overall),
            "tail_union": rows(tail_union),
            "tail_intersection": rows(tail_intersection)
        }
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[written] {args.out_json}")
    if args.stdout:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())