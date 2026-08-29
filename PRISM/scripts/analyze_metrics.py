#!/usr/bin/env python3
"""
Standalone analysis tool for PRISM per-update metrics.jsonl files.

Usage:
    python scripts/analyze_metrics.py runs/dpmorl_only/policy_0_metrics.jsonl
    python scripts/analyze_metrics.py runs/dpmorl_only/policy_*.jsonl   # multi-policy comparison

Stdlib only -- no prism/torch/CaRL/nuPlan/numpy/pandas dependency, so this
runs anywhere (lab machine or laptop) directly against the raw log file,
without needing the training environment installed. Point of this script:
run it, share only the printed report, not the raw multi-MB log.

Handles, without crashing:
  - a file containing multiple runs (training was restarted and the
    MetricsLogger, which opens in append mode, kept writing to the same
    file) -- split into separate reports at each "config" record boundary.
  - a truncated final line (the file was copied while still being written).
  - older log files missing newer fields (z_comfort, frac_collision, v_loss,
    etc.) -- prints "n/a" for whatever isn't present instead of failing.

What it reports, per run:
  - config (the ablation toggles that produced it) and wall-clock pace.
  - stability: NaN/Inf flags, slow-update outliers (after the known
    2-update warmup).
  - binned trend (default 10 bins) for the cost/CVaR side (mu_c, cvar_hat,
    cost_critic_loss), the "is it learning" side (v_loss, entropy,
    mean_reward_this_update), and episode outcomes (collision / off_road /
    completed fractions, episode length) -- so a real trend is visible
    instead of two noisy single-update endpoints.
  - grad_norm_cost_critic's share of the total pre-clip gradient norm, to
    check whether the cost critic is dominating the shared clip.
  - a handful of directional flags (early-half vs late-half comparison) --
    these are pointers to look at, not an automated verdict; read the
    binned tables before trusting them.

When multiple files are given, also prints a cross-policy z_T comparison
(mean z_T per style dimension, early vs late) to check whether each policy
is trending higher on its own preferred dimension than the others --
assumes the standard preference ordering (policy 0=comfort, 1=progress,
2=lateral, 3=spacing, for whatever K <= 4 was used) from scripts/train.py's
_get_preference_vectors(); if your run used a different K or preference
assignment, read the raw
z_comfort/z_progress/z_lateral/z_spacing columns yourself instead of
trusting the "own dimension" highlighting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_STANDARD_K4_DIMS = ["comfort", "progress", "lateral", "spacing"]


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

def load_runs(path: Path):
    """Parse one metrics.jsonl file into a list of runs (split at each
    'config' record). Returns [{"config": dict, "updates": [dict, ...]}]."""
    runs = []
    cur = None
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if r.get("record_type") == "config":
                cur = {"config": r, "updates": []}
                runs.append(cur)
            elif cur is not None:
                cur["updates"].append(r)
    if n_bad:
        print(f"  (skipped {n_bad} truncated/unparseable line(s) -- file was likely "
              f"copied while still being written)")
    return runs


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_pct(x):
    return "n/a" if x is None else f"{100*x:.0f}%"


def bin_updates(ups, n_bins=10):
    n = len(ups)
    if n == 0:
        return []
    bin_size = max(n // n_bins, 1)
    return [ups[b:b + bin_size] for b in range(0, n, bin_size) if ups[b:b + bin_size]]


def half_compare(ups, key, pct=False):
    """Mean of key over the first half vs second half of ups. Returns
    (early, late) or (None, None) if the field is never present."""
    n = len(ups)
    if n < 4:
        return None, None
    early = mean(u.get(key) for u in ups[:n // 2])
    late = mean(u.get(key) for u in ups[n // 2:])
    return early, late


# ----------------------------------------------------------------------
# Per-run report
# ----------------------------------------------------------------------

def report_run(label, cfg, ups, n_bins=10):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    if not ups:
        print("  no update records -- nothing to analyze")
        return None

    n = len(ups)
    span = ups[-1]["wall_time"] - ups[0]["wall_time"]
    print(f"config: beta={cfg.get('beta')} cf_coef={cfg.get('cf_coef')} "
          f"fixed_alpha={cfg.get('fixed_alpha')} outcome_costs_enabled={cfg.get('outcome_costs_enabled')} "
          f"active_indicators={cfg.get('active_indicators')} cost_scale={cfg.get('cost_scale')}")
    print(f"updates: {n}  (#{ups[0]['update']} -> #{ups[-1]['update']})   "
          f"span: {span/3600:.2f}h  ({span/max(n-1,1):.2f} s/update avg)")

    # -- stability --
    n_nan = sum(1 for u in ups if u.get("nan_detected") or u.get("dense_cost_nan"))
    max_streak = max((u.get("nan_streak", 0) for u in ups), default=0)
    times = [u["update_total_time_s"] for u in ups if u.get("update_total_time_s") is not None]
    slow = [u for u in ups[2:] if (u.get("update_total_time_s") or 0) > 30]
    print(f"\n-- stability --")
    print(f"  NaN/Inf flagged updates: {n_nan}/{n}  (max consecutive streak: {max_streak})")
    if times:
        print(f"  update_total_time_s: mean={fmt(mean(times))} max={fmt(max(times))}  "
              f"updates >30s (excluding the first 2, known warmup): {len(slow)}")

    # -- binned trend --
    bins = bin_updates(ups, n_bins)
    print(f"\n-- binned trend ({len(bins)} bins) --")
    print(f"{'range':>15} | {'mu_c':>6} {'cvar':>6} {'cc_loss':>8} | "
          f"{'v_loss':>7} {'entropy':>7} {'reward':>8} | "
          f"{'n_eps':>5} {'ep_len':>7} {'coll%':>6} {'road%':>6} {'done%':>6} | {'gn_cc%':>6}")
    for chunk in bins:
        lo, hi = chunk[0]["update"], chunk[-1]["update"]
        gnc = mean(u.get("grad_norm_cost_critic") for u in chunk)
        gno = mean(u.get("grad_norm_other") for u in chunk)
        gn_frac = gnc / (gnc + gno) if (gnc is not None and gno is not None and (gnc + gno) > 0) else None
        print(
            f"{f'{lo}-{hi}':>15} | "
            f"{fmt(mean(u.get('mu_c') for u in chunk)):>6} "
            f"{fmt(mean(u.get('cvar_hat') for u in chunk)):>6} "
            f"{fmt(mean(u.get('cost_critic_loss') for u in chunk), 4):>8} | "
            f"{fmt(mean(u.get('v_loss') for u in chunk), 4):>7} "
            f"{fmt(mean(u.get('entropy') for u in chunk)):>7} "
            f"{fmt(mean(u.get('mean_reward_this_update') for u in chunk)):>8} | "
            f"{fmt(mean(u.get('n_episodes_this_update') for u in chunk), 1):>5} "
            f"{fmt(mean(u.get('mean_episode_length') for u in chunk), 1):>7} "
            f"{fmt_pct(mean(u.get('frac_collision') for u in chunk)):>6} "
            f"{fmt_pct(mean(u.get('frac_off_road') for u in chunk)):>6} "
            f"{fmt_pct(mean(u.get('frac_completed') for u in chunk)):>6} | "
            f"{fmt_pct(gn_frac):>6}"
        )

    # -- actor health (if present) --
    if any("approx_kl" in u for u in ups):
        print(f"\n-- actor health ({len(bins)} bins) -- ppo_loss: raw actor surrogate loss "
              f"(split out of total_loss). approx_kl: how much the policy moved this update "
              f"(near 0 = barely changing, growing over time can mean instability). "
              f"clip_fraction: share of minibatch samples hitting the PPO clip boundary "
              f"(rule of thumb: healthy is roughly 0.1-0.3; near 0 the whole run can mean "
              f"steps are too small to matter, consistently >0.5 can mean too-large/unstable updates). "
              f"reward_advantage_std: spread of the RAW (pre-normalization) reward advantage -- "
              f"this is what actually drives the actor's gradient; near 0 means little signal "
              f"telling the actor which actions are better, regardless of what mean reward is doing.")
        print(f"{'range':>15} | {'ppo_loss':>9} {'approx_kl':>10} {'clip_frac':>10} "
              f"{'entropy':>8} {'adv_std':>9} {'rew_std':>9}")
        for chunk in bins:
            lo, hi = chunk[0]["update"], chunk[-1]["update"]
            print(
                f"{f'{lo}-{hi}':>15} | "
                f"{fmt(mean(u.get('ppo_loss') for u in chunk), 5):>9} "
                f"{fmt(mean(u.get('approx_kl') for u in chunk), 5):>10} "
                f"{fmt(mean(u.get('clip_fraction') for u in chunk)):>10} "
                f"{fmt(mean(u.get('entropy') for u in chunk)):>8} "
                f"{fmt(mean(u.get('reward_advantage_std') for u in chunk), 5):>9} "
                f"{fmt(mean(u.get('std_reward_this_update') for u in chunk), 5):>9}"
            )

    # -- z_T style-dimension trend (if present) --
    if any("z_comfort" in u for u in ups):
        print(f"\n-- mean z_T by style dimension (early half vs late half) --")
        for dim in _STANDARD_K4_DIMS:
            key = f"z_{dim}"
            early, late = half_compare(ups, key)
            if early is not None:
                arrow = "up" if (late or 0) > (early or 0) else "down"
                print(f"  z_{dim:<10} early={fmt(early)}  late={fmt(late)}  ({arrow})")

    # -- directional flags (pointers, not a verdict) --
    print(f"\n-- directional flags (early half vs late half; read the binned table above, "
          f"this is a pointer not a verdict) --")
    for key, nd in [("cost_critic_loss", 4), ("cvar_hat", 3), ("v_loss", 4),
                     ("entropy", 3), ("approx_kl", 5), ("clip_fraction", 3),
                     ("mean_reward_this_update", 4), ("std_reward_this_update", 4),
                     ("reward_advantage_std", 5), ("frac_completed", 3),
                     ("current_lr", 6)]:
        early, late = half_compare(ups, key)
        if early is None:
            continue
        direction = "↑" if late > early * 1.05 else ("↓" if late < early * 0.95 else "≈flat")
        print(f"  {key:<28} early={fmt(early, nd)}  late={fmt(late, nd)}  {direction}")

    return {"cfg": cfg, "ups": ups}


# ----------------------------------------------------------------------
# Cross-policy style-divergence comparison
# ----------------------------------------------------------------------

def report_cross_policy(labeled_runs):
    if len(labeled_runs) < 2:
        return
    print(f"\n{'='*78}\nCross-policy z_T comparison (assumes standard ordering: "
          f"0=comfort,1=progress,2=lateral,3=spacing)\n{'='*78}")
    have_z = [(label, r) for label, r in labeled_runs if r and any("z_comfort" in u for u in r["ups"])]
    if not have_z:
        print("  no z_T fields present in any file (older log, or run predates this logging) -- skip")
        return
    print(f"{'policy':>10} | " + " | ".join(f"{d:>10}" for d in _STANDARD_K4_DIMS) + " | own-dim highest late?")
    for label, r in have_z:
        ups = r["ups"]
        late_means = {}
        for dim in _STANDARD_K4_DIMS:
            _, late = half_compare(ups, f"z_{dim}")
            late_means[dim] = late
        row = f"{label:>10} | " + " | ".join(fmt(late_means[d]).rjust(10) for d in _STANDARD_K4_DIMS)
        print(row)
    print("\n  Read this as: does each policy's row peak on a DIFFERENT column than the others?")
    print("  If all rows look similar, styles are not diverging -- personalization isn't working yet.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="one or more policy_*_metrics.jsonl files")
    parser.add_argument("--bins", type=int, default=10, help="number of bins for the trend table (default 10)")
    parser.add_argument("--out", type=Path, default=None, help="also write the report to this file")
    args = parser.parse_args()

    out_lines = []
    if args.out:
        import io
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf

    labeled_runs = []
    for path in args.files:
        if not path.exists():
            print(f"!! {path} does not exist -- skipping")
            continue
        runs = load_runs(path)
        if not runs:
            print(f"!! {path}: no parseable records")
            continue
        for i, run in enumerate(runs):
            label = f"{path.name}" + (f"  [run {i}]" if len(runs) > 1 else "")
            result = report_run(label, run["config"], run["updates"], n_bins=args.bins)
            # Only the LAST run in a file is used for cross-policy comparison
            # (earlier runs in the same file are aborted/superseded attempts).
            if i == len(runs) - 1:
                labeled_runs.append((path.stem, result))

    report_cross_policy(labeled_runs)

    if args.out:
        sys.stdout = real_stdout
        text = buf.getvalue()
        print(text)
        args.out.write_text(text)
        print(f"\n(report also written to {args.out})")


if __name__ == "__main__":
    main()
