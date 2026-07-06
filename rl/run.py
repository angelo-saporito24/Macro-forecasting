"""Driver: train agents and evaluate the calibrated-vs-overconfident comparison.

Examples
--------
    # Fast end-to-end smoke run (tiny budget, proves the pipeline works)
    python -m rl.run --smoke

    # Reduced but real PPO comparison, Tier-1 beliefs, one seed
    python -m rl.run --algos ppo --belief tier1 --seeds 42 \
        --steps-ppo 150000 --eval-episodes 20

    # Full reproduction (as in the report; heavy — use a GPU/Colab)
    python -m rl.run --algos both --belief both --seeds 42 123 7

Results (per-seed metrics + across-seed comparison tables) are written to the
output directory as JSON and CSV.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl.train import (
    CONDITIONS, TRAINABLE, PPO_TOTAL_STEPS, SAC_TOTAL_STEPS, SEEDS, EP_LEN,
    train_agent,
)
from rl.evaluate import evaluate_condition, results_table
from stable_baselines3 import PPO, SAC

LABELS = {"taylor_rule": "Taylor rule", "full_obs_rl": "Full obs",
          "pomdp_calibrated": "Calibrated", "pomdp_overconfident": "Overconfident"}
_ALGO_CLS = {"ppo": PPO, "sac": SAC}


def _load_model(algo, condition, belief_mode, seed, out_dir):
    path = Path(out_dir) / f"{algo}_{belief_mode}_{condition}_seed{seed}" / "model.zip"
    return _ALGO_CLS[algo].load(str(path), device="cpu")


def run_matrix(algos, belief_modes, seeds, steps_ppo, steps_sac,
               eval_episodes, ep_len, out_dir, overwrite):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_records = []
    tables = {}

    for algo in algos:
        # SAC has no full-observability / Taylor training targets beyond POMDP,
        # but we evaluate all conditions for both algos for a common table.
        total_steps = steps_ppo if algo == "ppo" else steps_sac
        for belief_mode in belief_modes:
            print(f"\n{'='*64}\n  {algo.upper()}  |  belief={belief_mode}\n{'='*64}")

            # --- train ---
            for seed in seeds:
                for cond in TRAINABLE:
                    print(f"  train: {cond:22s} seed={seed} steps={total_steps}")
                    train_agent(algo, cond, belief_mode, total_steps, seed,
                                out_dir, overwrite=overwrite)

            # --- evaluate ---
            agg = defaultdict(lambda: defaultdict(list))
            for seed in seeds:
                for cond in CONDITIONS:
                    model = None
                    if cond in TRAINABLE:
                        model = _load_model(algo, cond, belief_mode, seed, out_dir)
                    metrics = evaluate_condition(
                        model, cond, belief_mode, eval_episodes, seed, ep_len)
                    metrics.update(algo=algo, belief_mode=belief_mode,
                                   condition=cond, seed=seed)
                    per_seed_records.append(metrics)
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            agg[cond][k].append(v)

            # --- across-seed means ---
            rows = {}
            for cond in CONDITIONS:
                rows[LABELS[cond]] = {k: float(np.mean(v))
                                      for k, v in agg[cond].items()
                                      if k != "seed"}
            table = results_table(rows)
            tables[(algo, belief_mode)] = table
            print("\n" + table.round(4).to_string())

    # --- persist ---
    with open(out_dir / "rl_metrics_per_seed.json", "w") as f:
        json.dump(per_seed_records, f, indent=2)
    for (algo, belief_mode), table in tables.items():
        table.round(6).to_csv(out_dir / f"rl_results_{algo}_{belief_mode}.csv")
    print(f"\nSaved metrics and tables -> {out_dir}")
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=["ppo"],
                   choices=["ppo", "sac"], help="algorithms to run")
    p.add_argument("--belief", nargs="+", default=["tier1"],
                   choices=["tier1", "placeholder"], dest="belief_modes")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--steps-ppo", type=int, default=PPO_TOTAL_STEPS)
    p.add_argument("--steps-sac", type=int, default=SAC_TOTAL_STEPS)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--ep-len", type=int, default=EP_LEN)
    p.add_argument("--out", default="artifacts/rl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget end-to-end sanity run")
    args = p.parse_args()

    if args.smoke:
        args.steps_ppo, args.steps_sac = 3000, 2000
        args.eval_episodes = 5

    run_matrix(args.algos, args.belief_modes, args.seeds, args.steps_ppo,
               args.steps_sac, args.eval_episodes, args.ep_len, args.out,
               args.overwrite)


if __name__ == "__main__":
    main()
