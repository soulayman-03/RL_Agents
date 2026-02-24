import argparse
import json
import os
from dataclasses import dataclass

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sl_tag(max_exposure_fraction: float) -> str:
    return f"sl_{float(max_exposure_fraction):.2f}".replace(".", "p")


def _candidate_run_roots(*, algorithm: str, model: str, max_exposure_fraction: float, seed: int) -> list[str]:
    safe_algo = (algorithm or "algo").replace(os.sep, "_")
    safe_model = (model or "model").replace(os.sep, "_")
    sl = _sl_tag(max_exposure_fraction)
    seed_dir = f"seed_{int(seed)}"

    # Current layout used by SingleAgent/dqn_train.py and SingleAgent/dqn_trainBigCNN.py
    #   results/resultDQN/<algo>/<model>/<sl>/seed_<seed>/train/train_log.jsonl
    roots = [
        os.path.join(SCRIPT_DIR, "results", "resultDQN", safe_algo, safe_model, sl, seed_dir),
        # Older/alternative layouts (kept for compatibility)
        os.path.join(SCRIPT_DIR, "results", "comparison", safe_algo, safe_model, sl, seed_dir),
        # BigCNN historical layout:
        #   results/result_<model>/<sl>/train/train_log.jsonl
        os.path.join(SCRIPT_DIR, "results", f"result_{safe_model}", sl),
        # Very old layout:
        #   results/resultDQN/<sl>/train/train_log.jsonl
        os.path.join(SCRIPT_DIR, "results", "resultDQN", sl),
    ]
    return roots


def _find_run_root(*, algorithm: str, model: str, max_exposure_fraction: float, seed: int) -> str:
    candidates = _candidate_run_roots(
        algorithm=algorithm, model=model, max_exposure_fraction=max_exposure_fraction, seed=seed
    )
    for root in candidates:
        if os.path.exists(os.path.join(root, "summary.json")):
            return root
        if os.path.exists(os.path.join(root, "train", "train_log.jsonl")):
            return root
    return candidates[0]


def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl_rewards(path: str) -> tuple[list[float], list[int]]:
    rewards: list[float] = []
    stalls: list[int] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rewards.append(float(obj.get("reward", 0.0)))
                    stalls.append(int(obj.get("stalls", 0)))
                except Exception:
                    continue
    except Exception:
        pass
    return rewards, stalls


@dataclass
class RunData:
    sl: float
    run_root: str
    summary: dict | None
    config: dict | None
    rewards: list[float]
    stalls: list[int]


def _moving_average(x: list[float], window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return np.asarray(x, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare runs across S_l values for a given model.")
    parser.add_argument("--algorithm", type=str, default="DQN")
    parser.add_argument("--model", type=str, default="cnn15", help="Model folder name (e.g., cnn15, simplecnn, hugcnn).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sl",
        type=float,
        nargs="+",
        default=[0.50, 0.33, 0.20],
        help="List of S_l (max exposure fraction) values to compare.",
    )
    parser.add_argument("--ma-window", type=int, default=50, help="Moving average window for reward curves.")
    args = parser.parse_args()

    algorithm = str(args.algorithm)
    model = str(args.model)
    seed = int(args.seed)
    sl_values = [float(v) for v in args.sl]

    runs: list[RunData] = []
    for sl in sl_values:
        root = _find_run_root(algorithm=algorithm, model=model, max_exposure_fraction=sl, seed=seed)
        summary = _read_json(os.path.join(root, "summary.json"))
        config = _read_json(os.path.join(root, "run_config.json"))
        rewards, stalls = _read_jsonl_rewards(os.path.join(root, "train", "train_log.jsonl"))
        runs.append(RunData(sl=sl, run_root=root, summary=summary, config=config, rewards=rewards, stalls=stalls))

    out_dir = os.path.join(
        SCRIPT_DIR,
        "results",
        "resultDQN",
        algorithm.replace(os.sep, "_"),
        model.replace(os.sep, "_"),
        "comparisons",
        f"seed_{seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    missing = [r for r in runs if r.summary is None and not r.rewards]
    if missing:
        print("Missing runs (no summary.json and no train_log.jsonl found):")
        for r in missing:
            print(f"  S_l={r.sl:.2f} expected at: {r.run_root}")

    # Aggregate metrics (prefer summary.json if present).
    rows: list[dict] = []
    for r in runs:
        if r.summary is not None:
            avg_reward_last_100 = r.summary.get("avg_reward_last_100", None)
            avg_stalls_last_100 = r.summary.get("avg_stalls_last_100", None)
        else:
            last_k = 100
            avg_reward_last_100 = float(np.mean(r.rewards[-last_k:])) if r.rewards else None
            avg_stalls_last_100 = float(np.mean(r.stalls[-last_k:])) if r.stalls else None

        rows.append(
            {
                "sl": float(r.sl),
                "run_root": r.run_root,
                "avg_reward_last_100": avg_reward_last_100,
                "avg_stalls_last_100": avg_stalls_last_100,
                "episodes_logged": int(len(r.rewards)),
            }
        )

    comp_path = os.path.join(out_dir, f"comparison_sl_{'_'.join([_sl_tag(v) for v in sl_values])}.json")
    try:
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algorithm": algorithm,
                    "model": model,
                    "seed": seed,
                    "sl_values": sl_values,
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    # Plots (optional, only if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        # Metrics plot
        sl_sorted = sorted(rows, key=lambda d: float(d["sl"]))
        xs = [float(d["sl"]) for d in sl_sorted]
        avg_r = [d["avg_reward_last_100"] for d in sl_sorted]
        avg_s = [d["avg_stalls_last_100"] for d in sl_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(xs, avg_r, marker="o")
        axes[0].set_title("Avg Reward (last 100) vs S_l")
        axes[0].set_xlabel("S_l (max exposure fraction)")
        axes[0].set_ylabel("Avg Reward (last 100)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xs, avg_s, marker="o", color="orange")
        axes[1].set_title("Avg Stalls (last 100) vs S_l")
        axes[1].set_xlabel("S_l (max exposure fraction)")
        axes[1].set_ylabel("Avg Stalls (last 100)")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"{algorithm} - {model} - seed {seed}")
        fig.tight_layout()
        metrics_png = os.path.join(out_dir, "compare_metrics.png")
        fig.savefig(metrics_png)
        plt.close(fig)

        # Reward curves
        plt.figure(figsize=(10, 6))
        for r in runs:
            if not r.rewards:
                continue
            ma = _moving_average(r.rewards, int(args.ma_window))
            start_x = (int(args.ma_window) - 1) if len(r.rewards) >= int(args.ma_window) else 0
            plt.plot(range(start_x, start_x + len(ma)), ma, label=f"S_l={r.sl:.2f}")
        plt.title(f"Reward Moving Average (window={int(args.ma_window)})")
        plt.xlabel("Episode")
        plt.ylabel("Reward (MA)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        curves_png = os.path.join(out_dir, "compare_reward_curves.png")
        plt.savefig(curves_png)
        plt.close()

        print(f"Wrote: {comp_path}")
        print(f"Wrote: {metrics_png}")
        print(f"Wrote: {curves_png}")
    except Exception as e:
        print(f"Wrote: {comp_path}")
        print(f"Plotting skipped (matplotlib not available or failed): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
