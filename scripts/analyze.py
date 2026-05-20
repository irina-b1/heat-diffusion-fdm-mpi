#!/usr/bin/env python3
"""Analyze raw benchmark CSV and compute speedup + Karp-Flatt metric.

Input: CSV created by scripts/bench.sh with columns:
  p,run,T_total,iters,N,dims

Output:
- Prints a Markdown table suitable for README/report.
- Optionally saves plots under results/ (runtime, speedup, efficiency, Karp-Flatt).

Karp-Flatt metric:
  e_p = (1/S_p - 1/p) / (1 - 1/p)
where S_p = T_1 / T_p and T_p is the *average* of 3 runs.

"""
import argparse
import csv
import math
from pathlib import Path


PLOT_CHOICES = ("runtime", "speedup", "efficiency", "karp", "all")


class Row:
    def __init__(self, p: int, run: int, t_total: float):
        self.p = p
        self.run = run
        self.t_total = t_total


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/raw_times.csv", help="Input raw CSV")
    ap.add_argument("--out", dest="out_path", default="", help="Optional output markdown file")
    ap.add_argument(
        "--plot",
        nargs="*",
        default=[],
        choices=PLOT_CHOICES,
        help="Plots to save: runtime speedup efficiency karp all",
    )
    ap.add_argument(
        "--plot-dir",
        default="results",
        help="Directory to write plot images into (default: results)",
    )
    ap.add_argument(
        "--plot-format",
        default="png",
        help="Image format for plots (default: png; e.g. png, pdf, svg)",
    )
    return ap.parse_args()


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append(
                Row(
                    p=int(line["p"]),
                    run=int(line["run"]),
                    t_total=float(line["T_total"]),
                )
            )
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute_stats(by_p: dict[int, list[float]]) -> dict[int, dict[str, float]]:
    """Compute mean/stdev runtime, speedup, efficiency, and Karp-Flatt e_p per p."""
    if 1 not in by_p:
        raise ValueError("Need p=1 data to compute speedup")

    t1 = mean(by_p[1])
    out: dict[int, dict[str, float]] = {}
    for p, ts in by_p.items():
        avg_tp = mean(ts)
        sd_tp = stdev(ts)
        sp = t1 / avg_tp if avg_tp > 0 else float("nan")
        eff = sp / p if p > 0 else float("nan")
        if p == 1:
            e_p = 0.0
        else:
            e_p = (1.0 / sp - 1.0 / p) / (1.0 - 1.0 / p) if sp > 0 else float("nan")

        out[p] = {
            "t_avg": avg_tp,
            "t_sd": sd_tp,
            "speedup": sp,
            "efficiency": eff,
            "karp_flatt": e_p,
        }

    return out


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Plotting requires matplotlib. Install it (e.g. pip install matplotlib). "
            f"Original error: {e}"
        )


def save_plots(
    *,
    ps: list[int],
    stats: dict[int, dict[str, float]],
    plot_dir: Path,
    plot_format: str,
    requested: list[str],
) -> list[Path]:
    if not requested:
        return []

    requested_set = set(requested)
    if "all" in requested_set:
        requested_set = {"runtime", "speedup", "efficiency", "karp"}

    plt = _ensure_matplotlib()
    plot_dir.mkdir(parents=True, exist_ok=True)

    xs = ps
    t_avgs = [stats[p]["t_avg"] for p in ps]
    t_sds = [stats[p]["t_sd"] for p in ps]
    speedups = [stats[p]["speedup"] for p in ps]
    effs = [stats[p]["efficiency"] for p in ps]
    karps = [stats[p]["karp_flatt"] for p in ps]

    # Approximate error propagation (uses only T_p stdev):
    #   S_p = T_1 / T_p  =>  sigma(S_p) ≈ S_p * sigma(T_p) / mean(T_p)
    speedup_sds: list[float] = []
    eff_sds: list[float] = []
    for p in ps:
        t_avg = stats[p]["t_avg"]
        t_sd = stats[p]["t_sd"]
        sp = stats[p]["speedup"]
        if t_avg > 0 and math.isfinite(sp):
            sp_sd = abs(sp) * (t_sd / t_avg)
        else:
            sp_sd = float("nan")
        speedup_sds.append(sp_sd)
        eff_sds.append(sp_sd / p if p > 0 and math.isfinite(sp_sd) else float("nan"))

    saved: list[Path] = []

    def _save(fig, name: str) -> None:
        out_path = plot_dir / f"{name}.{plot_format}"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved.append(out_path)

    if "runtime" in requested_set:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.errorbar(xs, t_avgs, yerr=t_sds, marker="o", linestyle="-", capsize=3)
        ax.set_xlabel("p (processes)")
        ax.set_ylabel("Average runtime T_p [s]")
        ax.set_title("Runtime vs processes")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        _save(fig, "runtime_vs_p")

    if "speedup" in requested_set:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.errorbar(xs, speedups, yerr=speedup_sds, marker="o", linestyle="-", capsize=3, label="Measured")
        ax.plot(xs, xs, linestyle="--", label="Ideal")
        ax.set_xlabel("p (processes)")
        ax.set_ylabel("Speedup S_p")
        ax.set_title("Speedup vs processes")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "speedup_vs_p")

    if "efficiency" in requested_set:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.errorbar(xs, effs, yerr=eff_sds, marker="o", linestyle="-", capsize=3, label="Measured")
        ax.axhline(1.0, linestyle="--", label="Ideal")
        ax.set_xlabel("p (processes)")
        ax.set_ylabel("Efficiency E_p = S_p / p")
        ax.set_title("Efficiency vs processes")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "efficiency_vs_p")

    if "karp" in requested_set:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(xs, karps, marker="o", linestyle="-")
        ax.set_xlabel("p (processes)")
        ax.set_ylabel("Karp-Flatt e_p")
        ax.set_title("Karp-Flatt metric vs processes")
        ax.grid(True, alpha=0.3)
        _save(fig, "karp_flatt_vs_p")

    return saved


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    rows = read_rows(in_path)

    by_p: dict[int, list[float]] = {}
    for row in rows:
        by_p.setdefault(row.p, []).append(row.t_total)

    ps = sorted(by_p.keys())
    stats = compute_stats(by_p)

    lines: list[str] = []
    lines.append("| p | T₁ | T₂ | T₃ | avg Tₚ [s] | Sₚ | Eₚ | Karp-Flatt e_p |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    for p in ps:
        ts = by_p[p]
        ts_sorted = ts[:]
        # Keep stable ordering for first three, but don't crash if user has more/less.
        while len(ts_sorted) < 3:
            ts_sorted.append(float("nan"))

        avg_tp = stats[p]["t_avg"]
        sp = stats[p]["speedup"]
        eff = stats[p]["efficiency"]
        e_p = stats[p]["karp_flatt"]

        lines.append(
            "| {p} | {t1:.6f} | {t2:.6f} | {t3:.6f} | {avg:.6f} | {sp:.3f} | {eff:.3f} | {e_p:.4f} |".format(
                p=p,
                t1=ts_sorted[0],
                t2=ts_sorted[1],
                t3=ts_sorted[2],
                avg=avg_tp,
                sp=sp,
                eff=eff,
                e_p=e_p,
            )
        )

    out_md = "\n".join(lines) + "\n"
    print(out_md)

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_md, encoding="utf-8")

    plot_dir = Path(args.plot_dir)
    saved = save_plots(
        ps=ps,
        stats=stats,
        plot_dir=plot_dir,
        plot_format=str(args.plot_format).lstrip("."),
        requested=list(args.plot),
    )
    if saved:
        print("Saved plots:")
        for p in saved:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
