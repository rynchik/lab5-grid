"""Локальные наборы ЛР4; 1 прогрев + 5 измерений. Большие файлы не включены."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import generate


def main():
    config = json.loads((ROOT / "data/local.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[key] = "1"
    results = []
    for case in config["cases"]:
        directory = ROOT / "data/generated" / case["name"]
        generate(directory, case["M"], case["N"], case["seed"])
        for p in [1, 4, 9, 16]:
            print(f"{case['name']}: p={p}", flush=True)
            completed = subprocess.run(
                [
                    os.environ.get("MPIEXEC", "mpiexec"),
                    "-n",
                    str(p),
                    sys.executable,
                    str(ROOT / "scripts/worker.py"),
                    str(directory),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            for row in result["rows"]:
                row.update(case=case["name"], M=case["M"], N=case["N"])
                results.append(row)
            (ROOT / "report/raw.json").write_text(
                json.dumps(
                    dict(
                        mpi=result["mpi"], logical_cpus=os.cpu_count(), results=results
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    for row in results:
        base = next(
            r["mean_s"]
            for r in results
            if r["case"] == row["case"]
            and r["version"] == row["version"]
            and r["processes"] == 1
        )
        row["speedup"] = base / row["mean_s"]
        row["efficiency"] = row["speedup"] / row["processes"]
    fields = [
        "case",
        "M",
        "N",
        "version",
        "processes",
        "mean_s",
        "std_s",
        "speedup",
        "efficiency",
        "max_relative_error",
    ]
    with (ROOT / "report/results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    names = {
        "full": "ЛР3: распределённые векторы",
        "simple": "ЛР3: полные векторы",
        "grid": "ЛР5: двумерная сетка",
    }
    for c, case in enumerate(config["cases"]):
        for version, label in names.items():
            rows = [
                r
                for r in results
                if r["case"] == case["name"] and r["version"] == version
            ]
            for ax, metric in zip(axes[:, c], ["speedup", "efficiency"]):
                ax.plot(
                    [r["processes"] for r in rows],
                    [r[metric] for r in rows],
                    "o-",
                    label=label,
                )
                ax.set_xticks([1, 4, 9, 16])
                ax.grid(alpha=0.3)
        axes[0, c].set_title(f"{case['name']}: {case['M']} × {case['N']}")
        axes[1, c].set_xlabel("Число MPI-процессов")
    axes[0, 0].set_ylabel("Ускорение S(p)")
    axes[1, 0].set_ylabel("Эффективность E(p)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Сравнение МСГ: среднее пяти запусков, один компьютер")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(ROOT / "report/comparison.png", dpi=150)


if __name__ == "__main__":
    main()
