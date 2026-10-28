"""Один MPI-запуск: прогрев и пять повторений трёх версий МСГ."""

import json
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cg import conjugate_gradient_method
from src.grid import Grid
from src.previous.cg_full import solve as full
from src.previous.cg_simple import solve as simple

comm = MPI.COMM_WORLD
directory = Path(sys.argv[1])
A = b = truth = None
if comm.rank == 0:
    A = np.load(directory / "A.npy")
    b = np.load(directory / "b.npy")
    truth = np.load(directory / "x.npy")
rows = []
grid = Grid(A, comm)
local_b = grid.scatter(b, "M")
for name in ["full", "simple", "grid"]:
    times, iterations, errors = [], [], []
    for trial in range(6):
        if name == "grid":
            result = conjugate_gradient_method(grid, local_b, grid.row, grid.col)
        else:
            result = (full if name == "full" else simple)(A, b, comm=comm)
        error = None
        if comm.rank == 0:
            error = float(np.linalg.norm(result.x - truth) / np.linalg.norm(truth))
            if not result.converged or error > 1e-7:
                comm.Abort(2)
        if trial:
            times.append(result.kernel_seconds)
            iterations.append(result.iterations)
            if comm.rank == 0:
                errors.append(error)
    if comm.rank == 0:
        rows.append(
            dict(
                version=name,
                processes=comm.size,
                samples_s=times,
                mean_s=float(np.mean(times)),
                std_s=float(np.std(times, ddof=1)),
                iterations=iterations,
                max_relative_error=max(errors),
            )
        )
grid.free()
if comm.rank == 0:
    print(json.dumps(dict(mpi=MPI.Get_library_version().strip(), rows=rows)))
