"""Общие операции ввода, разбиения и завершения MPI-решателей."""

from dataclasses import dataclass

from mpi4py import MPI
import numpy as np


@dataclass
class Result:
    x: object
    iterations: int
    converged: bool
    history: list
    kernel_seconds: float
    stop_reason: str


def partition(length, size):
    if length < 0 or size <= 0:
        raise ValueError("Требуются length >= 0 и size > 0")
    base, extra = divmod(length, size)
    counts = np.full(size, base, dtype=np.int32)
    counts[:extra] += 1
    offsets = np.zeros(size, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])
    return counts, offsets


def on_root(comm, load):
    data = error = None
    if comm.rank == 0:
        try:
            data = load()
        except (OSError, ValueError, TypeError) as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)
    return data


def validate(A, b, alpha, tol, atol, maxiter):
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or min(A.shape) <= 0 or b.shape != (A.shape[0],):
        raise ValueError("Требуется A размера M x N и b длины M")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Все элементы должны быть конечными числами")
    if not all(np.isfinite(v) and v >= 0 for v in (alpha, tol, atol)):
        raise ValueError("alpha, tol, atol должны быть конечными и неотрицательными")
    if maxiter is None:
        maxiter = 4 * A.shape[1]
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter должен быть положительным целым")
    return A, b, float(alpha), float(tol), float(atol), int(maxiter)


def distribute(A, b, alpha, tol, atol, maxiter, comm):
    data = on_root(comm, lambda: validate(A, b, alpha, tol, atol, maxiter))
    if comm.rank == 0:
        A, b, alpha, tol, atol, maxiter = data
    dims = np.array(A.shape if comm.rank == 0 else [0, 0], dtype=np.int64)
    comm.Bcast(dims, root=0)
    alpha, tol, atol, maxiter = comm.bcast(
        (alpha, tol, atol, maxiter) if comm.rank == 0 else None, root=0
    )
    M, N = map(int, dims)
    rows, row_offsets = partition(M, comm.size)
    cols, col_offsets = partition(N, comm.size)
    local_A = np.empty((int(rows[comm.rank]), N), dtype=np.float64)
    local_b = np.empty(int(rows[comm.rank]), dtype=np.float64)
    comm.Scatterv(
        [A, rows * N, row_offsets * N, MPI.DOUBLE] if comm.rank == 0 else None,
        local_A,
        root=0,
    )
    comm.Scatterv(
        [b, rows, row_offsets, MPI.DOUBLE] if comm.rank == 0 else None, local_b, root=0
    )
    return local_A, local_b, N, cols, col_offsets, alpha, tol, atol, maxiter


def finish(comm, x, counts, offsets, replicated, history, elapsed, reason):
    seconds = comm.reduce(elapsed, op=MPI.MAX, root=0)
    answer = None
    if replicated:
        answer = x.copy() if comm.rank == 0 else None
    else:
        answer = (
            np.empty(int(counts.sum()), dtype=np.float64) if comm.rank == 0 else None
        )
        comm.Gatherv(
            x, [answer, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None, root=0
        )
    return Result(
        answer,
        len(history) - 1,
        reason == "tolerance",
        history,
        float(seconds) if comm.rank == 0 else 0.0,
        reason,
    )
