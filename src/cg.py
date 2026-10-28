"""МСГ для (A.T A + alpha I)x=A.T b; векторы распределены по сетке."""

import numpy as np
from mpi4py import MPI

from .previous.common import Result


def conjugate_gradient_method(
    grid, b, comm_row, comm_col, alpha=0.0, tol=1e-10, atol=1e-12, maxiter=None
):
    # Параметры согласованы на всех рангах до начала коллективных вызовов.
    alpha, tol, atol, maxiter = grid.comm.bcast(
        (alpha, tol, atol, maxiter) if grid.comm.rank == 0 else None, root=0
    )
    if not all(np.isfinite(v) and v >= 0 for v in (alpha, tol, atol)):
        raise ValueError("alpha, tol, atol должны быть конечными и неотрицательными")
    maxiter = 4 * grid.N if maxiter is None else maxiter
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter должен быть положительным целым")

    def dot(a, c):
        scalar = np.zeros(1)
        if grid.i == 0:
            comm_row.Allreduce(np.array([a @ c]), scalar, op=MPI.SUM)
        comm_col.Bcast(scalar, root=0)
        return float(scalar[0])

    r = grid.atx(b)
    x = np.zeros(grid.n) if grid.i == 0 else None
    p = r.copy() if grid.i == 0 else None
    rr = dot(r, r)
    history = [float(np.sqrt(rr))]
    threshold = max(atol, tol * history[0])
    reason = "tolerance" if history[0] <= threshold else "maxiter"
    grid.comm.Barrier()
    start = MPI.Wtime()
    if history[0] > threshold:
        for _ in range(maxiter):
            q = grid.atx(grid.ax(p))
            if grid.i == 0:
                q += alpha * p
            denominator = dot(p, q)
            if not np.isfinite(denominator) or denominator <= 0:
                reason = "nonpositive_curvature"
                break
            step = rr / denominator
            if grid.i == 0:
                x += step * p
                r -= step * q
            rr_new = dot(r, r)
            if not np.isfinite(rr_new):
                reason = "nonfinite_residual"
                break
            history.append(float(np.sqrt(rr_new)))
            if history[-1] <= threshold:
                reason = "tolerance"
                break
            if grid.i == 0:
                p *= rr_new / rr
                p += r
            rr = rr_new
    elapsed = grid.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
    answer = grid.gather(x, "N")
    return Result(
        answer, len(history) - 1, reason == "tolerance", history, elapsed, reason
    )
