"""Запуск умножения или МСГ. in.dat: M строк, затем N столбцов."""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.cg import conjugate_gradient_method
from src.grid import Grid
from src.previous.common import on_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["matvec", "cg"])
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).parent / "data/sample"
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    def read():
        M, N = map(int, (args.data_dir / "in.dat").read_text().split())
        A = np.loadtxt(args.data_dir / "AData.dat").reshape(M, N)
        v = np.loadtxt(
            args.data_dir / ("xData.dat" if args.operation == "matvec" else "bData.dat")
        ).reshape(-1)
        return A, v

    try:
        data = on_root(comm, read)
        A, v = data if comm.rank == 0 else (None, None)
        grid = Grid(A, comm)
        try:
            if args.operation == "matvec":
                result = grid.gather(grid.ax(grid.scatter(v, "N")), "M")
                if comm.rank == 0:
                    np.testing.assert_allclose(result, A @ v, rtol=1e-12, atol=1e-12)
                    np.savetxt("Results_matvec.dat", result)
                    print("A @ x: проверка NumPy пройдена")
            else:
                b = grid.scatter(v, "M")
                result = conjugate_gradient_method(
                    grid, b, grid.row, grid.col, alpha=args.alpha
                )
                if comm.rank == 0:
                    if args.alpha:
                        ref = np.linalg.solve(
                            A.T @ A + args.alpha * np.eye(grid.N), A.T @ v
                        )
                    else:
                        ref = np.linalg.lstsq(A, v, rcond=None)[0]
                    np.testing.assert_allclose(result.x, ref, rtol=1e-7, atol=1e-8)
                    if not result.converged:
                        raise ValueError(f"МСГ не сошёлся: {result.stop_reason}")
                    np.savetxt("Results_cg.dat", result.x)
                    print(
                        f"МСГ: проверка NumPy пройдена; итераций {result.iterations}; цикл {result.kernel_seconds:.6f} с"
                    )
        finally:
            grid.free()
    except (OSError, ValueError) as exc:
        if comm.rank == 0:
            print(f"Ошибка: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
