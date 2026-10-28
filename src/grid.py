"""Двумерное распределение и операции A @ x, A.T @ y."""

from math import isqrt

import numpy as np
from mpi4py import MPI

from .previous.common import on_root, partition


class Grid:
    def __init__(self, A=None, comm=MPI.COMM_WORLD):
        self.comm = comm
        q = isqrt(comm.size)
        if q * q != comm.size:
            raise ValueError(
                "Число процессов должно быть полным квадратом: 1, 4, 9, 16..."
            )
        self.q = q
        self.i, self.j = divmod(comm.rank, q)
        self.row = comm.Split(self.i, comm.rank)
        self.col = comm.Split(self.j, comm.rank)

        def validate():
            value = np.asarray(A, dtype=np.float64)
            if value.ndim != 2 or min(value.shape) <= 0 or not np.isfinite(value).all():
                raise ValueError("Требуется конечная непустая матрица M x N")
            return np.ascontiguousarray(value)

        A = on_root(comm, validate)
        dims = np.array(A.shape if comm.rank == 0 else [0, 0], dtype=np.int64)
        comm.Bcast(dims, root=0)
        self.M, self.N = map(int, dims)
        self.mc, self.md = partition(self.M, q)
        self.nc, self.nd = partition(self.N, q)
        m, n = np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
        if self.i == 0:
            self.row.Scatter(self.nc if self.j == 0 else None, n, root=0)
        self.col.Bcast(n, root=0)
        if self.j == 0:
            self.col.Scatter(self.mc if self.i == 0 else None, m, root=0)
        self.row.Bcast(m, root=0)
        self.m, self.n = int(m[0]), int(n[0])
        self.A = np.empty((self.m, self.n))

        # Временная группа: процесс 0 и одна строка сетки.
        group = comm.Get_group()
        for i in range(q):
            ranges = (
                [(0, q - 1, 1)] if i == 0 else [(0, 0, 1), (i * q, (i + 1) * q - 1, 1)]
            )
            selected = group.Range_incl(ranges)
            temporary = comm.Create(selected)  # коллективно на COMM_WORLD
            if temporary != MPI.COMM_NULL:
                counts = self.mc[i] * self.nc
                if i:
                    counts = np.concatenate(([0], counts)).astype(np.int32)
                offsets = np.zeros(len(counts), dtype=np.int32)
                offsets[1:] = np.cumsum(counts[:-1])
                packed = None
                if comm.rank == 0:
                    first = int(self.md[i])
                    block = A[first : first + int(self.mc[i])]
                    packed = np.concatenate(
                        [
                            block[
                                :, int(self.nd[j]) : int(self.nd[j] + self.nc[j])
                            ].ravel()
                            for j in range(q)
                        ]
                    )
                receive = self.A.ravel() if self.i == i else np.empty(0)
                temporary.Scatterv(
                    [packed, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None,
                    receive,
                    root=0,
                )
                temporary.Free()
            selected.Free()
        group.Free()

    def scatter(self, vector, axis):
        """axis=N: первая строка; axis=M: первый столбец."""
        length = self.N if axis == "N" else self.M

        def validate():
            v = np.ascontiguousarray(vector, dtype=np.float64)
            if v.shape != (length,) or not np.isfinite(v).all():
                raise ValueError(f"Требуется конечный вектор длины {length}")
            return v

        vector = on_root(self.comm, validate)
        active = self.i == 0 if axis == "N" else self.j == 0
        if not active:
            return None
        counts, offsets = (self.nc, self.nd) if axis == "N" else (self.mc, self.md)
        communicator = self.row if axis == "N" else self.col
        local = np.empty(self.n if axis == "N" else self.m)
        communicator.Scatterv(
            [vector, counts, offsets, MPI.DOUBLE] if self.comm.rank == 0 else None,
            local,
            root=0,
        )
        return local

    def ax(self, x):
        local = x.copy() if self.i == 0 else np.empty(self.n)
        self.col.Bcast(local, root=0)
        result = np.empty(self.m) if self.j == 0 else None
        self.row.Reduce(self.A @ local, result, op=MPI.SUM, root=0)
        return result

    def atx(self, y):
        local = y.copy() if self.j == 0 else np.empty(self.m)
        self.row.Bcast(local, root=0)
        result = np.empty(self.n) if self.i == 0 else None
        self.col.Reduce(self.A.T @ local, result, op=MPI.SUM, root=0)
        return result

    def gather(self, local, axis):
        active = self.i == 0 if axis == "N" else self.j == 0
        result = None
        if active:
            counts, offsets = (self.nc, self.nd) if axis == "N" else (self.mc, self.md)
            communicator = self.row if axis == "N" else self.col
            if self.comm.rank == 0:
                result = np.empty(self.N if axis == "N" else self.M)
            communicator.Gatherv(
                local,
                [result, counts, offsets, MPI.DOUBLE] if self.comm.rank == 0 else None,
                root=0,
            )
        return result

    def free(self):
        self.row.Free()
        self.col.Free()
