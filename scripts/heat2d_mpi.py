import argparse
import sys
import numpy as np
from mpi4py import MPI

class LocalGrid:
    """Part of global matrix belonging to one rank."""

    def __init__(self,start_row,rows,start_col,cols):
        self.start_row=start_row # first global row handled by this process
        self.rows=rows # number of rows owned by this process
        self.start_col=start_col # first global column handled by this process
        self.cols=cols  # number of columns owned by this process


def decomp_1d(n: int, p: int, pid: int) -> tuple[int, int]:
    """Split N grid elements among P processes as evenly as possible.

    Returns:
        start -> first element handled by this process
        count -> number of elements assigned to this process
    This gives either floor(n/p) or ceil(n/p) elements.
    """

    if p <= 0:
        raise ValueError("p must be positive")
    if pid < 0 or pid >= p:
        raise ValueError("pid out of range")

    base = n // p
    rem = n % p

    if pid < rem:
        count = base + 1
        start = pid * (base + 1)
    else:
        count = base
        start = rem * (base + 1) + (pid - rem) * base

    return start, count


def build_boundary_masks(
    N: int, sub: LocalGrid
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Find which cells in this process belong to the global boundaries.

    Returns:
      fixed_mask, (top_mask, left_mask, bottom_mask, right_mask)

    Each mask matches the size of this process's local interior grid.
    """

    g_rows = sub.start_row + np.arange(sub.rows, dtype=np.int64)
    g_cols = sub.start_col + np.arange(sub.cols, dtype=np.int64)

    rows = g_rows[:, None]  # (rows, 1)
    cols = g_cols[None, :]  # (1, cols)

    # Make masks shape (rows, cols) via broadcasting.
    row_any = np.ones_like(rows, dtype=bool)  # (rows, 1)
    col_any = np.ones_like(cols, dtype=bool)  # (1, cols)

    top_mask = (rows == 0) & col_any
    bottom_mask = (rows == N - 1) & col_any
    left_mask = row_any & (cols == 0)
    right_mask = row_any & (cols == N - 1)

    fixed_mask = top_mask | bottom_mask | left_mask | right_mask
    return fixed_mask, (top_mask, left_mask, bottom_mask, right_mask)


def apply_dirichlet_bc(
    T_interior: np.ndarray,
    fixed_mask: np.ndarray,
    boundary_masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bc: tuple[float, float, float, float],
) -> None:
    """Set fixed temperatures on boundary cells.

    `bc` order is: (top, left, bottom, right).

    Corner cells belong to two walls at ocolse.
    If two boundary values conflict, top/bottom values win.
    (For typical tasks, boundary values are consistent, e.g. top=100 and others=0.)
    """

    if not fixed_mask.any():
        return

    top_mask, left_mask, bottom_mask, right_mask = boundary_masks
    top_v, left_v, bottom_v, right_v = bc

    # Apply left/right first, then top/bottom so that top/bottom win at corners.
    T_interior[left_mask] = left_v
    T_interior[right_mask] = right_v
    T_interior[top_mask] = top_v
    T_interior[bottom_mask] = bottom_v


def exchange_halos(
    cart: MPI.Comm,
    T: np.ndarray,
    sub: LocalGrid,
    north: int,
    south: int,
    west: int,
    east: int,
) -> None:
    """Exchange halo rows/cols with 4 neighbors using Sendrecv.

    Local array layout:
      T.shape == (rows+2, cols+2)
      interior is T[1:rows+1, 1:cols+1]
      halos are row 0, row rows+1, col 0, col cols+1

    We use:
    - Row exchanges with contiguous slices.
    - Column exchanges by packing into 1D contiguous buffers.

    The neighbor ranks may be MPI.PROC_NULL, which makes Sendrecv a no-op.
    """

    rows, cols = sub.rows, sub.cols

    # Exchange north/south halo rows (contiguous)
    # Send first row upward, receive north neighbor's last row into top halo.
    cart.Sendrecv(
        sendbuf=T[1, 1 : cols + 1],
        dest=north,
        sendtag=0,
        recvbuf=T[0, 1 : cols + 1],
        source=north,
        recvtag=1,
    )

    # Send last row downward, receive south neighbor's first row into bottom halo.
    cart.Sendrecv(
        sendbuf=T[rows, 1 : cols + 1],
        dest=south,
        sendtag=1,
        recvbuf=T[rows + 1, 1 : cols + 1],
        source=south,
        recvtag=0,
    )

    # Copy left/right edge columns into temporary buffers and exchange them
    send_w = T[1 : rows + 1, 1].copy(order="C")
    recv_w = np.empty(rows, dtype=T.dtype)
    cart.Sendrecv(
        sendbuf=send_w,
        dest=west,
        sendtag=20,
        recvbuf=recv_w,
        source=west,
        recvtag=21,
    )
    T[1 : rows + 1, 0] = recv_w

    send_e = T[1 : rows + 1, cols].copy(order="C")
    recv_e = np.empty(rows, dtype=T.dtype)
    cart.Sendrecv(
        sendbuf=send_e,
        dest=east,
        sendtag=21,
        recvbuf=recv_e,
        source=east,
        recvtag=20,
    )
    T[1 : rows + 1, cols + 1] = recv_e


def jacobi_step_vectorized(T: np.ndarray, Tnext: np.ndarray, sub: LocalGrid) -> None:
    """Compute new temperatures using the Jacobi stecolsil formula.

    Computes:
      Tnext[i,j] = 0.25 * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1])

    for i=1..rows, j=1..cols in local indexing.

    Boundary/source constraints must be reapplied after this.
    """

    rows, cols = sub.rows, sub.cols

    # Each cell becomes the average of its four neighbors.
    Tnext[1 : rows + 1, 1 : cols + 1] = 0.25 * (
        T[0:rows, 1 : cols + 1]
        + T[2 : rows + 2, 1 : cols + 1]
        + T[1 : rows + 1, 0:cols]
        + T[1 : rows + 1, 2 : cols + 2]
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D heat diffusion (Jacobi) with MPI 2D domain decomposition")
    p.add_argument("--N", type=int, default=512, help="Global grid size N (grid is NxN)")
    p.add_argument("--iters", type=int, default=100000, help="Maximum Jacobi iterations")
    p.add_argument(
        "--bc",
        nargs=4,
        type=float,
        default=[100.0, 0.0, 0.0, 0.0],
        metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"),
        help="Dirichlet boundaries as 4 numbers: TOP LEFT BOTTOM RIGHT",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() #the ID number of a process inside the communicator
    size = comm.Get_size()

    N = int(args.N)
    iters = int(args.iters)
    bc = (float(args.bc[0]), float(args.bc[1]), float(args.bc[2]), float(args.bc[3]))

    if N < 2:
        if rank == 0:
            print("N must be >= 2", file=sys.stderr)
        return 2

    # 2D Cartesian topology
    dims = MPI.Compute_dims(size, [0, 0])  # e.g. as square as possible
    periods = (False, False)

    # reorder=False keeps mapping predictable for debugging.
    cart = comm.Create_cart(dims=dims, periods=periods, reorder=False) #creates a virtual coordinate system
    coords = cart.Get_coords(rank) # rank 2 for example coords=(1,0), second row, first column in a 2x2 grid of processes

    # Find Neighbors (use MPI.PROC_NULL on outer boundary where there is no neighbor).
    north, south = cart.Shift(direction=0, disp=1)
    west, east = cart.Shift(direction=1, disp=1)

    Px, Py = dims[0], dims[1]

    # Local block sizes and offsets, split matrix among ranks in 2D blocks.
    start_row, rows = decomp_1d(N, Px, coords[0])
    start_col, cols = decomp_1d(N, Py, coords[1])
    sub = LocalGrid(start_row=start_row, rows=rows, start_col=start_col, cols=cols)

    if rows <= 0 or cols <= 0:
        # This can happen if N is smaller than the process grid.
        # We keep it simple: require at least 1x1 interior per rank.
        print(
            f"Process {rank}: invalid decomposition rows={rows}, cols={cols}. "
            f"Choose larger N or fewer processes.",
            file=sys.stderr,
        )
        comm.Abort(2)
        return 2

    # Create local temperature grids icolsluding ghost-cell borders
    T = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    Tnext = np.zeros_like(T)

    # Identify cells that belong to the global walls
    fixed_mask, boundary_masks = build_boundary_masks(N, sub)

    # The boundary values never move. Only interior cells change. Like if we had heaters around walls.
    apply_dirichlet_bc(T[1 : rows + 1, 1 : cols + 1], fixed_mask, boundary_masks, bc)
    apply_dirichlet_bc(Tnext[1 : rows + 1, 1 : cols + 1], fixed_mask, boundary_masks, bc)

    # Time only the compute loop (halo exchange + update)
    cart.Barrier()
    t0 = MPI.Wtime()

    performed_iters = 0

    for k in range(iters):

        # 1) Exchange halos
        exchange_halos(cart, T, sub, north=north, south=south, west=west, east=east)

        # 2) Update
        jacobi_step_vectorized(T, Tnext, sub)

        # 3) Re-apply fixed boundary values, they cant change sicolse they are fixed by the BCs. This is important to do after the update, otherwise the fixed cells would drift.
        apply_dirichlet_bc(
            Tnext[1 : rows + 1, 1 : cols + 1],
            fixed_mask,
            boundary_masks,
            bc,
        )

        # 4) Swap buffers - Use the new temperatures as input for the next iteration
        T, Tnext = Tnext, T
        performed_iters = k + 1

    cart.Barrier()
    t1 = MPI.Wtime()

    # Print timing in machine-readable form for becolsh scripts
    total_time = t1 - t0
    # We use the max runtime across ranks as the parallel time.
    #the whole computation finishes only when the slowest process finishes, so we take the max time across ranks.
    Tp = cart.allreduce(total_time, op=MPI.MAX)

    if rank == 0:
        #rank 0's local block size for scaling analysis.
        # Note: other ranks may have different block sizes due to decomposition.
        msg = f"T_total={Tp:.6f} iters={performed_iters} N={N} p={size} dims={dims} local_rank0={rows}x{cols}"
        print(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
