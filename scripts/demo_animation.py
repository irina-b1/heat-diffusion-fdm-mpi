#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


def decomp_1d(n,p,pid):

    base=n//p
    rem=n%p

    if pid<rem:
        count=base+1
        start=pid*(base+1)
    else:
        count=base
        start=rem*(base+1)+(pid-rem)*base

    return start,count


class LocalGrid:

    def __init__(self,start_row,rows,start_col,cols):
        self.start_row=start_row
        self.rows=rows
        self.start_col=start_col
        self.cols=cols


def build_boundary_masks(N, sub):

    g_rows=sub.start_row+np.arange(sub.rows)
    g_cols=sub.start_col+np.arange(sub.cols)

    rows=g_rows[:,None]
    cols=g_cols[None,:]

    row_any=np.ones_like(rows,dtype=bool)
    col_any=np.ones_like(cols,dtype=bool)

    top_mask=(rows==0) & col_any
    bottom_mask=(rows==N-1) & col_any
    left_mask=row_any & (cols==0)
    right_mask=row_any & (cols==N-1)

    fixed_mask=top_mask | bottom_mask | left_mask | right_mask
    return fixed_mask,(top_mask,left_mask,bottom_mask,right_mask)


def apply_dirichlet_bc(T_interior,fixed_mask,boundary_masks,bc):

    if not fixed_mask.any():
        return

    top_mask,left_mask,bottom_mask,right_mask=boundary_masks
    top_v,left_v,bottom_v,right_v=bc

    T_interior[left_mask]=left_v
    T_interior[right_mask]=right_v
    T_interior[top_mask]=top_v
    T_interior[bottom_mask]=bottom_v


def exchange_halos(cart,T,sub,north,south,west,east):

    rows=sub.rows
    cols=sub.cols

    cart.Sendrecv(
        T[1,1:cols+1],
        north,0,
        T[0,1:cols+1],
        north,1
    )

    cart.Sendrecv(
        T[rows,1:cols+1],
        south,1,
        T[rows+1,1:cols+1],
        south,0
    )

    send_w=T[1:rows+1,1].copy()
    recv_w=np.empty(rows)

    cart.Sendrecv(
        send_w,west,20,
        recv_w,west,21
    )

    T[1:rows+1,0]=recv_w


    send_e=T[1:rows+1,cols].copy()
    recv_e=np.empty(rows)

    cart.Sendrecv(
        send_e,east,21,
        recv_e,east,20
    )

    T[1:rows+1,cols+1]=recv_e


def jacobi(T,Tnext,rows,cols):

    Tnext[1:rows+1,1:cols+1]=0.25*(
        T[0:rows,1:cols+1]
        +T[2:rows+2,1:cols+1]
        +T[1:rows+1,0:cols]
        +T[1:rows+1,2:cols+2]
    )


def gather_global(cart,T,sub,N):

    local=T[1:sub.rows+1,1:sub.cols+1]

    rank=cart.Get_rank()
    size=cart.Get_size()

    gathered=cart.gather(
        (
            sub.start_row,
            sub.rows,
            sub.start_col,
            sub.cols,
            local
        ),
        root=0
    )

    if rank!=0:
        return None

    full=np.zeros((N,N))

    for r0,nr,c0,nc,block in gathered:

        full[
            r0:r0+nr,
            c0:c0+nc
        ]=block

    return full


def save_gif(frames, gif_name, fps):

    if not frames:
        return

    try:
        import matplotlib.animation as animation
    except Exception as exc:
        raise RuntimeError(
            "GIF output requires matplotlib's animation support. "
            "Install matplotlib and pillow in WSL."
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0][1], origin="upper")
    ax.set_title(f"Iteration {frames[0][0]}")
    fig.colorbar(im, ax=ax)

    def update(item):
        iteration, grid = item
        im.set_data(grid)
        ax.set_title(f"Iteration {iteration}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(gif_name, writer=writer)
    plt.close(fig)


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--N",type=int,default=100)
    parser.add_argument("--iters",type=int,default=300)

    parser.add_argument(
        "--init",
        choices=["zero","random"],
        default="zero"
    )

    parser.add_argument(
        "--init-value",
        type=float,
        default=50
    )

    parser.add_argument(
        "--bc",
        nargs=4,
        type=float,
        default=[100,0,0,0]
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=5
    )

    args=parser.parse_args()

    project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir=os.path.join(project_root,"results")
    os.makedirs(results_dir,exist_ok=True)

    comm=MPI.COMM_WORLD

    rank=comm.Get_rank()
    size=comm.Get_size()

    dims=MPI.Compute_dims(
        size,
        [0,0]
    )

    cart=comm.Create_cart(
        dims,
        periods=[False,False]
    )

    coords=cart.Get_coords(rank)

    north,south=cart.Shift(0,1)
    west,east=cart.Shift(1,1)

    Px,Py=dims

    start_row,rows=decomp_1d(
        args.N,
        Px,
        coords[0]
    )

    start_col,cols=decomp_1d(
        args.N,
        Py,
        coords[1]
    )

    sub=LocalGrid(
        start_row,
        rows,
        start_col,
        cols
    )

    T=np.zeros(
        (rows+2,cols+2)
    )

    Tnext=np.zeros_like(T)

    fixed_mask,boundary_masks=build_boundary_masks(args.N,sub)

    if args.init=="random":

        T[1:rows+1,1:cols+1]=(
            np.random.rand(rows,cols)
            *args.init_value
        )

    top,left,bottom,right=args.bc

    apply_dirichlet_bc(
        T[1:rows+1,1:cols+1],
        fixed_mask,
        boundary_masks,
        (top,left,bottom,right)
    )
    apply_dirichlet_bc(
        Tnext[1:rows+1,1:cols+1],
        fixed_mask,
        boundary_masks,
        (top,left,bottom,right)
    )

    frames=[]

    for i in range(args.iters):

        exchange_halos(
            cart,T,sub,
            north,south,
            west,east
        )

        jacobi(
            T,Tnext,
            rows,
            cols
        )

        apply_dirichlet_bc(
            Tnext[1:rows+1,1:cols+1],
            fixed_mask,
            boundary_masks,
            (top,left,bottom,right)
        )

        T,Tnext=Tnext,T

        if i%args.frame_step==0:

            grid=gather_global(
                cart,
                T,
                sub,
                args.N
            )

            if rank==0:
                frames.append((i, grid.copy()))

    if rank==0:
        gif_name=os.path.join(
            results_dir,
            f"heat_N{args.N}_p{size}.gif"
        )

        save_gif(frames, gif_name, args.fps)

        print(
            f"Saved: {gif_name}"
        )


if __name__=="__main__":
    main()