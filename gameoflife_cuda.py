import numpy as np
from numba import cuda
from matplotlib.pyplot import imsave

# globals
nx, ny = 256, 512
iters = 500
outfolder = "out_gpu"

# CUDA setup
threadsperblock = (32, 32)
blockspergrid_x = int(np.ceil(nx / threadsperblock[0]))
blockspergrid_y = int(np.ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

@cuda.jit
def ghostcells(grida, gridb):
    x, y = cuda.grid(2)
    if (x == 0 or x == nx+1) and (y >= 1 and y <= ny):
        gridb[x,y] = grida[nx-x-1,y-1]
    if (y == 0 or y == ny+1) and (x >= 1 and x <= nx):
        gridb[x,y] = grida[x-1,ny-y-1]

@cuda.jit
def update(grida, gridb):
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        neighbors = 0
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                neighbors += grida[i,j]
        if grida[x,y] == 1 and neighbors not in [3,4]:
            gridb[x+1,y+1] = 0
        elif grida[x,y] == 0 and neighbors == 3:
            gridb[x+1,y+1] = 1

def simulate():
    grid0 = cuda.to_device(np.random.randint(2, size=(nx,ny)))
    grid1 = cuda.device_array((nx+2, ny+2))
    for iter in range(iters):
        ghostcells[blockspergrid, threadsperblock](grid0, grid1)
        update[blockspergrid, threadsperblock](grid0, grid1)
        grid0 = grid1[1:nx+1,1:ny+1]
        imsave(outfolder + "/{0:04d}.png".format(iter), grid0)

if __name__ == "__main__":
    simulate()