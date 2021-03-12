from numba import cuda
from numpy import ceil
from numpy.random import randint
from time import time as timer
from visualize import gif

# globals
nx, ny = 512, 256
outfolder = "out_gpu"

# CUDA setup
threadsperblock = (32, 32)
blockspergrid_x = int(ceil(nx / threadsperblock[0]))
blockspergrid_y = int(ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

@cuda.jit
def toroidalize(grida, gridb):
    """Morph the NDarray `grida` into
    a toroidal NDarray output in `gridb`
    """
    x, y = cuda.grid(2)
    if (x == 0 or x == nx+1) and (y >= 1 and y <= ny):
        gridb[x,y] = grida[nx-x-1,y-1]
    if (y == 0 or y == ny+1) and (x >= 1 and x <= nx):
        gridb[x,y] = grida[x-1,ny-y-1]

@cuda.jit
def update(grida, gridb):
    """Local cell update kernel
    """
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

def simulate(iterations=250):
    """CA simulation steps generation
    """
    frames = []
    print("Starting simulation...")
    start = timer()
    grid0 = cuda.to_device(randint(2, size=(nx,ny)))
    grid1 = cuda.device_array((nx+2, ny+2))
    for _ in range(iterations):
        toroidalize[blockspergrid, threadsperblock](grid0, grid1)
        update[blockspergrid, threadsperblock](grid0, grid1)
        grid0 = grid1[1:nx+1,1:ny+1]
        frames.append(grid0.copy_to_host())
    end = timer()
    print("Simulation done in {:3.3f} seconds.".format(end - start))
    return frames

if __name__ == "__main__":
    gif(simulate(), 25, outfolder)