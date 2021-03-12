from numba import cuda
from numpy import ceil, zeros
from numpy.random import normal
from time import time as timer
from visualize import image_seq

# globals
nx, ny = 512, 256
kernel_ray = 3
rings = [0.1, 0.15, 0.25]
outfolder = "out_gpu"

# CUDA setup
threadsperblock = (32, 32)
blockspergrid_x = int(ceil(nx / threadsperblock[0]))
blockspergrid_y = int(ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

def kernel(ray, rings):
    edge = ray * 2 + 1
    mask = zeros((edge, edge))
    for r in range(ray):
        mask[r:edge-r,r:edge-r] = rings[r] / (8*(r+1))
    mask[ray,ray] = 0.5
    return mask

@cuda.jit
def toroidalize(grida, gridb):
    """Morph the NDarray `grida` into
    a toroidal NDarray output in `gridb`
    """
    x, y = cuda.grid(2)
    if y >= 1 and y <= ny:
        if x >= 0 and x <= r:
            gridb[x,y] = grida[r-kernel_ray, y-kernel_ray]
        elif x >= nx+kernel_ray and x <= nx+2*kernel_ray-1:
            gridb[x,y] = grida[r, y-kernel_ray]
    elif x >= 1 and x <= ny:
        if y >= 0 and y <= r:
            gridb[x,y] = grida[x-kernel_ray, r-kernel_ray]
        elif y >= ny+kernel_ray and y <= ny+2*kernel_ray-1:
            gridb[x,y] = grida[x-kernel_ray, r]

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
    grid0 = cuda.to_device(normal(0, 1, (nx,ny)))
    grid1 = cuda.device_array((nx+kernel_ray*2, ny+kernel_ray*2))
    for _ in range(iterations):
        toroidalize[blockspergrid, threadsperblock](grid0, grid1)
        update[blockspergrid, threadsperblock](grid0, grid1)
        grid0 = grid1[kernel_ray:nx+kernel_ray,
                      kernel_ray:ny+kernel_ray]
        frames.append(grid0.copy_to_host())
    end = timer()
    print("Simulation done in {:3.3f} seconds.".format(end - start))
    return frames

if __name__ == "__main__":
    image_seq(simulate(), outfolder, cmap="jet")