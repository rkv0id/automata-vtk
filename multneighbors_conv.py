from numba import cuda
from numpy import ceil, zeros
from numpy.random import normal
from time import time as timer
from visualize import image_seq

# globals
nx, ny = 1024, 512
outfolder = "out_gpu"
kernel_ray = 4
rings = [0.04, 0.085, 0.125, 0.25]

def kernel(ray, rings):
    edge = ray * 2 + 1
    mask = zeros((edge, edge))
    for r in range(ray):
        mask[r:edge-r,r:edge-r] = rings[r] / (8*(r+1))
    mask[ray,ray] = 0.5
    return mask
ker = kernel(kernel_ray, rings)

# CUDA setup
threadsperblock = (32, 16)
blockspergrid_x = int(ceil(nx / threadsperblock[0]))
blockspergrid_y = int(ceil(ny / threadsperblock[1]))
blockspergrid   = (blockspergrid_x, blockspergrid_y)

@cuda.jit
def toroidalize(grida, gridb):
    """Morph the NDarray `grida` into
    a toroidal NDarray output in `gridb`
    """
    x, y = cuda.grid(2)
    if y >= kernel_ray and y < ny+kernel_ray:
        if x >= 0 and x < kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-kernel_ray]
        elif x >= nx+kernel_ray and x < nx+2*kernel_ray:
            gridb[x,y] = grida[x-nx-kernel_ray, y-kernel_ray]
    elif x >= kernel_ray and x < nx+kernel_ray:
        if y >= 0 and y < kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-kernel_ray]
        elif y >= ny+kernel_ray and y < ny+2*kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-ny-kernel_ray]

@cuda.jit
def update(grida, gridb):
    """Local cell global kernel convolution
    """
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        conv = 0
        for i in range(x, x+2*kernel_ray+1):
            for j in range(y, y+2*kernel_ray+1):
                conv += ker[i-x, j-y] * gridb[i,j]
        grida[x,y] = conv

def simulate(iterations=500):
    """CA simulation steps generation
    """
    frames = []
    print("Starting simulation...")
    grid    = cuda.to_device(normal(0, 1, (nx,ny)))
    toroid  = cuda.device_array((nx+kernel_ray*2, ny+kernel_ray*2))
    start = timer()
    for _ in range(iterations):
        toroidalize[blockspergrid, threadsperblock](grid, toroid)
        toroid[kernel_ray:nx+kernel_ray,
               kernel_ray:ny+kernel_ray] = grid
        update[blockspergrid, threadsperblock](grid, toroid)
        frames.append(grid.copy_to_host())
    end = timer()
    print("Simulation done in {:3.3f} seconds.".format(end - start))
    return frames

if __name__ == "__main__":
    image_seq(simulate(), outfolder, cmap="jet")