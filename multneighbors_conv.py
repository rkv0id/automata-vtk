from numba import cuda
from math import tanh
from numpy import ceil, zeros
from numpy.random import seed, randn
from matplotlib.image import imread
from time import time as timer
from visualize import mp4

# globals
seed(12345)
nx, ny = 2048, 1024
outname = "out_gpu"
kernel_ray = 7
rings = [0.5, 0.01, -0.01, 0.1, -0.05, 0.15, -0.1, 0.2]

def kernel(ray, rings):
    edge = ray * 2 + 1
    mask = zeros((edge, edge))
    for r in range(ray):
        mask[r:edge-r,r:edge-r] = rings[r+1] / (8*(r+1))
    mask[ray,ray] = rings[0]
    return mask
ker = kernel(kernel_ray, rings)

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
    if kernel_ray <= y < ny+kernel_ray:
        if 0 <= x < kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-kernel_ray]
        elif nx+kernel_ray <= x < nx+2*kernel_ray:
            gridb[x,y] = grida[x-nx-kernel_ray, y-kernel_ray]
    elif kernel_ray <= x < nx+kernel_ray:
        if 0 <= y < kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-kernel_ray]
        elif ny+kernel_ray <= y < ny+2*kernel_ray:
            gridb[x,y] = grida[x-kernel_ray, y-ny-kernel_ray]

@cuda.jit
def update(grida, gridb):
    """Local cell global kernel convolution
    """
    x, y = cuda.grid(2)
    if 0 <= x < nx and 0 <= y < ny:
        conv = 0
        for i in range(x, x+2*kernel_ray+1):
            for j in range(y, y+2*kernel_ray+1):
                conv += ker[i-x, j-y] * gridb[i,j]
        grida[x,y] = tanh(conv)

def simulate(iterations=250, initgrid=None):
    """CA simulation steps generation
    """
    frames = []
    print("Starting simulation...")
    grid = cuda.to_device(initgrid if initgrid is not None else randn(nx,ny))
    toroid = cuda.device_array((nx+kernel_ray*2, ny+kernel_ray*2))
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
    # im = imread("sample.png").astype("float64")
    mp4(simulate(500), 25, outname)