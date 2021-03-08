import numpy as np
from matplotlib.pyplot import imsave

def update(grida, gridb, i, j):
    if grida[i,j] == 1 and grida[i-1:i+2, j-1:j+2].sum() not in [3,4]:
        gridb[i+1,j+1] = 0
    elif grida[i,j] == 0 and grida[i-1:i+2, j-1:j+2].sum() == 3:
        gridb[i+1,j+1] = 1

def main(nx, ny, iters, outfolder):
    grid0 = np.random.randint(2, size=(nx,ny))
    grid1 = np.zeros((nx+2,ny+2))
    grid1[1:nx+1,1:ny+1] = grid0.copy()
    for iter in range(iters):
        grid1[0,1:ny+1] = grid0[-1,:].copy()
        grid1[nx,1:ny+1] = grid0[0,:].copy()
        grid1[1:nx+1,0] = grid0[-1,:].copy()
        grid1[1:nx+1,ny] = grid0[:,0].copy()
        for i in range(nx):
            for j in range(ny):
                update(grid0, grid1, i, j)
        grid0 = grid1[1:nx+1,1:ny+1].copy()
        imsave(outfolder + "/{0:04d}.png".format(iter), grid0)

if __name__ == "__main__":
    nx, ny = 512, 512
    iters = 200
    outfolder = "out"
    main(nx, ny, iters, outfolder)