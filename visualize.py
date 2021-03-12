from numpy import stack, zeros
from matplotlib.pyplot import imsave
from array2gif import write_gif

def image_seq(grids, outfolder="out_gpu", cmap="jet"):
    print("Generating image sequence...")
    for i, grid in zip(range(len(grids)), grids):
        imsave(outfolder + "/{:04d}.png".format(i),
            grid.transpose(), cmap=cmap)
    print("Images generated at " + outfolder)

def gif(grids, fps=25, outname="out_gpu"):
    frames = []
    print("Generating GIF sequence...")
    for grid in grids:
        grid[grid == 1] = 255
        frames.append(
            stack([zeros(grid.shape), grid,
                zeros(grid.shape)], axis=2))
    write_gif(frames, outname+".gif", fps=fps)
    print("GIF created: " + outname + ".gif")