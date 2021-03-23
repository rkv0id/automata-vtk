from numpy import stack, zeros, max, min
from matplotlib.pyplot import imsave, cm
from array2gif import write_gif
from skvideo.io import FFmpegWriter

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

def mp4(grids, fps=25, outname="out_gpu"):
    w = FFmpegWriter(outname + ".mp4", inputdict={"-r": str(fps)})
    print("Generating MP4 video...")
    for frame in grids:
        mn, mx = min(frame), max(frame)
        frame -= mn
        frame /= (mx-mn)
        frame = cm.jet(frame.transpose())
        frame *= 255
        frame = frame.astype("uint8")
        w.writeFrame(frame)
    w.close()
    print("MP4 created: " + outname + ".mp4")