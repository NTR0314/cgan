import os

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision.utils as vutils


def gen_plots(img_list, G_losses, D_losses, base_path, model_name=""):
    # Generate Animation of training progress
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list[::3]]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    writer = animation.FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(base_path / 'progress.mp4', writer=writer)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss During Training ({model_name})")
    plt.plot(G_losses[::10], label="G")
    plt.plot(D_losses[::10], label="D")
    ax = plt.gca()
    ax.set_ylim([0, 5.])
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(base_path / 'losses.svg')

