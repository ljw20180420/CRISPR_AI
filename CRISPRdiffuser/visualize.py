import matplotlib.pyplot as plt
from diffusers.utils import make_image_grid
import os
from .config import args

def show_mh_matrix(mh_maxtrix):
    # display mh_maxtrix
    plt.imshow(mh_maxtrix, cmap="Oranges", vmin=0, extent=(0, mh_maxtrix.shape[1], mh_maxtrix.shape[0], 0))
    plt.colorbar()
    plt.show()

def sample_images(images, epoch):
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=int(len(images)**0.5))
    # Save the images
    test_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")