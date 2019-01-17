import numpy as np
from skimage.segmentation import mark_boundaries, find_boundaries
from scipy.ndimage.filters import gaussian_filter
from skelerator import Tree, Skeleton
from mahotas import cwatershed
import h5py
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt
import sys
import traceback
import time
import multiprocessing as mp

def create_segmentation(shape, n_objects, points_per_skeleton, interpolation, smoothness, write_to=None, seed=0):
    """
    Creates a toy segmentation containing skeletons.

    Args:

    shape: Size of the desired dataset
    
    n_objects: The number of skeleton/neurons to generate in the given volume

    points_per_skeleton: The number of potential branch points that are sampled per skeleton. 
                         Higher numbers lead to more complex shapes.

    interpolation: Method of interpolation between two sample points. Can be either linear or
                   random (constrained random walk).

    smoothness: Controls the smoothness of the initial noise map used to generate object boundaries.
    """
    try:
        shape = np.array(shape)
        if len(shape) != 3:
            raise ValueError("Provide 3D shape.")

        if np.any(shape % 2 != 0):
            raise ValueError("All shape dimensions have to be even.")

        noise = np.abs(np.random.randn(*shape))
        smoothed_noise = gaussian_filter(noise, sigma=smoothness)
        
        # Sample one tree for each object and generate its skeleton:
        max_dim = np.max(shape)
        double_max_dim = 2*max_dim
        seeds = np.zeros(2*np.array([max_dim]*len(shape)), dtype=int)

        pid = mp.current_process().pid
        seed = pid*3 + seed
        np.random.seed(seed)

        for i in range(n_objects):
            """
            We make the virtual volume twice as large to avoid border effects. To keep the density
            of points the same we also increase the number of points by a factor of 8 = 2**3. Such that
            on average we keep the same number of points per unit volume.
            """
            
            points = np.random.randint(0, double_max_dim, (3, 2**3*points_per_skeleton)).T
            tree = Tree(points)
            skeleton = Skeleton(tree, [1,1,1], "linear", generate_graph=False)
            seeds = skeleton.draw(seeds, np.array([0,0,0]), i + 1)

        """
        Cut the volume to original size (slice out middle of largevirtual volume).
        """
        seeds = seeds[int((double_max_dim-shape[0])/2):int((double_max_dim+shape[0])/2), int((double_max_dim-shape[1])/2):int((double_max_dim+shape[1])/2), int((double_max_dim-shape[2])/2):int((double_max_dim+shape[2])/2)]

        """
        We generate an artificial segmentation by first filtering
        skeleton points that are too close to each other via a non max supression
        to avoid artifacts. A distance transform of the skeletons plus smoothed noise
        is then used to calculate a watershed transformation with the skeletons as seeds
        resulting in the final segmentation.
        """
        seeds[maximum_filter(seeds, size=4) != seeds] = 0
        seeds_dt = distance_transform_edt(seeds==0) + 5. * smoothed_noise
        segmentation = cwatershed(seeds_dt, seeds)
        boundaries = find_boundaries(segmentation)

        if write_to is not None:
            f = h5py.File(write_to, "w")
            f.create_dataset("segmentation", data=segmentation.astype(np.uint64))
            f.create_dataset("skeletons", data=seeds.astype(np.uint64))
            f.create_dataset("boundaries", data=boundaries.astype(np.uint64))
            f.create_dataset("smoothed_noise", data=smoothed_noise)
            f.create_dataset("distance_transform", data=seeds_dt)

        data = {"segmentation": segmentation, "skeletons": seeds, "raw": boundaries}

    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    return data