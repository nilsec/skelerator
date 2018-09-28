import multiprocessing
import time
import numpy as np
from skelerator import create_segmentation

class Skelerator(object):
    def __init__(self, 
                 shape,
                 interpolation,
                 smoothness):

        self.shape = shape
        self.interpolation = interpolation
        self.smoothness = smoothness
        self.pool = multiprocessing.Pool()
        self.batch_x = []
        self.batch_y = []

    def next_batch(self, 
                   batch_size,
                   n_objects,
                   points_per_skeleton):
    
        self.batch_x = []    
        self.batch_y = []
        for batch in range(batch_size):
            self.pool.apply_async(create_segmentation, 
                                  args=(
                                      self.shape,
                                      n_objects,
                                      points_per_skeleton,
                                      self.interpolation,
                                      self.smoothness),
                                   callback=self.__catch_batch)
            #create_segmentation(self.shape, n_objects, points_per_skeleton, self.interpolation, self.smoothness)
        self.pool.close()
        self.pool.join()

        return np.stack(self.batch_x), np.stack(self.batch_y)

    def __catch_batch(self, batch):
        self.batch_x.append(batch["raw"].astype(np.uint64))
        self.batch_y.append(batch["skeletons"].astype(np.uint64))

if __name__ == "__main__":
    n_batches = 5

    bp = Skelerator([100,100,100],
                    "linear",
                    2.0)


    start = time.time()
    batch_x, batch_y = bp.next_batch(n_batches, 20, 5)
    end = time.time()
    elapsed = end - start
    
    print("Batch shapes: ")
    print(np.shape(batch_x), np.shape(batch_y))
    print("Batch types: ")
    print(type(batch_x), type(batch_y))
    print("Generating {} batches took {} seconds".format(n_batches, elapsed))
