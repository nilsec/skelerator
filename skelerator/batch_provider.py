import time
import numpy as np
import  multiprocessing
from skelerator import create_segmentation
import pdb
import h5py


class BatchProvider(object):
    def __init__(self, 
                 shape_in,
                 shape_out,
                 interpolation,
                 smoothness,
                 n_workers=8,
                 verbose=False):

        self.shape = np.array(shape_in)
        self.shape_out = np.array(shape_out)
        if np.any(self.shape-self.shape_out < 0):
            raise ValueError("Output shape needs to be smaller than input shape")
        if np.any((self.shape - self.shape_out) % 2 != 0):
            raise ValueError("Input shape minus output shape must be divisible by 2 in all dimensions")
        self.interpolation = interpolation
        self.smoothness = smoothness
        self.n_workers = n_workers
        self.verbose = verbose

        self.queue = multiprocessing.Queue(50)
        self.worker_queue = multiprocessing.Queue(n_workers)
        self.done = False
        self.processes = []

    def next_batch(self,
                   batch_size,
                   n_objects,
                   points_per_skeleton):

        if self.verbose:
            print("Request batch...")
        if not self.queue.full() and not self.worker_queue.full():
            if self.verbose:
                print("Queue not full, spawn workers...")
            for i in range(self.n_workers):
                p = multiprocessing.Process(target=self.queue_next_batch, args=(batch_size, n_objects, points_per_skeleton,))
                p.start()
                self.processes.append(p)

        batch = self.queue.get()
        return batch

    def queue_next_batch(self, 
                         batch_size,
                         n_objects,
                         points_per_skeleton):

        self.worker_queue.put(1)

        if self.verbose:
            print("Worker started")
        batch_x = []
        batch_y = []
        batch_y_out = []

        for batch in range(batch_size):
            batch = create_segmentation(self.shape, n_objects, points_per_skeleton, self.interpolation, self.smoothness)
            batch_x.append(batch["raw"].astype("bool"))
            batch_y.append(batch["skeletons"].astype("bool"))
            batch_y_out.append(self.crop(batch["skeletons"]).astype("bool"))

        if self.verbose:
            print("Add batch to queue...")
        if not self.done:
            self.queue.put([np.stack(batch_x), np.stack(batch_y), np.stack(batch_y_out)])
            self.worker_queue.get()

    def crop(self, y):
        lower = ((self.shape - self.shape_out)/2).astype(int)
        upper = self.shape - lower
        y = y[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        return y
            


    def finished(self):
        if self.verbose:
            print("Batch generation finished. Exiting.")
        self.done = True
        for p in self.processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    bp = BatchProvider([100,100,100],
                       [50, 50, 50],
                       "linear",
                       2.0)

    batch = bp.next_batch(2, 10, 5)
    bp.finished()

    f = h5py.File("./test_crop.h5")
    f.create_dataset("x", data=batch[0][0])
    f.create_dataset("y", data=batch[1][0])
    dset = f.create_dataset("y_out", data=batch[2][0])
    dset.attrs.create("offset", np.array([25,25,25]))
    pdb.set_trace()

