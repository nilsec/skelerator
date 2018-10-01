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
                #self.queue_next_batch(batch_size, n_objects, points_per_skeleton, seed + i)

        batch = self.queue.get()
        return batch

    def queue_next_batch(self, 
                         batch_size,
                         n_objects,
                         points_per_skeleton):

        self.worker_queue.put(1)

        if self.verbose:
            print("Worker started")

        batch_raw = []
        batch_skel = []
        batch_seg = []
        batch_raw_out = []
        batch_skel_out = []
        batch_seg_out = []

        k = 0
        for batch in range(batch_size):
            batch = create_segmentation(self.shape, n_objects, points_per_skeleton, self.interpolation, self.smoothness, seed=int(time.time()/((k+1)*3)))
            batch_raw.append(batch["raw"].astype("bool"))
            batch_skel.append(batch["skeletons"].astype("bool"))
            batch_seg.append(batch["segmentation"].astype(np.uint32))
            batch_raw_out.append(self.crop(batch["raw"]).astype("bool"))
            batch_skel_out.append(self.crop(batch["skeletons"].astype("bool")))
            batch_seg_out.append(self.crop(batch["segmentation"].astype(np.uint32)))
            k += 1

        if self.verbose:
            print("Add batch to queue...")

        self.queue.put([np.stack(batch_raw), np.stack(batch_skel), np.stack(batch_seg), 
                        np.stack(batch_raw_out), np.stack(batch_skel_out), np.stack(batch_seg_out)])
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

    for i in range(5):
        batch = bp.next_batch(2, 10, 5)
        f = h5py.File("./test_crop_{}.h5".format(i))
        f.create_dataset("x", data=batch[0][0])
        dset = f.create_dataset("y", data=batch[-1][0].astype(np.uint32))
        dset.attrs.create("offset", np.array([25,25,25]))
    bp.finished()
