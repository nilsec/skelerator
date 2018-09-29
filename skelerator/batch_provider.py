import time
import numpy as np
import  multiprocessing
from skelerator import create_segmentation
import pdb
import h5py


class BatchProvider(object):
    def __init__(self, 
                 shape,
                 interpolation,
                 smoothness,
                 n_workers=8,
                 verbose=False):

        self.shape = shape
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

        for batch in range(batch_size):
            batch = create_segmentation(self.shape, n_objects, points_per_skeleton, self.interpolation, self.smoothness)
            batch_x.append(batch["raw"].astype("bool"))
            batch_y.append(batch["skeletons"].astype("bool"))

        if self.verbose:
            print("Add batch to queue...")
        if not self.done:
            self.queue.put([np.stack(batch_x), np.stack(batch_y)])
            self.worker_queue.get()

    def finished(self):
        if self.verbose:
            print("Batch generation finished. Exiting.")
        self.done = True
        for p in self.processes:
            p.terminate()
            p.join()
