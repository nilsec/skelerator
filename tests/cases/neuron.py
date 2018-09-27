import unittest
import numpy as np

from tree import SmallTreeTestCase
from skelerator import Skeleton
from skelerator import Tree
from skelerator import Neuron
import h5py
import pdb

class NeuronTestCaseLinear(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1, 1, 1], "linear")

        self.min_radius = 5
        self.max_radius = 50
        neuron = Neuron(skeleton, 
                        self.min_radius, 
                        self.max_radius)

        for v in neuron.get_vertex_iterator():
            radius_v = neuron.get_radius(v)
            self.assertTrue(radius_v<=self.max_radius)
            self.assertTrue(radius_v>=self.min_radius)
            
            nbs = neuron.get_neighbours(v)
            for u in nbs:
                radius_u = neuron.get_radius(u)
                self.assertTrue(np.abs(radius_u - radius_v)<=1)

class NeuronTestCaseRandom(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1, 1, 1], "random")

        self.min_radius = 5
        self.max_radius = 50
        neuron = Neuron(skeleton, 
                        self.min_radius, 
                        self.max_radius)

        for v in neuron.get_vertex_iterator():
            radius_v = neuron.get_radius(v)
            self.assertTrue(radius_v<=self.max_radius)
            self.assertTrue(radius_v>=self.min_radius)
            
            nbs = neuron.get_neighbours(v)
            for u in nbs:
                radius_u = neuron.get_radius(u)
                self.assertTrue(np.abs(radius_u - radius_v)<=1)

class DrawTestCaseLinear(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1,1,1], "linear")
        neuron = Neuron(skeleton, 5, 50)
        canvas, offset = neuron.get_minimal_canvas()
        canvas = neuron.draw(canvas, offset)

        f = h5py.File("./small_neuron_linear.h5", "w")
        f.create_dataset("neuron", data=canvas)
        f.close()

class DrawTestCaseRandom(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1,1,1], "random")
        neuron = Neuron(skeleton, 5, 50)
        canvas, offset = neuron.get_minimal_canvas()
        canvas = neuron.draw(canvas, offset)

        f = h5py.File("./small_neuron_random.h5", "w")
        f.create_dataset("neuron", data=canvas)
        f.close()

        
if __name__ == "__main__":
    unittest.main()
