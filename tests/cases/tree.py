import unittest
import numpy as np
from skelerator import Tree

class SmallTreeTestCase(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0,0,0], 
                                [0,0,0], 
                                [0,100,0], 
                                [0,500,0], 
                                [0,800,0],
                                [0,500,100]])

        self.unique_points = 5
        self.expected_edges = 4
        self.expected_branches = 1
        self.expected_leaves = 3

class TreeGenTestCase(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        self.assertEqual(tree.get_number_of_vertices(), self.unique_points)
        self.assertEqual(tree.get_number_of_edges(), self.expected_edges)

        incident_edges = []
        for v in tree.get_vertex_iterator():
            incident_edges.append(tree.get_incident_edges(v))

        n_leaves = len([1 for es in incident_edges if len(es)==1])
        n_branches = len([1 for es in incident_edges if len(es)==3])
        self.assertEqual(n_leaves, self.expected_leaves)
        self.assertEqual(n_branches, self.expected_branches)
        tree.to_nml("./small_tree.nml")

if __name__ == "__main__":
    unittest.main()
