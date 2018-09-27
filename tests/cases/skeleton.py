import unittest
import numpy as np

from tree import SmallTreeTestCase
from skelerator.skeleton import Skeleton
from skelerator.tree import Tree

class SkeletonTestCaseLinear(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1,1,1], "linear")
        
        tree_points = tree.get_points()
        skeleton_points = skeleton.get_points()
        for point in tree_points:
            self.assertTrue(point in skeleton_points)

        skeleton_graph_points = []
        for v in skeleton.get_vertex_iterator():
            pos = skeleton.get_position(v)
            skeleton_graph_points.append(pos)
            self.assertTrue(pos in skeleton_points)

        for p in skeleton_points:
            self.assertTrue(p in np.array(skeleton_graph_points))

        root_nodes_tree = tree.get_root_nodes()
        root_nodes_skeleton = skeleton.get_root_nodes()
        for v in root_nodes_tree:
            self.assertTrue(v in root_nodes_skeleton)
        for v  in root_nodes_skeleton:
            self.assertTrue(v in root_nodes_tree)

        incident_edges = []
        for v in skeleton.get_vertex_iterator():
            incident_edges.append(skeleton.get_incident_edges(v))

        n_leaves = len([1 for es in incident_edges if len(es) == 1])
        n_branches = len([1 for es in incident_edges if len(es) == 3])
        self.assertEqual(n_leaves, self.expected_leaves)
        self.assertEqual(n_branches, self.expected_branches)
        skeleton.to_nml("./small_skeleton_linear.nml")
        
class SkeletonTestCaseRandom(SmallTreeTestCase):
    def runTest(self):
        tree = Tree(self.points)
        skeleton = Skeleton(tree, [1,1,1], "random")
        
        tree_points = tree.get_points()
        skeleton_points = skeleton.get_points()
        for point in tree_points:
            self.assertTrue(point in skeleton_points)

        skeleton_graph_points = []
        for v in skeleton.get_vertex_iterator():
            pos = skeleton.get_position(v)
            skeleton_graph_points.append(pos)
            self.assertTrue(pos in skeleton_points)

        for p in skeleton_points:
            self.assertTrue(p in np.array(skeleton_graph_points))

        root_nodes_tree = tree.get_root_nodes()
        root_nodes_skeleton = skeleton.get_root_nodes()
        for v in root_nodes_tree:
            self.assertTrue(v in root_nodes_skeleton)
        for v  in root_nodes_skeleton:
            self.assertTrue(v in root_nodes_tree)

        incident_edges = []
        for v in skeleton.get_vertex_iterator():
            incident_edges.append(skeleton.get_incident_edges(v))

        n_leaves = len([1 for es in incident_edges if len(es) == 1])
        n_branches = len([1 for es in incident_edges if len(es) == 3])
        self.assertEqual(n_leaves, self.expected_leaves)
        self.assertEqual(n_branches, self.expected_branches)
        skeleton.to_nml("./small_skeleton_random.nml")
 
if __name__ == "__main__":
    unittest.main()
