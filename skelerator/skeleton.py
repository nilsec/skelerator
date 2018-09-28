import graph_tool as gt
import numpy as np

from skelerator.dda3 import DDA3
from skelerator.crw import ConstrainedRandomWalk
from skelerator.tree import Tree

class Skeleton(Tree):
    def __init__(self, tree, scaling, interpolation, verbose=False):
        """
        A skeleton is a graph on a 3D voxel grid where each voxel is encoded by
        a vertex.
        """
        self.tree = tree
        self.scaling = scaling
        self.verbose = verbose

        self.points, self.edge_to_line = self.__generate(interpolation)
        self.g = self.__to_graph(self.points, self.edge_to_line)

    def get_tree(self):
        return self.tree

    def get_scaling(self):
        return self.scaling

    def get_points(self):
        return self.points

    def get_graph(self):
        return self.g

    def draw(self, canvas, offset, label):
        if self.verbose:
            print("Draw skeleton...")
        canvas_size = np.shape(canvas)

        for v in self.g.vertices():
            position = self.get_position(v) + offset
            x = position[0]
            y = position[1]
            z = position[2]

            try:
                canvas[z, y, x] = label
            except IndexError:
                print("WARNING: Provided canvas is too small to draw all skeleton points.")

        return np.array(canvas, dtype=int)

    def __generate(self, interpolation):
        """
        Interpolate each edge linearly or randomly
        in voxel space with appropriate
        scaling.
        """
        if not interpolation in ["linear", "random"]:
            raise ValueError("Choose between random or linear interpolation")
        if self.verbose:
            print("Interpolate edges {}...".format(interpolation))

        points = []
        edge_to_line = {}
        for e in self.tree.get_edge_iterator():
            start = self.tree.get_position(e.source())
            end = self.tree.get_position(e.target())

            if interpolation == "linear":
                dda = DDA3(start, end, self.scaling)
                line = dda.draw()
            else:
                if not np.all(self.scaling == np.array([1,1,1])):
                    raise NotImplementedError("For random interpolation no scaling is supported")
                rw = ConstrainedRandomWalk(start, end)
                line = rw.walk()
                
            points.extend(line)
            edge_to_line[e] = line

        points_unique = np.unique(points, axis=0) 
        return points_unique, edge_to_line


    def __to_graph(self, points_unique, edge_to_line):
        """
        Convert point interpolation to
        gt graph to preserve neighborhood
        information.
        """
        if self.verbose:
            print("Initialize skeleton graph...")
        g = gt.Graph(directed=False)
        g.add_vertex(len(points_unique))
        vp_pos = g.new_vertex_property("vector<int>")
        g.vertex_properties["position"] = vp_pos

        """
        Initialize original tree vertices
        as fixed points.
        """
        vg = 0
        pos_to_vg = {}
        for v in self.tree.get_vertex_iterator():
            v_pos = self.tree.get_position(v)
            g.vertex_properties["position"][vg] = v_pos
            pos_to_vg[tuple(v_pos)] = vg
            vg += 1

        """
        Add edge interpolation in between each 
        fixed tree vertex.
        """
        for e in self.tree.get_edge_iterator():
            points = edge_to_line[e]
            start_vertex = pos_to_vg[tuple(points[0])]
            end_vertex = pos_to_vg[tuple(points[-1])]
            for pos in points[1:-1]:
                g.add_edge(start_vertex, vg)
                g.vertex_properties["position"][vg] = pos
                start_vertex = vg
                vg += 1
            g.add_edge(start_vertex, end_vertex)

        return g
