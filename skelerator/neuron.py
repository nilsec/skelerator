import numpy as np
import graph_tool as gt
from graph_tool.search import BFSVisitor, bfs_search
import pdb
from skelerator.tree import Tree

class Neuron(Tree):
    def __init__(self, skeleton, min_radius, max_radius):
        """
        A neuron is a graph on 3d voxel grid (its skeleton/centerline)
        together with associated radii for each vertex indicaing
        the radius of a sphere at that point. Together they
        represent a volumetric description of a neuron. The radius of 
        two neighbouring voxels is constrained to be maximally
        different by 1. This leads to smooth changes of morhphology.
        """
        assert(min_radius>0)
        assert(max_radius>=min_radius)
        self.skeleton = skeleton
        self.g = self.skeleton.get_graph()
        self.source = self.skeleton.get_root_nodes()[0]
        self.points = skeleton.get_points()
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_vp = self.generate()

    def generate(self):
        print("Generate neuron radii...")
        rrg = RandomRadiusGenerator(self.skeleton,
                                    self.source,
                                    self.min_radius,
                                    self.max_radius)

        bfs_search(self.skeleton.get_graph(), 
                   self.source,
                   rrg)

        return rrg.radius_vp

    def get_radius(self, v):
        return int(self.radius_vp[v])

    def __get_bounding_box(self):
        min_point = np.min(self.points, axis=0)
        max_point = np.max(self.points, axis=0)
        
        # Note that this value can potentially be negative
        min_point_radius = min_point - (self.max_radius + 1)
        max_point_radius = max_point + (self.max_radius + 1)
        
        bounding_box = {"min": min_point_radius, "max": max_point_radius}
        return bounding_box


    def draw(self, canvas, offset):
        print("Draw neuron...")
        canvas_size = np.shape(canvas)
        xx,yy,zz = np.mgrid[:2*self.max_radius, :2*self.max_radius, :2*self.max_radius]

        for v in self.g.vertices():
            radius = self.get_radius(v)
            position = self.get_position(v) + offset

            x = position[0]
            y = position[1]
            z = position[2]

            rz = np.arange(z-self.max_radius,z+self.max_radius)
            rx = np.arange(y-self.max_radius,y+self.max_radius)
            ry = np.arange(x-self.max_radius,x+self.max_radius)

            sphere = (xx - self.max_radius)**2 + (yy - self.max_radius)**2 + (zz - self.max_radius)**2 <= radius**2

            canvas[z-self.max_radius:z+self.max_radius, 
                   y-self.max_radius:y+self.max_radius,
                   x-self.max_radius:x+self.max_radius] = np.logical_or(canvas[z-self.max_radius:z+self.max_radius, 
                                                                               y-self.max_radius:y+self.max_radius,
                                                                               x-self.max_radius:x+self.max_radius], sphere)
                                                            
        return np.array(canvas, dtype=int) * 255


    def get_minimal_canvas(self):
        bounding_box = self.__get_bounding_box()
        """
        Handle negative min bounding box
        by setting a new zero-point/offset
        to the absolute value of the negative min
        bounding box dimension
        """
        offset = []
        for d in bounding_box["min"][::-1]:
            if d<0:
                offset.append(abs(d))
            else:
                offset.append(0)

        # Gen canvas with all zeros
        canvas = np.zeros((bounding_box["max"]-bounding_box["min"])[::-1], dtype=int)
        return canvas, np.array(offset)



class RandomRadiusGenerator(BFSVisitor):
    """
    This class implements a breadth first search
    visitor that sets the radius of each vertex
    that is newly discoverd to a random value
    that varies by +- 1 or 0 from its neighbour
    while staying in the given bounds.
    """
    def __init__(self, skeleton, source, min_radius, max_radius):
        self.g = skeleton.get_graph()
        self.radius_vp = self.g.new_vertex_property("int", val=0)
        self.skeleton = skeleton
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.source = source

    def discover_vertex(self, v):
        if v == self.source:
            self.radius_vp[v] = np.random.choice(np.arange(self.min_radius, self.max_radius), 1)[0]
        else:
            nbs = self.skeleton.get_neighbours(v)

            non_zero = 0
            nb_radius = 0
            for u in nbs:
                radius = self.radius_vp[u]
                assert(radius >= 0)
                assert(radius <= self.max_radius)
                if radius > 0:
                    non_zero += 1
                    nb_radius = radius

            assert(non_zero == 1)
            assert(nb_radius>=self.min_radius)
            assert(nb_radius<=self.max_radius)

            if nb_radius == self.min_radius:
                self.radius_vp[v] = nb_radius + 1

            elif nb_radius == self.max_radius:
                self.radius_vp[v] = nb_radius - 1

            else:
                self.radius_vp[v] = nb_radius + np.random.choice(np.array([-1,0,1]), 1)[0]
