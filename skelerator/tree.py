import graph_tool as gt
gt.openmp_set_num_threads(1)
from graph_tool.generation import triangulation
from graph_tool.topology import min_spanning_tree
import numpy as np
from xml.dom import minidom

class Tree(object):
    def __init__(self, points, verbose=False):
        """
        A tree is a minimal spanning tree
        between a list of points in 3D.
        """
        assert(type(points) == np.ndarray)
        assert(points.dtype==int)
        assert(points.shape[1] == 3)
        assert(points.shape[0]>1)

        # Make points unique to avoid duplicate vertices:
        self.points = np.unique(points, axis=0)
        self.verbose = verbose
        self.g = self.__generate()

    def get_root_nodes(self):
        root_nodes = []
        for v in self.get_vertex_iterator():
            if len(self.get_incident_edges(v)) == 1:
                root_nodes.append(v)

        assert(len(root_nodes)>=2)
        return root_nodes

    def get_points(self):
        return self.points

    def to_nml(self, path):
        doc = minidom.Document()
        annotations = doc.createElement("things")
        doc.appendChild(annotations)

        n = 1
        nodes = doc.createElement("nodes")
        vertex_map = {}
        for v in self.g.vertices():
            node = doc.createElement("node")
            position = np.array(self.g.vertex_properties["position"][v], dtype=int)
            self.__build_attributes(node, [["x", position[0]],
                                           ["y", position[1]],
                                           ["z", position[2]],
                                           ["id", n]])
            vertex_map[v] = n
            nodes.appendChild(node)
            n += 1

        edges = doc.createElement("edges")
        for e in self.g.edges():
            source_id = vertex_map[e.source()]
            target_id = vertex_map[e.target()]

            edge = doc.createElement("edge")
            self.__build_attributes(edge, [["source", source_id],
                                           ["target", target_id]])
            edges.appendChild(edge)

        annotation = doc.createElement("thing")
        annotation.appendChild(nodes)
        annotation.appendChild(edges)

        annotations.appendChild(annotation)
        doc = doc.toprettyxml()

        with open(path, "w+") as f:
            f.write(doc)

    def __build_attributes(self, xml_elem, attributes):
        for attr in attributes:
            try:
                xml_elem.setAttribute(attr[0], str(attr[1]))
            except UnicodeEncodeError:
                xml_elem.setAttribute(attr[0], str(attr[1].encode('ascii', 'replace')))
        return xml_elem


    def get_position(self, v):
        return np.array(self.g.vertex_properties["position"][v], dtype=int)

    def get_incident_edges(self, v):
        edges = self.g.get_out_edges(v)
        edges = np.array(sorted(edges, key=lambda x: x[2]))
        return [self.get_edge(e[0], e[1]) for e in edges]

    def get_edge_array(self):
        return self.g.get_edge_array()

    def get_edge(u, v):
        return self.g.get_edge(u,v)

    def get_neighbours(self, v):
        incident_edges = self.get_incident_edges(v)
        nbs = set()
        for e in incident_edges:
            nbs.add(int(e.source()))
            nbs.add(int(e.target()))
        nbs.remove(int(v))
        return list(nbs)

    def get_edge(self, u, v):
        edges = self.g.edge(u,v, all_edges=True, add_missing=False)
        assert(len(edges)<=1)
        try:
            return edges[0]
        except IndexError:
            raise KeyError("Nonexistent edge: ({},{})".format(u,v))

    def get_number_of_vertices(self):
        return self.g.num_vertices()

    def get_number_of_edges(self):
        return self.g.num_edges()

    def get_vertex_iterator(self):
        return self.g.vertices()

    def get_edge_iterator(self):
        return self.g.edges()

    def get_vertex_array(self):
        return self.g.get_vertices()

    def get_edge_array(self):
        return self.g.get_edges()

    def set_edge_filter(self, ep):
        self.g.set_edge_filter(ep)

    def get_components(self):
        components = self.g.get_components(min_vertices=0, output_folder=None)
        return components

    def __generate(self):
        if self.verbose:
            print("Generate tree...")
        g = self.__gen_delaunay_graph()
        distance_weights = self.__get_distance_weights(g)
        tree = self.__get_minimal_spanning_tree(g, weights=distance_weights)
        return tree

    def __gen_delaunay_graph(self):
        if self.verbose:
            print("Generate delaunay triangulation of unique points...")
        g, pos = triangulation(self.points, type="delaunay")
        g.vertex_properties["position"] = pos
        return g

    def __get_distance_weights(self, g):
        if self.verbose:
            print("Generate edge weights...")
        weights = g.new_edge_property("double")
        for e in g.edges():
            pos_source = np.array(g.vertex_properties["position"][e.source()], dtype=float) 
            pos_target = np.array(g.vertex_properties["position"][e.target()], dtype=float) 
            weights[e] = np.linalg.norm(pos_source - pos_target)
        return weights

    def __get_minimal_spanning_tree(self, g, weights=None):
        if self.verbose:
            print("Get minimal spanning tree...")
        tree_map = min_spanning_tree(g, weights=weights)
        g.set_edge_filter(tree_map)
        return g
