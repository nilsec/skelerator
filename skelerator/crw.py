import numpy as np
import matplotlib.pyplot as plt

class ConstrainedRandomWalk(object):
    def __init__(self, start, end):
        """
        Interpolate two points on a 2D or 3D
        voxel grid by taking random moves from 
        the start point to any 
        of the neighboring voxels that decrease
        the distance to the end point.
        """
        self.start = start
        self.end = end

        if len(self.start) == len(self.end) == 3:
            self.adjacency = [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if not (i == j == k == 0)]
        elif len(self.start) == len(self.end) == 2:
            self.adjacency = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if not (i == j == 0)]
        else:
            raise NotImplementedError("Provide 2 or 3 dimensional start & endpoints")

    def walk(self):
        line = []
        line.append(np.copy(self.start))
        p = np.copy(self.start)

        while not np.all(p == self.end):
            possible_moves = [adj for adj in self.adjacency if (np.linalg.norm((p + adj) - self.end) < np.linalg.norm(p - self.end))]
            if not possible_moves:
                raise ValueError("No possible moves")

            move = possible_moves[np.random.randint(0, len(possible_moves))]
            p += move
            line.append(np.copy(p))

        assert(np.all(line[0] == self.start))
        assert(np.all(line[-1] == self.end))
        return line

    def show(self, line):
        if len(line[0]) != 2:
            raise ValueError("Can only visualize 2d lines")
        x = [j[0] for j in line]
        y = [j[1] for j in line]
        plt.scatter(x, y)
        plt.show()
