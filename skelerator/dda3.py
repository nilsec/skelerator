import numpy as np
import operator

def dda_round(x):
    return (x + 0.5).astype(int) 

class DDA3:
    def __init__(self, start, end, scaling=np.array([1,1,1])):
        assert(start.dtype == int)
        assert(end.dtype == int)

        self.start = (start * scaling).astype(float)
        self.end = (end * scaling).astype(float)
        self.line = [dda_round(self.start)]
        
        self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))

        try:
            self.dv = (self.end - self.start) / self.max_length
        except RuntimeWarning:
            print("max length:", self.max_length, "\n")
            raise ValueError


    def draw(self): 
        for step in range(int(self.max_length)):
            self.line.append(dda_round((step + 1) * self.dv + self.start))

        return self.line
