import numpy as np

class kdtree2:
    def __init__(self, data):
        self.data = data
        self.n, self.k = data.shape
        self.children = np.full((self.n, 2), -1)
        self.axis = np.full(self.n, -1)
        self.root = self.construct(np.arange(self.n), 0)

    def construct(self, indices, level):
        n = len(indices)
        if n == 0:
            return -1

        axis = level % self.k
        if n == 1:
            node = indices[0]
            self.axis[node] = axis
            return node

        i = n // 2
        indices = indices[np.argpartition(self.data[indices, axis], i)]

        node = indices[i]
        self.axis[node] = axis

        # construct children
        next_level = level + 1
        left = self.construct(indices[:i], next_level)
        right = self.construct(indices[i + 1:], next_level)
        self.children[node] = left, right

        return node


    def distance(self, point1, point2):
        return np.linalg.vector_norm(point1 - point2)

    def nearest_neighbor(self, point):
        node, distance = self._nearest_neighbor(point, self.root)
        return node, distance

    def _nearest_neighbor(self, point, node):
        if node == -1:
            return node, np.inf

        axis = self.axis[node]
        splitting_plane_distance = point[axis] - self.data[node, axis]
        if splitting_plane_distance < 0:
            near, far = self.children[node]
        else:
            far, near = self.children[node]

        # Distance to closest near-side node
        best_node, best_distance = self._nearest_neighbor(point, near)

        # Distance to current node
        this_distance = self.distance(point, self.data[node])
        if this_distance < best_distance:
            best_node, best_distance = node, this_distance

        # Distance to closest far-side node
        if abs(splitting_plane_distance < best_distance):
            far_node, far_distance = self._nearest_neighbor(point, far)
            if far_distance < best_distance:
                best_node, best_distance = far_node, far_distance

        return best_node, best_distance


    def find(self, point):
        node = self.root
        while True:
            axis = self.axis[node]
            if point[axis] == self.data[node, axis]:
                return node, -1

            right = int(point[axis] > self.data[node, axis])
            if self.children[node, right] < 0:
                return node, right
            else:
                node = self.children[node, right]
