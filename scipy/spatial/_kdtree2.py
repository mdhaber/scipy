import numpy as np

class kdtree2:
    def __init__(self, data):
        n, k = data.shape
        self.data = data
        self.children = np.full((n, 2), -1)
        self.parent = np.full(n, -1)
        self.axis = np.full(n, -1)
        self.nodes = np.arange(n)
        self.k = k
        self.n = n
        i_median = self.n // 2
        self.nodes = np.argpartition(self.data[:, 0], i_median)
        self.root = self.nodes[i_median]
        self.axis[self.root] = 0
        self.construct(self.nodes[:i_median], self.nodes[i_median+1:], self.root, 1)

    def construct(self, left, right, node, level):
        d = level % self.k

        if len(left) > 1:
            i_left = len(left) // 2
            left[:] = left[np.argpartition(self.data[left, d], i_left)]
            left_node = left[i_left]
            self.axis[left_node] = d
            self.construct(left[:i_left], left[i_left+1:], left_node, level + 1)
            self.children[node, 0] = left_node
        elif len(left) == 1:
            self.axis[left[0]] = d
            self.children[node, 0] = left[0]

        if len(right) > 1:
            i_right = len(right) // 2
            right[:] = right[np.argpartition(self.data[right, d], i_right)]
            right_node = right[i_right]
            self.axis[right_node] = d
            self.construct(right[:i_right], right[i_right+1:], right_node, level + 1)
            self.children[node, 1] = right_node
        elif len(right) == 1:
            self.axis[right[0]] = d
            self.children[node, 1] = right[0]

    def distance(self, point1, point2):
        return np.sum((point1 - point2)**2)

    def nearest_neighbor(self, point):
        node, distance = self._nearest_neighbor(point, self.root)
        return node, distance

    def _nearest_neighbor(self, point, node):
        axis = self.axis[node]

        if node == -1:
            return node, np.inf

        best_node = node
        best_distance = self.distance(point, self.data[node])

        right = int(point[axis] > self.data[node, axis])
        right_node, right_distance = self._nearest_neighbor(point, self.children[node, right])
        if right_distance < best_distance:
            best_node = right_node
            best_distance = right_distance

        other_node = self.children[node, 1-right]
        if (other_node >= 0) and (abs(point[axis] - self.data[node, axis]) > best_distance**0.5):
            return best_node, best_distance

        other_node, other_distance = self._nearest_neighbor(point, self.children[node, 1-right])
        if other_distance < best_distance:
            best_node = other_node
            best_distance = other_distance

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
