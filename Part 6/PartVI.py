# sp_algorithm.py
class SPAlgorithm:
    def calc_sp(self, graph, source: int, dest: int):
        raise NotImplementedError("Subclasses should implement this method")


# short_path_finder.py
class ShortPathFinder:
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def set_graph(self, graph):
        self.graph = graph

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source: int, dest: int):
        if not self.graph or not self.algorithm:
            raise Exception("Graph and algorithm must be set first.")
        return self.algorithm.calc_sp(self.graph, source, dest)


# dijkstra.py
from sp_algorithm import SPAlgorithm
class Dijkstra(SPAlgorithm):
    def __init__(self, k: int):
        self.k = k

    def calc_sp(self, graph, source: int, dest: int):
        dist = {node: float('inf') for node in graph.adj}
        prev = {node: None for node in graph.adj}
        relax_count = {node: 0 for node in graph.adj}
        dist[source] = 0
        pq = MinHeap([Node(node, dist[node]) for node in graph.adj])

        while not pq.is_empty():
            u = pq.extract_min().value
            if relax_count[u] >= self.k:
                continue
            for v in graph.adj[u]:
                alt = dist[u] + graph.weights[(u, v)]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    pq.decrease_key(v, alt)
                    relax_count[v] += 1

        return dist, prev


# bellman_ford.py
from sp_algorithm import SPAlgorithm
class BellmanFord(SPAlgorithm):
    def __init__(self, k: int):
        self.k = k

    def calc_sp(self, graph, source: int, dest: int):
        dist = {node: float('inf') for node in graph.adj}
        prev = {node: None for node in graph.adj}
        dist[source] = 0

        for _ in range(self.k):
            for u in graph.adj:
                for v in graph.adj[u]:
                    if dist[u] + graph.w(u, v) < dist[v]:
                        dist[v] = dist[u] + graph.w(u, v)
                        prev[v] = u

        return dist, prev


# a_star.py
import math
import heapq
from sp_algorithm import SPAlgorithm

class Node:
    def __init__(self, value, key=float('inf')):
        self.value = value
        self.key = key

    def __lt__(self, other):
        return self.key < other.key

class AStar(SPAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def calc_sp(self, graph, source: int, dest: int):
        dist = {node: float('inf') for node in graph.adj}
        prev = {node: None for node in graph.adj}
        dist[source] = 0
        pq = [Node(source, self.heuristic[source])]
        seen = set()

        while pq:
            u = heapq.heappop(pq).value
            if u in seen:
                continue
            if u == dest:
                break
            seen.add(u)

            for v in graph.adj[u]:
                alt = dist[u] + graph.weights[(u, v)]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    if v in seen:
                        continue
                    heapq.heappush(pq, Node(v, alt + self.heuristic[v]))

        return dist, prev


