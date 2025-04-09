from abc import ABC, abstractmethod
import math
import heapq
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
            raise Exception("Please set the graph and algorithm!")
        return self.algorithm.calc_sp(self.graph, source, dest)

#interface:
class Graph():
    @abstractmethod
    def get_adj_nodes(self,node:int):
        pass
    
    @abstractmethod
    def add_node(self, node):
        pass

    @abstractmethod
    def add_edge(self, start:int, end:int, w: float):
        pass

    @abstractmethod
    def get_num_of_nodes(self):
        pass

    @abstractmethod
    def w(self, node):
        pass

class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()
        self.adj = {}
        self.weights = {}
    
    def get_adj_nodes(self,node:int):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, start:int, end:int, w: float):
        if end not in self.adj[start]:
            self.adj[start].append(end)
        self.weights[(start, end)] = w

    def get_num_of_nodes(self):
        return len(self.adj)
    
    def w(self, node1, node2):
        if (node1, node2) in self.weights:
            return self.weights[(node1,node2)]

class HeuristicGraph(WeightedGraph):
    def __init__(self):
        super().__init__()
        self.heuristic = {}
    
    def set_heuristic(self, h):
        self.heuristic = h
    
    def find_heuristic(self, f):
        for i in range(len(self.adj)):
            self.heuristic[i] = f(i)
    
    def get_heuristic(self):
        return self.heuristic

#interface:
class SPAlgorithm:
    @abstractmethod
    def calc_sp(self, graph: Graph, source: int, dest: int):
        pass

class Dijkstra(SPAlgorithm):
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int):
        #Initialization
        dist = {}
        prev = {}
        for node in range(graph.get_num_of_nodes()):
            dist[node] = math.inf
            prev[node] = None
        dist[source] = 0
        pq = MinHeap([Node(node, dist[node]) for node in graph.graph])

        while not pq.is_empty():
            u = pq.extract_min().value
            
            if u == dest:
                return dist[dest]

            for v in graph.get_adj_nodes(u):
                alt = dist[u] + graph.w(u, v)
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    pq.decrease_key(v, alt)

        return math.inf #PATH NOT FOUND

class Bellman_Ford(SPAlgorithm):
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int):
        dist = {}
        prev = {}
        for i in range(graph.get_num_of_nodes()):
            dist[i] = math.inf
            prev[i] = None
        dist[source] = 0

        for _ in range(graph.get_num_of_nodes() -1):
            for u in range(graph.get_num_of_nodes()):
                for v in graph.get_adj_nodes(u):
                    if dist[u] + graph.w(u, v) < dist[v]:
                        dist[v] = dist[u] + graph.w(u, v)
                        prev[v] = u

        return dist[dest]

class A_Star(SPAlgorithm):
    def calc_sp(self, graph: HeuristicGraph, source: int, dest: int):
        #Initialization
        dist = {}
        prev = {}
        for node in range(graph.get_num_of_nodes()):
            dist[node] = math.inf
            prev[node] = None
        dist[source] = 0
        h = graph.get_heuristic()
        pq = [(h[source], source)]#f's are computed as needed
        # streamlined the comparison with how Python auto compares first value in a tuple first
        
        seen = set()

        while pq:
            _, u = heapq.heappop(pq)
            if u in seen: #dijkstra processes each node only once
                continue #only added because of the awkward way i used heapq ;-;
            
            if u == dest:
                return dist[dest]
            
            seen.add(u)

            for v in graph.get_adj_nodes(u):
                alt = dist[u] + graph.w(u, v)
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    if v in seen: #limits duplicates in the pq
                        continue
                    #might still have dupes but need to make sure smallest f is in pq
                    #so it's like a primitive decrease_key lol
                    heapq.heappush(pq, (h[v], v))
                    
        return math.inf #PATH NOT FOUND


#--------------------------------------------------------------------------------------------
# other useful stuff
class MinHeap:
    def __init__(self, data):
        self.items = data 
        self.length = len(data) 
        self.map = {} 
        self.build_heap()  
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self, index):
        return 2 * (index + 1) - 1

    def find_right_index(self, index):
        return 2 * (index + 1)

    def find_parent_index(self, index):
        return (index + 1) // 2 - 1

    def heapify(self, index):
        smallest_known_index = index
        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)
        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)
        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index
            self.heapify(smallest_known_index) 

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1): 
            self.heapify(i)

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node) 
        else:
            self.items[self.length] = node  
        self.map[node.value] = self.length  
        self.length += 1 
        self.swim_up(self.length - 1)  

    def extract_min(self):
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0
        min_node = self.items[self.length - 1]
        self.length -= 1 
        self.map.pop(min_node.value)  
        self.heapify(0) 
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]  
        self.items[index].key = new_key 
        self.swim_up(index)  

    def swim_up(self, index):
        while index > 0 and self.items[index].key < self.items[self.find_parent_index(index)].key: 
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)  

    def is_empty(self):
        return self.length == 0
    
class Node:
    def __init__(self, value, key=float('inf')):
        self.value = value
        self.key = key