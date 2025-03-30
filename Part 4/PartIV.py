import math
import heapq

#A star implementation
#copied and modified from Part II Dijkstra's (credit to jinny!)

class Node:
    def __init__(self, value, key=float('inf')):
        self.value = value   
        self.key = key 
    
    def __lt__(self, other):
        return self.key < other.key

def A_Star(graph, source, destination, heuristic): 
    #assumed heuristic to be a dict, change to round parenthesis for function

    #Initialization
    dist = {node: float('inf') for node in graph.adj}
    prev = {node: None for node in graph.adj}
    dist[source] = 0
    pq = [Node(source, heuristic[source])]#f's are computed as needed
    
    seen = set()

    while pq:
        u = heapq.heappop(pq).value
        if u in seen: #dijkstra processes each node once
            continue
        
        if u == destination:
            return prev, dist[u]
        
        seen.add(u)

        for v in graph.adj[u]:
            alt = dist[u] + graph.weights[(u, v)]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                if v in seen: #limits duplicates in the pq
                    continue
                #might still have dupes but need to make sure smallest f is in pq
                #so it's like a primitive decrease_key lol
                heapq.heappush(pq,Node(v,alt + heuristic[v]))
                

    return math.inf 

#graph class (directed weighted)
class Graph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)
    
    def has_edge(self,src,dst):
        return dst in self.adj[src]