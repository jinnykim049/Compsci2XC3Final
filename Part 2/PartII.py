import matplotlib
import numpy
import random
import math

#utility functions
#draw_plot()

#non-negative weights connected graph generator
#NOT FOR THE CORRECT GRAPH CLASS WILL FIX LATER
def create_random_graph(nodes, edges):
    # if edges exceed maximum, infinite loop
    max = nodes*(nodes-1)//2 #formula for undirected graph max edge number
    if edges > max:
        edges = max
    
    graph = Graph(nodes)
    
    #ensure it is connected
    for i in range(nodes-1):
        graph.add_edge(i,i+1,random.randint(0,100)) # picked arbitrary max weight 
    
    for _ in range(edges-nodes+1): #node-1 number of edges have already been added
        src = random.randint(0, nodes-1)
        dst = random.randint(0, nodes-1)
        while graph.has_edge(src, dst) or src==dst: #verifying valid src dst, if not change it
            src = random.randint(0, nodes-1)
            dst = random.randint(0, nodes-1)
            
        graph.add_edge(src, dst, random.randint(0,100))

    return graph

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


#2.1 Dijkstra's
def dijkstra(graph, source, k):

    #implementation

    return


#2.2 Bellman Ford's
def bellman_ford(graph, source, k):

    #initialization
    dist = {}
    prev = {}
    
    for i in graph.adj:
        dist[i] = math.inf
        prev[i] = None
        
    dist[source] = 0
    
    #edge relaxation
    for _ in range(k): #only k times
        
        for src in graph.adj:
            for dst in graph.adj[src]: #two for loops together gets every edge (src, dst)
                
                # if shorter path found, update dist and prev
                if dist[src] + graph.w(src,dst) < dist[dst]:
                    dist[dst] = dist[src] + graph.w(src,dst)
                    prev[dst] = src
    
    # no negative cycle detection because that depended on relaxing more than V-1 times, 
    # but 2.2 dictates relaxing k times exactly. 
    return dist, prev


#2.3 Experiment
def experiment_A():

    #implementation

    return