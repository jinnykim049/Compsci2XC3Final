import matplotlib
import numpy
import random

#utility functions
#draw_plot()

#non-negative weights connected graph generator
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

#graph class (undirected weighted)
class Graph():
    # you may use the above weighted graph class here and modify it if necessary.
    # aslo feel free to borrow any of the suitable graph class implementations discussed in class. 
    # weighted (undirected) graph implementation from lecture notes:
    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []

    def are_connected(self, node1, node2):
        for node in self.adj[node1]:
            if node == node2:
                return True
        return False

    def connected_nodes(self, node):
        return self.graph[node]

    def add_node(self,):
        #add a new node number = length of existing node
        self.graph[len(self.graph)] = []

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight

            #since it is undirected
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight

    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
                
        # because it is undirected
        return total/2


#2.1 Dijkstra's
def dijkstra(graph, source, k):

    #implementation

    return


#2.2 Bellman Ford's
def bellman_ford(graph, source, k):

    #implementation

    return

#2.3 Experiment
def experiment_A():

    #implementation

    return