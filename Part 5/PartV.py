import csv
import math
import time

# data structures and useful functions

class Graph():
    # weighted (undirected) graph implementation from lecture notes:
    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []
        self.line = {}

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

    def add_edge(self, node1, node2, weight, line):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight
            self.line[(node1, node2)] = line

            #since it is undirected
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight
            self.line[(node1, node2)] = line

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

#takes in the IDs of two stations and computes the euclidean distance between them
#uses the longitude and latitude of the stations from the stations dictionary
def station_dist(a, b):
    
    latA = stations[a][0]
    lonA = stations[a][1]
    
    latB = stations[b][0]
    lonB = stations[b][1]
    
    dist = math.sqrt((latB - latA)**2 + (lonB - lonA)**2)
    
    return dist


# data processing

stations = {} #Example data: {1: [51.5028,-0.2801,"Acton Town","Acton<br />Town",3,2,0]}
# process london_stations.csv into a dictionary

l = [i for i in range(1,304)]
theTube = Graph(l)
# process london_connections.csv into a graph


# searching algorithms

# reconstruct path function with line count

# src dst dijkstra on undirected g

# A* heuristic function

# src dst A* on undirected g



# experiments