import csv
import math
import time
import heapq

# data structures and useful functions ----------------------------------------------

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
            self.line[(node2, node1)] = line

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

class Node:
    def __init__(self, value, key=float('inf')):
        self.value = value   
        self.key = key     

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


#takes in the IDs of two stations and computes the euclidean distance between them
#uses the longitude and latitude of the stations from the stations dictionary
def station_dist(a, b):
    
    latA = stations[a][0]
    lonA = stations[a][1]
    
    latB = stations[b][0]
    lonB = stations[b][1]
    
    dist = math.sqrt((latB - latA)**2 + (lonB - lonA)**2)
    
    return dist


# data processing ----------------------------------------------

stations = {} #Example data: {1: [51.5028,-0.2801,"Acton Town","Acton<br />Town",3,2,0]}
# process london_stations.csv into a dictionary

l = [i for i in range(1,304)]
theTube = Graph(l)
# process london_connections.csv into a graph


# searching algorithms ----------------------------------------------
# (RETURNS ([SHORTEST PATH AS ARRAY OF NODES], NUMBER OF LINES (if 1, no transfers), LENGTH OF SHORTEST PATH))
# RETURNS INFINITY IF PATH NOT FOUND (PATH DOESN'T EXIST)

# reconstruct path function with line count
def reconstruct_path(graph, dict, dst):
    path = [dst]
    prev_node = dict[dst]
    while prev_node is not None:
        path.append[prev_node]
        prev_node = dict[prev_node]
    path.reverse()
    
    lines = set()
    for i in range(len(path) - 1):
        lines.add(graph.line[(path[i],path[i+1])])
    
    line_num = len(lines)
    # if 1, no transfers
    # if 2, one transfer
    # if more lines were taken along the path, multiple transfers
    return path, line_num

# src dst dijkstra on undirected g

#2.1 Dijkstra's
def dijkstra(graph, source, destination):

    #Initialization
    dist = {}
    prev = {}
    for node in graph.adj:
        dist[node] = math.inf
        prev[node] = None
    dist[source] = 0
    pq = MinHeap([Node(node, dist[node]) for node in graph.adj])

    while not pq.is_empty():
        u = pq.extract_min().value
        
        if u == destination:
            return reconstruct_path(graph, prev, destination), dist[destination]

        for v in graph.adj[u]:
            alt = dist[u] + graph.weights[(u, v)]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                pq.decrease_key(v, alt)

    return math.inf #PATH NOT FOUND

# src dst A* on undirected g
# built in with specific heuristic of distance to destination, no seperate heuristic function
def A_Star(graph, source, destination): 

    #Initialization
    dist = {}
    prev = {}
    for node in graph.adj:
        dist[node] = math.inf
        prev[node] = None
    dist[source] = 0
    pq = [(station_dist(source,destination), source)]#f's are computed as needed
    # streamlined the comparison with how Python auto compares first value in a tuple first
    
    seen = set()

    while pq:
        u = heapq.heappop(pq).value
        if u in seen: #dijkstra processes each node only once
            continue #only added because of the awkward way i used heapq ;-;
        
        if u == destination:
            return reconstruct_path(graph, prev, destination), dist[destination]
        
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
                heapq.heappush(pq, (station_dist(v,destination), v))
                
    return math.inf #PATH NOT FOUND


# experiments ----------------------------------------------