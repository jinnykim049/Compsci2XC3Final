import matplotlib.pyplot as plt
import numpy as np
import random
import math
import tracemalloc
import time

#---------------------------------Helper functions------------------------------------------------------------------------------------------------------------------
#utility functions
#draw_plot()

#directed weighted connected graph generator
#int number of nodes, int number of edges, int minimum edge weight, int maximum edge weight
def create_random_graph(nodes, edges, min_weight, max_weight): 
    # if edges exceed maximum, infinite loop
    max = nodes*(nodes-1) #formula for directed graph max edge number
    if edges > max:
        edges = max
    
    graph = Graph()
    for i in range(nodes):
        graph.add_node(i)
    
    #ensure it is connected
    for i in range(nodes-1):
        graph.add_edge(i,i+1,random.randint(min_weight, max_weight))
    
    for _ in range(edges-nodes+1): #node-1 number of edges have already been added
        src = random.randint(0, nodes-1)
        dst = random.randint(0, nodes-1)
        while graph.has_edge(src, dst) or src==dst: #verifying valid src dst, if not change it (no self-loops)
            src = random.randint(0, nodes-1)
            dst = random.randint(0, nodes-1)
            
        graph.add_edge(src, dst, random.randint(min_weight,max_weight))

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
    
    def has_edge(self,src,dst):
        return dst in self.adj[src]

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

#----------------------------------------------------------Implementations---------------------------------------------------------------------------------------------------------------

#2.1 Dijkstra's
def dijkstra(graph, source, k):

    #Initialization
    dist = {node: float('inf') for node in graph.adj}
    prev = {node: None for node in graph.adj}
    relax_count = {node: 0 for node in graph.adj}  # Record relaxation count 
    dist[source] = 0
    pq = MinHeap([Node(node, dist[node]) for node in graph.adj])

    while not pq.is_empty():
        u = pq.extract_min().value

        # limit relaxation up to k times 
        if relax_count[u] >= k:
            continue

        for v in graph.adj[u]:
            alt = dist[u] + graph.weights[(u, v)]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                pq.decrease_key(v, alt)
                relax_count[v] += 1  # Increase relaxation count

    return dist, prev 



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

    sizes = [10, 50]
    densities = [0.1, 0.5, 0.9]
    k_values = [1, 3, 5]

    result = []

    for size in sizes:
        for density in densities:
            edges = int(density * (size) * (size - 1)) # for directed weighted graph
            graph = create_random_graph(size, edges, 1, 10 )

            for k in k_values:

                tracemalloc.start() # track for memory usage
                start_time = time.time()
                dijkstra_distance, _ = dijkstra(graph, 0, k)
                dijkstra_time = time.time() - start_time
                dijkstra_memory = tracemalloc.get_traced_memory()[1] # give the memory usage
                tracemalloc.stop()

                tracemalloc.start() # track for memory usage
                start_time = time.time()
                bellman_distance, _ = bellman_ford(graph, 0, k)
                bellman_time = time.time() - start_time
                bellman_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()

                # run both of them with max possible relextion steps (k = N-1)
                dijkstra_optimal_diatance, _ = dijkstra(graph, 0, size - 1)
                bellman_optimal_disance, _ = bellman_ford(graph, 0, size - 1)

                # for the accuracy. The number of times that distance matches with optimal distance, and divided by the total size
                dijkstra_accuracy = sum(1 for node in dijkstra_distance if dijkstra_distance[node] == dijkstra_optimal_diatance[node]) / size
                bellman_accuracy = sum(1 for node in bellman_distance if bellman_distance[node] == bellman_optimal_disance[node]) / size

                result.append({
                    "size": size,
                    "density": density,
                    "k": k,
                    "dijkstra_time": dijkstra_time,
                    "bellman_time": bellman_time,
                    "dijkstra_memory": dijkstra_memory,
                    "bellman_memory": bellman_memory,
                    "dijkstra_accuracy": dijkstra_accuracy,
                    "bellman_accuracy": bellman_accuracy
                })

    plot_result(result)

def plot_result(result):
    # Extract data into separate lists
    labels = []

    dijkstra_time = []
    bellman_time = []

    dijkstra_memory = []
    bellman_memory = []

    dijkstra_accuracy = []
    bellman_accuracy = []

    for entry in result:
        labels.append((entry["size"], entry["density"], entry["k"]))
        
        dijkstra_time.append(entry["dijkstra_time"])
        bellman_time.append(entry["bellman_time"])
        
        dijkstra_memory.append(entry["dijkstra_memory"])
        bellman_memory.append(entry["bellman_memory"])
        
        dijkstra_accuracy.append(entry["dijkstra_accuracy"])
        bellman_accuracy.append(entry["bellman_accuracy"])

    x = np.arange(len(labels)) 
    width = 0.35  

    def create_bar_chart(y1, y2, title, ylabel, xlabel):
        plt.figure()  # Create a new figure
        plt.bar(x - width/2, y1, width, label='Dijkstra', color='b')
        plt.bar(x + width/2, y2, width, label='Bellman-Ford', color='r')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        

        actual_labels = [f"Size: {label[0]}, Density: {label[1]}, k: {label[2]}" for label in labels]
        
        # Apply xticks with rotation to set the labels under the bars
        plt.xticks(x, actual_labels, rotation=45, ha="right", fontsize=10)

        # Add custom text annotations for size, density, and k values below the bars
        for i, (size, density, k) in enumerate(labels):
            # Add size, density, k text below the bars
            y_offset = -max(y1 + y2) * 0.1  # Adjust the vertical offset
            plt.text(x[i], y_offset, f"{size}", color='red', ha="center", fontsize=10)
            plt.text(x[i], y_offset - max(y1 + y2) * 0.05, f"{density}", color='blue', ha="center", fontsize=10)
            plt.text(x[i], y_offset - max(y1 + y2) * 0.1, f"{k}", color='green', ha="center", fontsize=10)


        plt.legend()
        plt.show()  

    create_bar_chart(dijkstra_time, bellman_time, 'Execution Time (seconds)', 'Time (seconds)', 'Size, Density, k_value')
    create_bar_chart(dijkstra_memory, bellman_memory, 'Memory Usage (bytes)', 'Memory (bytes)', 'Size, Density, k_value')
    create_bar_chart(dijkstra_accuracy, bellman_accuracy, 'Algorithm Accuracy', 'Accuracy (%)', 'Size, Density, k_value')


experiment_A()