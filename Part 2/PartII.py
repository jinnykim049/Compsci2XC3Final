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
# Time comparison
def experiment_excution_time():

    sizes = [10, 50, 100, 150, 250]
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    k_values = [1, 5, 10, 20, 50]

    dijkstra_size_times, bellman_size_times = [], []
    dijkstra_density_times, bellman_density_times = [], []
    dijkstra_k_times, bellman_k_times = [], []

    # Test graph size effect
    for nodes in sizes:
        edges = int(nodes * (nodes - 1) * 0.5)  # Medium density
        graph = create_random_graph(nodes, edges, 1, 10)

        start_time = time.time()
        dijkstra(graph, 0, max(k_values))
        dijkstra_size_times.append(time.time() - start_time)

        start_time = time.time()
        bellman_ford(graph, 0, max(k_values))
        bellman_size_times.append(time.time() - start_time)

    # Test graph density effect (fixing size to 100 nodes)
    fixed_nodes = 100
    for density in densities:
        edges = int(fixed_nodes * (fixed_nodes - 1) * density)
        graph = create_random_graph(fixed_nodes, edges, 1, 10)

        start_time = time.time()
        dijkstra(graph, 0, max(k_values))
        dijkstra_density_times.append(time.time() - start_time)

        start_time = time.time()
        bellman_ford(graph, 0, max(k_values))
        bellman_density_times.append(time.time() - start_time)

    # Test k-value effect (fixing size to 100 nodes and density to 0.5)
    graph = create_random_graph(100, int(100 * 99 * 0.5), 1, 10)
    for k in k_values:
        start_time = time.time()
        dijkstra(graph, 0, k)
        dijkstra_k_times.append(time.time() - start_time)

        start_time = time.time()
        bellman_ford(graph, 0, k)
        bellman_k_times.append(time.time() - start_time)

    # Plot execution time vs. graph size
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, dijkstra_size_times, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(sizes, bellman_size_times, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Effect of Graph Size on Execution Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot execution time vs. graph density
    plt.figure(figsize=(8, 5))
    plt.plot(densities, dijkstra_density_times, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(densities, bellman_density_times, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Graph Density (Fraction of Max Edges)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Effect of Graph Density on Execution Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot execution time vs. k-value
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, dijkstra_k_times, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(k_values, bellman_k_times, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Value of k (Relaxation Count)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Effect of k-value on Execution Time")
    plt.legend()
    plt.grid(True)
    plt.show()

experiment_excution_time()

# Accuracy compariosn
def calculate_accuracy(true_distances, estimated_distances):
    errors = [abs(true_distances[node] - estimated_distances.get(node, float('inf'))) for node in true_distances]
    return 1 - (sum(errors) / len(errors) / max(true_distances.values()))  # Normalize accuracy to [0,1]

def experiment_accuracy():
    node_sizes = [10, 50, 100, 150, 250]    
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]  
    k_values = [1, 5, 10, 20, 50, 100]  

    dijkstra_size_accuracy, bellman_size_accuracy = [], []
    dijkstra_density_accuracy, bellman_density_accuracy = [], []
    dijkstra_k_accuracy, bellman_k_accuracy = [], []

    # Test graph size effect
    for nodes in node_sizes:
        edges = int(nodes * (nodes - 1) * 0.5) 
        graph = create_random_graph(nodes, edges, 1, 10)
        true_distances, _ = bellman_ford(graph, 0, nodes-1)  # Use full Bellman-Ford as ground truth

        est_distances, _ = dijkstra(graph, 0, max(k_values))
        dijkstra_size_accuracy.append(calculate_accuracy(true_distances, est_distances))

        est_distances, _ = bellman_ford(graph, 0, max(k_values))
        bellman_size_accuracy.append(calculate_accuracy(true_distances, est_distances))

    # Test graph density effect (fixing size to 100 nodes)
    fixed_nodes = 100
    for density in densities:
        edges = int(fixed_nodes * (fixed_nodes - 1) * density)
        graph = create_random_graph(fixed_nodes, edges, 1, 10)
        true_distances, _ = bellman_ford(graph, 0, fixed_nodes-1)

        est_distances, _ = dijkstra(graph, 0, max(k_values))
        dijkstra_density_accuracy.append(calculate_accuracy(true_distances, est_distances))

        est_distances, _ = bellman_ford(graph, 0, max(k_values))
        bellman_density_accuracy.append(calculate_accuracy(true_distances, est_distances))

    # Test k-value effect (fixing size to 100 nodes and density to 0.5)
    graph = create_random_graph(100, int(100 * 99 * 0.5), 1, 10)
    true_distances, _ = bellman_ford(graph, 0, 100-1)
    for k in k_values:
        est_distances, _ = dijkstra(graph, 0, k)
        dijkstra_k_accuracy.append(calculate_accuracy(true_distances, est_distances))

        est_distances, _ = bellman_ford(graph, 0, k)
        bellman_k_accuracy.append(calculate_accuracy(true_distances, est_distances))

        
    # Plot accuracy vs. graph size (Bar Chart)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(node_sizes))
    width = 0.35
    plt.bar(x - width/2, dijkstra_size_accuracy, width, label="Dijkstra's")
    plt.bar(x + width/2, bellman_size_accuracy, width, label="Bellman-Ford's")
    plt.xticks(x, node_sizes)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.title("Effect of Graph Size on Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot accuracy vs. graph density (Bar Chart)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(densities))
    plt.bar(x - width/2, dijkstra_density_accuracy, width, label="Dijkstra's")
    plt.bar(x + width/2, bellman_density_accuracy, width, label="Bellman-Ford's")
    plt.xticks(x, densities)
    plt.xlabel("Graph Density (Fraction of Max Edges)")
    plt.ylabel("Accuracy")
    plt.title("Effect of Graph Density on Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()  

    # Plot accuracy vs. k-value (Bar Chart)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(k_values)) 
    width = 0.3  # Reduce bar width to avoid overlap
    plt.bar(x - width/2, dijkstra_k_accuracy, width, label="Dijkstra's", color='blue', alpha=0.7)
    plt.bar(x + width/2, bellman_k_accuracy, width, label="Bellman-Ford's", color='orange', alpha=0.7)
    plt.xticks(x, [str(k) for k in k_values])  # Ensure proper label alignment
    plt.xlabel("Value of k (Relaxation Count)")
    plt.ylabel("Accuracy")
    plt.title("Effect of k-value on Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

experiment_accuracy()

# space complexity compariosn
def experiment_space_complexity():
    node_sizes = [10, 50, 100, 150, 250]  
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]  
    k_values = [0, 1, 5, 10, 20, 50, 100]  

    dijkstra_size_memory, bellman_size_memory = [], []
    dijkstra_density_memory, bellman_density_memory = [], []
    dijkstra_k_memory, bellman_k_memory = [], []

    # Test graph size effect
    for nodes in node_sizes:
        edges = int(nodes * (nodes - 1) * 0.5)  # Medium density
        graph = create_random_graph(nodes, edges, 1, 10)

        tracemalloc.start()
        dijkstra(graph, 0, max(k_values))
        _, peak_memory = tracemalloc.get_traced_memory()
        dijkstra_size_memory.append(peak_memory / (1024 * 1024))  # Convert to MB
        tracemalloc.stop()

        tracemalloc.start()
        bellman_ford(graph, 0, max(k_values))
        _, peak_memory = tracemalloc.get_traced_memory()
        bellman_size_memory.append(peak_memory / (1024 * 1024))  # Convert to MB
        tracemalloc.stop()

    # Test graph density effect (fixing size to 100 nodes)
    fixed_nodes = 100
    for density in densities:
        edges = int(fixed_nodes * (fixed_nodes - 1) * density)
        graph = create_random_graph(fixed_nodes, edges, 1, 10)

        tracemalloc.start()
        dijkstra(graph, 0, max(k_values))
        _, peak_memory = tracemalloc.get_traced_memory()
        dijkstra_density_memory.append(peak_memory / (1024 * 1024))
        tracemalloc.stop()

        tracemalloc.start()
        bellman_ford(graph, 0, max(k_values))
        _, peak_memory = tracemalloc.get_traced_memory()
        bellman_density_memory.append(peak_memory / (1024 * 1024))
        tracemalloc.stop()

    # Test k-value effect (fixing size to 100 nodes and density to 0.5)
    graph = create_random_graph(100, int(100 * 99 * 0.5), 1, 10)
    for k in k_values:
        tracemalloc.start()
        dijkstra(graph, 0, k)
        _, peak_memory = tracemalloc.get_traced_memory()
        dijkstra_k_memory.append(peak_memory / (1024 * 1024))
        tracemalloc.stop()

        tracemalloc.start()
        bellman_ford(graph, 0, k)
        _, peak_memory = tracemalloc.get_traced_memory()
        bellman_k_memory.append(peak_memory / (1024 * 1024))
        tracemalloc.stop()

    # Plot memory usage vs. graph size
    plt.figure(figsize=(8, 5))
    plt.plot(node_sizes, dijkstra_size_memory, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(node_sizes, bellman_size_memory, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Effect of Graph Size on Memory Usage")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot memory usage vs. graph density
    plt.figure(figsize=(8, 5))
    plt.plot(densities, dijkstra_density_memory, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(densities, bellman_density_memory, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Graph Density (Fraction of Max Edges)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Effect of Graph Density on Memory Usage")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot memory usage vs. k-value
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, dijkstra_k_memory, marker='o', linestyle='dotted', label="Dijkstra's")
    plt.plot(k_values, bellman_k_memory, marker='s', linestyle='dotted', label="Bellman-Ford's")
    plt.xlabel("Value of k (Relaxation Count)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Effect of k-value on Memory Usage")
    plt.legend()
    plt.grid(True)
    plt.show()

experiment_space_complexity()
    







    