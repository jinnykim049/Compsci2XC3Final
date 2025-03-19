import matplotlib.pyplot as plt
import numpy
import random
import math
import tracemalloc
import time

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

    sizes = [10, 50, 100, 200]
    densities = [0.1, 0.5, 0.9]
    k_values = [1, 3, 5]

    result = []

    for size in sizes:
        for density in densities:
            edges = int(density * (size) * (size - 1)) # for directed weighted graph
            graph = create_random_graph

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
                _, dijkstra_optimal = dijkstra(graph, 0, size - 1)
                _, bellman_optimal = bellman_ford(graph, 0, size - 1)

                # for the accuracy. The number of times that distance matches with optimal distance, and divided by the total size
                dijkstra_accuracy = sum(1 for node in dijkstra_distance if dijkstra_distance[node] == dijkstra_optimal[node]) / size
                bellman_accuracy = sum(1 for node in bellman_distance if bellman_distance[node] == bellman_optimal[node]) / size

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
    time_results = {}
    memory_results = {}
    accuracy_results = {}

    for result in result:
        key = (result["size"], result["density"], result["k"])
    
        if key not in time_results:
            time_results[key] = {"dijkstra": [], "bellman": []}
        time_results[key]["dijkstra"].append(result["dijkstra_time"])
        time_results[key]["bellman"].append(result["bellman_time"])

   
        if key not in memory_results:
            memory_results[key] = {"dijkstra": [], "bellman": []}
        memory_results[key]["dijkstra"].append(result["dijkstra_memory"])
        memory_results[key]["bellman"].append(result["bellman_memory"])

   
        if key not in accuracy_results:
            accuracy_results[key] = {"dijkstra": [], "bellman": []}
        accuracy_results[key]["dijkstra"].append(result["dijkstra_accuracy"])
        accuracy_results[key]["bellman"].append(result["bellman_accuracy"])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Time Plot
    axs[0].set_title('Execution Time (seconds)')
    axs[0].set_xlabel('Graph Size and Density (size, density, k)')
    axs[0].set_ylabel('Time (seconds)')
    for key in time_results:
        size, density, k = key
        axs[0].plot([k for _ in range(len(time_results[key]["dijkstra"]))], time_results[key]["dijkstra"], label=f'Dijkstra: size={size}, density={density}')
        axs[0].plot([k for _ in range(len(time_results[key]["bellman"]))], time_results[key]["bellman"], label=f'Bellman-Ford: size={size}, density={density}')
    axs[0].legend()

    # Memory Plot
    axs[1].set_title('Memory Usage (bytes)')
    axs[1].set_xlabel('Graph Size and Density (size, density, k)')
    axs[1].set_ylabel('Memory Usage (bytes)')
    for key in memory_results:
        size, density, k = key
        axs[1].plot([k for _ in range(len(memory_results[key]["dijkstra"]))], memory_results[key]["dijkstra"], label=f'Dijkstra: size={size}, density={density}')
        axs[1].plot([k for _ in range(len(memory_results[key]["bellman"]))], memory_results[key]["bellman"], label=f'Bellman-Ford: size={size}, density={density}')
    axs[1].legend()

    # Accuracy Plot
    axs[2].set_title('Algorithm Accuracy')
    axs[2].set_xlabel('Graph Size and Density (size, density, k)')
    axs[2].set_ylabel('Accuracy (%)')
    for key in accuracy_results:
        size, density, k = key
        axs[2].plot([k for _ in range(len(accuracy_results[key]["dijkstra"]))], accuracy_results[key]["dijkstra"], label=f'Dijkstra: size={size}, density={density}')
        axs[2].plot([k for _ in range(len(accuracy_results[key]["bellman"]))], accuracy_results[key]["bellman"], label=f'Bellman-Ford: size={size}, density={density}')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# Run the experiment
experiment_A()