import random
import math


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
    graph.add_edge(nodes-1,0,random.randint(min_weight,max_weight)) #all nodes reachable
    
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




#----------------------------------------------------Part 3-------------------------------------------



#-----------------------------Bellman-Ford DP 
class ShortestPathMemo:
    def __init__(self):
        self.distance = {}
        self.previous = {}
        self.in_negative_cycle = set()

def incoming_nodes(graph, target):
    result = []
    for u in graph.adj:
        if target in graph.adj[u]: #if u-> target edges exist 
            result.append(u)
    return result


#Dynamic programming for Bellman-Ford
def sp_dp(graph, v, memo, visited):
    #memoization (memo) is used to store already computed shortest distances to reuse later 
    if v in memo.distance:
        return memo.distance[v]

    if v in visited:
        memo.in_negative_cycle.add(v) #negative cycle detected
        return float('-inf')

    #Initialization
    visited.add(v)
    memo.distance[v] = float('inf')
    memo.previous[v] = None


    #calculate the shortest path for all nodes u that are coming in rto node v
    for u in incoming_nodes(graph, v):  
        this_distance = sp_dp(graph, u, memo, visited) + graph.w(u, v)
        if this_distance < memo.distance[v]: #shorther path is found (update)
            memo.distance[v] = this_distance
            memo.previous[v] = u #to track path 

    visited.remove(v)
    return memo.distance[v]
 


# ----------------------------- APSP Functions 

#Check if the graph has negative weights
def has_negative_edge(graph):
    for weight in graph.weights.values():
        if weight < 0:
            return True
    return False



#Use Dijkstra with heap (find all-pairs path)
def all_pairs_dijkstra(graph):
    all_dist = {}
    all_prev = {} #to track path 
    for source in graph.adj:
        dist, prev = dijkstra(graph, source)
        all_dist[source] = dist
        all_prev[source] = prev
    return all_dist, all_prev


#Use Bellman-Ford with DP (find all-pairs path) 
def all_pairs_bellman_dp(graph):
    all_dist = {}
    all_prev = {}
    negative_cycles = set()

    for source in graph.adj:
        memo = ShortestPathMemo()
        memo.distance[source] = 0
        memo.previous[source] = None

        for v in graph.adj: #to calculate the shortest path for all nodes
            visited = set()
            sp_dp(graph, v, memo, visited)

        all_dist[source] = memo.distance
        all_prev[source] = memo.previous
        negative_cycles |= memo.in_negative_cycle

    return all_dist, all_prev, negative_cycles



#main: Select a shortest path algorithm based on the graph
def all_pairs_shortest_path(graph):
    if has_negative_edge(graph): #if there are negative edges in the graph
        print("Has negative edge: <Use Bellman-Ford with DP>")
        return all_pairs_bellman_dp(graph)
    else:
        print("Has only positive edges: <Use Dijkstra>")
        dist, prev = all_pairs_dijkstra(graph)
        return dist, prev, set()





#-----------------------------Test case 
g = Graph()
g.add_node(0)
g.add_node(1)
g.add_node(2)
g.add_node(3)

g.add_edge(0, 1, 4)
g.add_edge(0, 2, 1)
g.add_edge(2, 1, -2)   # negative edge
g.add_edge(1, 3, 2)
g.add_edge(2, 3, 5)

dist, prev, neg_cycles = all_pairs_shortest_path(g)

def reconstruct_path(source, target, prev_dict):
    path = []
    current = target
    while current is not None:
        path.insert(0, current)
        current = prev_dict.get(current)
    if path[0] == source:
        return path
    else:
        return []  # No valid path



print("\n <All shortest paths>")
for u in prev:
    for v in prev[u]:
        if u != v:
            path = reconstruct_path(u, v, prev[u])
            if path:
                print(f"Path from {u} to {v}: {path}  (Distance: {dist[u][v]})")
            else:
                print(f"Path from {u} to {v}: No path")
  


