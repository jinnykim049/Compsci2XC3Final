import math
import time

# Replace this with your actual reconstruct_path function if needed
def reconstruct_path(graph, prev, dst):
    path = [dst]
    prev_node = prev[dst]
    while prev_node is not None:
        path.append(prev_node)
        prev_node = prev[prev_node]
    path.reverse()

    lines = set()
    for i in range(len(path) - 1):
        if (path[i], path[i+1]) in graph.line:
            lines.add(graph.line[(path[i], path[i+1])])

    return path, len(lines)
def dijkstra(graph, source, destination):

    #Initialization
    dist = {}
    prev = {}
    for node in graph.graph:
        dist[node] = math.inf
        prev[node] = None
    dist[source] = 0
    pq = MinHeap([Node(node, dist[node]) for node in graph.graph])

    while not pq.is_empty():
        u = pq.extract_min().value
        
        if u == destination:
            return reconstruct_path(graph, prev, destination), dist[destination]

        for v in graph.graph[u]:
            alt = dist[u] + graph.weight[(u, v)]
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
    for node in graph.graph:
        dist[node] = math.inf
        prev[node] = None
    dist[source] = 0
    pq = [(station_dist(source,destination), source)]#f's are computed as needed
    # streamlined the comparison with how Python auto compares first value in a tuple first
    
    seen = set()

    while pq:
        _, u = heapq.heappop(pq)
        if u in seen: #dijkstra processes each node only once
            continue #only added because of the awkward way i used heapq ;-;
        
        if u == destination:
            return reconstruct_path(graph, prev, destination), dist[destination]
        
        seen.add(u)

        for v in graph.graph[u]:
            alt = dist[u] + graph.weight[(u, v)]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                if v in seen: #limits duplicates in the pq
                    continue
                #might still have dupes but need to make sure smallest f is in pq
                #so it's like a primitive decrease_key lol
                heapq.heappush(pq, (station_dist(v,destination), v))
                
    return math.inf #PATH NOT FOUND


def pair_type(src, dst, transfers):
    if transfers == 1:
        return "same_line"
    elif transfers == 2:
        return "adjacent_line"
    else:
        return "multi_transfer"

known_same_line_pairs = [
    (1, 52),     
    (73, 182),   
    (11, 83),    
    (4, 27),     
    (74, 287),   
]

for src, dst in known_same_line_pairs:
    print(f"\nTesting: {src} → {dst}")

    # Dijkstra
    t1 = time.perf_counter()
    dijkstra_result, d_len = dijkstra(graph, src, dst)
    dijkstra_time = time.perf_counter() - t1

    # A*
    t2 = time.perf_counter()
    astar_result, a_len = A_Star(graph, src, dst)
    astar_time = time.perf_counter() - t2

    if dijkstra_result == math.inf or astar_result == math.inf:
        print("No path found.")
        continue

    path, transfers = dijkstra_result
    print("Path:", path)
    print("Transfers:", transfers)
    print("Dijkstra Time:", round(dijkstra_time, 6), "seconds")
    print("A* Time:", round(astar_time, 6), "seconds")
    print("Pair Type:", pair_type(src, dst, transfers))
