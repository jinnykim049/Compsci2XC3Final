#Part3 
#---------------------------------3. Function Implementation----------------------------------------------------------------------------------------------------------------- 
#Objective: design an all-pair shortest path algorithm for both positive edge weights and negative edge weights.

#compute distance between i and j recursively with k as intermediate node
def path_recursive(i, j, k, graph, V, memo, pred): #V=number of total vertices, memo = dict that stores the shortest path
    
    #Basecase: If no intermediate nodes are available. 
    if k < 0:  
        if i == j:
            pred[(i, j, k)] = i
            return 0
        
        #check if a direct edge exists between i and j 
        for (v, weight) in graph.get(i, []):
            if v == j:
                pred[(i, j, k)] = i
                return weight
        return float('inf')
    
    #memoization to avoid redundant calculations
    if (i, j, k) in memo:
        return memo[(i, j, k)]

    without_k = path_recursive(i, j, k-1, graph, V, memo, pred) #shortest path from i to j without k
    left = path_recursive(i, k, k-1, graph, V, memo, pred) #shortest path from i to k 
    right = path_recursive(k, j, k-1, graph, V, memo, pred) #shortest path from k to j
    with_k = left + right if left != float('inf') and right != float('inf') else float('inf') #shortest path from i to k to j 
    

    #choose the shorteset path and stores the path in pred dict (to contruct path later)
    if with_k < without_k:
        memo[(i, j, k)] = with_k
        pred_kj = pred.get((k, j, k-1), k)
        pred[(i, j, k)] = pred_kj if pred_kj is not None else k #update predecessor 
    else:
        memo[(i, j, k)] = without_k
        pred_ij = pred.get((i, j, k-1), None)
        pred[(i, j, k)] = pred_ij if pred_ij is not None else None  #update predecessor  
    
    return memo[(i, j, k)] #Return the shortest path from i to j with k as intermediate node 


#Find all pairs shortest path 
def all_pairs_path(graph, V):
    distances = {} #Stores distances for each pair of vertices
    predecessors = {} #Stores predecessors for building the path 
    memo = {}  
    pred = {}  

    for i in range(V):
        for j in range(V):

            #call distances / path info for each pair of vertices
            d = path_recursive(i, j, V-1, graph, V, memo, pred)
            distances[(i, j)] = d
            predecessors[(i, j)] = pred.get((i, j, V-1), None)
    
    return distances, predecessors


#To build the shortest paths.
def reconstruct_path(i, j, predecessors):
    if i == j: #source = dest so path is itself 
        return [i]
    if predecessors.get((i, j)) is None:
        return []  # No path exists
    
    path = [j] #Start path from destination
    while j != i: #Traverse back using predecessor map 
        j = predecessors.get((i, j))
        if j is None:
            return []  # No valid path
        path.append(j)
    path.reverse()
    return path



#Test case 
graph = {
    0: [(1, 3), (2, 8), (4, -4)],
    1: [(3, 1), (4, 7)],
    2: [(1, 4)],
    3: [(0, 2), (2, -5)],
    4: [(3, 6)]
}
V = 5 #0-4 

distances, predecessors = all_pairs_path(graph, V)

print("All-pair Shortest Paths:")
for i in range(V):
    for j in range(V):
        path = reconstruct_path(i, j, predecessors)
        print(f"Path from {i} to {j}: {path if path else 'No path'}")


"""
---------------------------------3. Report----------------------------------------------------------------------------------------------------------------- 
Time complexity of the algorithm if it can handle both dense graphs and both negative / positive weights (which is part 3 algorithm)
*V: number of vertices, E: number of edges

<Time complexity>: Compute running time 
1. def path_recursive(i, j, k, graph, V, memo, pred): V * V * V = O(V^3)
There are three recursive calls in the function. path_recursive(i, j, k-1), path_recursive(i, k, k-1), and path_recursive(k, j, k-1).
Each call has a time complexity of O(V) because it iterates over all vertices in the graph (only one time each, because we have memoization).

2. def all_pairs_path(graph, V): V * V * V = O(V^3) 
The function calls path_recursive for all pairs of vertices, resulting in a time complexity of O(V^2 * V) = O(V^3) using memoization.

3. def reconstruct_path(i, j, predecessors): Worst case- O(V)
while loop has a time complexity of O(V) because it iterates over all vertices in the graph (only one time each).


Total time complexity: O(V^3 + V^3 + V) = O(V^3).


<Space complexity>: Compute all that store some data.  
-Note: Graph is dense, so E = O(V^2). 

1. memo[(i, j, k)]: stores maximum (V^2: pairs of vertices) * V: possible intermediate nodes = O(V^3)  
2. pred[(i, j, k)]: stores maximum (V^2: pairs of vertices) * V: possible intermediate nodes = O(V^3)
3. distances[(i, j)]: stores maximum V * V = O(V^2)
4. predecessors[(i, j)]: stores maximum V * V = O(V^2)

Since the graph is dense, the number of edges E is high, and E = O(V^2). This leads to a total space complexity of O(V^3).

Total space complexity: O(V^3 + V^3 + V^2 + V^2) = O(V^3).
 
""" 
 

     