# # for q2, to declare the pair type
# def pair_type(src, dst):
#     shared = station_lines[src] & station_lines[dst] #share the same src and dst
#     if shared:
#         return "same_line"
#     elif any(len(station_lines[src] & station_lines[n]) > 0 for n in G.neighbors(dst)): # does not directlly connect with src and dst but connect through another point
#         return "adjacent_line"
#     else:
#         return "multi_transfer"

# results_df["pair_type"] = results_df.apply(lambda row: pair_type(row["src"], row["dst"]), axis=1)

# same_line = results_df[results_df["pair_type"] == "same_line"]
# adjacent_line = results_df[results_df["pair_type"] == "adjacent_line"]
# multi_transfer = results_df[results_df["pair_type"] == "multi_transfer"]

# print("Same Line Stats:")
# print("  A* Time:", same_line["astar_time"].mean())
# print("  Dijkstra Time:", same_line["dijkstra_time"].mean())
# print("  Avg Transfers:", same_line["transfers"].mean())

# print("\nAdjacent Line Stats:")
# print("  A* Time:", adjacent_line["astar_time"].mean())
# print("  Dijkstra Time:", adjacent_line["dijkstra_time"].mean())
# print("  Avg Transfers:", adjacent_line["transfers"].mean())

# print("\nMulti-Transfer Stats:")
# print("  A* Time:", multi_transfer["astar_time"].mean())
# print("  Dijkstra Time:", multi_transfer["dijkstra_time"].mean())
# print("  Avg Transfers:", multi_transfer["transfers"].mean())

# #part 3
# results_df["transfers"] = results_df["path"].apply(count_line_transfers)
