import networkx as nx
import real_time as rt
import numpy as np
import heapq

import datetime

g_data = [(1, 2, 6), (1, 3, 1), (1, 4, 5),
          (2, 3, 5),  (2, 5, 3),
          (3, 4, 5), (3, 5, 6), (3, 6, 4), (4, 6, 2),
          (5, 6, 6)]

G = nx.Graph()
G.add_weighted_edges_from(g_data)
routes=rt.KSP(G,1,6,4)
print(routes)
# gen = dict(nx.all_pairs_dijkstra_path_length(G,weight="weight"))
# print(gen[1])
# print(gen[1][1])
# print(65086/3600)


# g_data = [(0, 1, 1),(2,0,5),(1,2,2),(4,2,6),(3,1,3)]
# G = nx.Graph()
# G.add_weighted_edges_from(g_data)
# nodes=G.neighbors(0)
# for node in nodes:
#     print(node)
# G_carNum = np.zeros((5, 5), dtype=int)
# q = pl.init_queries(query_num=5, graph_size=5)
# print(q)
# labels = pl.init_labels(q)
# combination_1, G_carNum = pl.greedy_algorithm(labels, G, G_carNum, 0.2)
# total_time1 = pl.show_answer(combination_1)
# print(total_time1)

# class PriorityQueue:
#     def __init__(self):
#         self.queue = []
#         self.index = 0
#
#     def empty(self):
#         return len(self.queue) == 0  # if Q is empty
#
#     def put(self, item, priority):
#         heapq.heappush(self.queue, (priority, self.index, item))  # reorder x using priority
#         self.index += 1
#
#     def get(self):
#         return heapq.heappop(self.queue)[-1]  # pop out element with smallest priority
#
# pq=PriorityQueue()
# pq.put(1,1)
# pq.put(2,2)
# print(pq.get())
# print(pq.get())

# list=[1,2,3]
# for num in list:
#     if num==1:
#         print(num)
#         list.remove(num)
#
# print(list)
# list.append(4)
# print(list)
list=[]
if list:
    print(1)


