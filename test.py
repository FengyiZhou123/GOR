import networkx as nx
import utils
import real_time as rt
import numpy as np
import heapq
from GOR_new import init_real_graph, PriorityQueue

import datetime

g_data = [(1, 2, 6), (1, 3, 1), (1, 4, 5),
          (2, 3, 5),  (2, 5, 3),
          (3, 4, 5), (3, 5, 6), (3, 6, 4), (4, 6, 2),
          (5, 6, 6)]

G = nx.Graph()
G.add_weighted_edges_from(g_data)
(A, A_len) = utils.k_shortest_paths(G, source=1, target=4, k=10, weight='weight')
for route in (A, A_len):
    print(route)




# routes=rt.KSP(G,1,6,4)
# print(routes)
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
# pq.put(3,-1)
# for i in pq:
#     print(i)

# list=[1,2,3]
# for num in list:
#     if num==1:
#         print(num)
#         list.remove(num)
#
# print(list)
# list.append(4)
# print(list)

# G= init_real_graph()
# print(nx.is_strongly_connected(G))
# print(len(G.nodes))

# class node:
#     def __init__(self, start):
#         self.start=start
#         self.routes=[]
#
# node1= node(1)
# node2= node(1)
# print(node1)

# time1= datetime.datetime.now()
# print(time1)
# for i in range(100000):
#     pass
# time2= datetime.datetime.now()
# print(time2-time1)
# i=0
# i+=(time2-time1).microseconds
# print(i)

dict={}

dict[(1,2)] = [[2, 3]]
dict[(1,2)].append([1,2])
dict.get((1,2))
for i in dict[(1,2)]:
    print(i[0])


