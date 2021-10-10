import datetime

import networkx as nx
import numpy as np
from random import choice
import heapq
import sys
import pandas as pd
import json

data = pd.read_csv('./data.csv')
data = data.iloc[:, 1:]
data = data.to_numpy()
print(data.shape)
print(data[0])
tmp=[]


def init_real_graph(data):
    G = nx.Graph()
    for weight in data:
        G.add_weighted_edges_from([(weight[0], weight[1], weight[2])])
        # if isinstance(weight[0], int) or isinstance(weight[1], int) or isinstance(weight[2], int):
        #     print('no')
        # else: print('yes')
    G.add_weighted_edges_from([(0, 7388, 1.4108)])
    # print(len(G.edges))
    return G


class label:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.timeList = [0]
        self.routes = [start]


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, self.index, item))  # reorder x using priority
        self.index += 1

    def get(self):
        return heapq.heappop(self.queue)[-1]  # pop out element with smallest priority



def init_labels(queries):
    labels = []
    for query in queries:
        labels.append(label(query[0], query[1]))
    return labels


def init_small_world_graph(n, k, p):
    """

    :param n: n vertices
    :param k: k neighbors
    :param p: probability of rewiring
    :return:
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.DiGraph()
    nodes = list(range(n))
    weights = np.random.randint(1, 50, size=len(nodes))

    # 和附近的点连起来
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_weighted_edges_from(zip(nodes, targets, weights))
        G.add_weighted_edges_from(zip(targets, nodes, weights))

    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if np.random.rand() < p:
                w = np.random.randint(0, 100)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = np.random.randint(0, 100)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    weight = np.random.randint(1, 50)
                    G.remove_edge(u, v)
                    G.remove_edge(v, u)
                    G.add_weighted_edges_from([(u, w, weight)])
                    G.add_weighted_edges_from([(w, u, weight)])

    return G


def init_queries(query_num, graph_size):
    """
    init queries
    :param query_num: number of queries
    :return:
    """
    start = np.random.randint(0, graph_size, size=query_num)
    end = np.random.randint(0, graph_size, size=query_num)
    queries = list(zip(start, end))
    for i in queries:
        if i[0] == i[1]:
            queries.remove(i)

    return queries


# def init_scale_free_graph(init_num, extra_num, m1):
#     """
#     generate scale free graph
#     :param init_num: initial numbers that are connected
#     :param extra_num: extra numbers that are added after
#     :return:
#     """
#     G = nx.DiGraph()
#     Nodes = list(range(init_num))
#     targets = Nodes[2:] + Nodes[:2]
#     weights = np.random.randint(1, 50, size=len(Nodes))
#     G.add_weighted_edges_from(zip(Nodes, targets, weights))
#     G.add_weighted_edges_from(zip(targets, Nodes, weights))
#     G.remove_edge(targets[0], Nodes[0])
#     G.remove_edge(Nodes[0], targets[0])
#
#     new_nodes = list(range(extra_num))
#     for u in new_nodes:
#         G.add_node(u)
#         for j in range(m1):
#             i = np.random.randint(0, len(G.nodes))
#             if G.degree(i) < 100:
#                 weight = np.random.randint(1, 50)
#                 G.add_weighted_edges_from([(i, u, weight)])
#                 G.add_weighted_edges_from([(u, i, weight)])
#
#     return G
#
#
# def init_random_graph(total_num):
#     """
#     randomly add edges
#     :param total_num:
#     :return:
#     """
#     G = nx.DiGraph()
#     Nodes = list(range(total_num))
#     for i in Nodes:
#         G.add_node(i)
#     while list(nx.isolates(G)):
#         p = np.random.randn()
#         start = np.random.randint(0, total_num)
#         end = np.random.randint(0, total_num)
#         if p < 0.5:
#             G.add_weighted_edges_from([(start, end, np.random.randint(1, 50))])
#
#     return G
#
#
# def init_scale_free_withtf_graph(init_num, extra_num, m1):
#     G = nx.DiGraph()
#     Nodes = list(range(init_num))
#     targets = Nodes[2:] + Nodes[:2]
#     weights = np.random.randint(1, 50, size=len(Nodes))
#     G.add_weighted_edges_from(zip(Nodes, targets, weights))
#     G.add_weighted_edges_from(zip(targets, Nodes, weights))
#     G.remove_edge(targets[0], Nodes[0])
#     G.remove_edge(Nodes[0], targets[0])
#
#     new_nodes = list(range(extra_num))
#     for u in new_nodes:
#         G.add_node(u)
#         for j in range(m1):
#             i = np.random.randint(0, len(G.nodes))
#             if G.degree(i) < 100:
#                 if G.has_edge(i, u):
#                     neighbors = list(G.successors(i) and G.predecessors(i))
#                     next = choice(neighbors)
#                     weight = np.random.randint(1, 50)
#                     G.add_weighted_edges_from([(next, u, weight)])
#                     G.add_weighted_edges_from([(u, next, weight)])
#
#                 else:
#                     weight = np.random.randint(1, 50)
#                     G.add_weighted_edges_from([(i, u, weight)])
#                     G.add_weighted_edges_from([(u, i, weight)])
#
#     return G


def greedy_algorithm(labels, G, G_carNum, alpa, gen):
    """
    step 1
    :param labels:
    :return:
    """
    res = []
    qp = PriorityQueue()
    for label in labels:
        qp.put(label, label.timeList[-1])

    while not qp.empty():
        label_q = qp.get()
        # print(label_q.timeList[-1])
        # print("test1")
        # print(label_q.routes[-1])
        # print("test2")
        nodes=list(G.neighbors(label_q.routes[-1]))
        # print(nodes)
        costs=[]
        neighbor_nodes=[]
        for node in nodes:
                # print(nodes)
                # print("in routes"+str(node))
            if node not in label_q.routes and gen[node][label_q.end] < gen[label_q.routes[-1]][label_q.end]:
                # print(node)
                # print("test5")
                # print("not in routes"+str(node))

                weight = G.get_edge_data(label_q.routes[-1], node)
                if weight:
                    weight = weight["weight"]
                else:
                    continue
                    # print(weight)
                    # print("test3")
                car_num = get_car_num(qp, label_q, node)
                EGT_tmp = label_q.timeList[-1] + gen[node][label_q.end] + weight * (1 + alpa * car_num)
                    # print(EGT)
                    # print("EGT test")
                neighbor_nodes.append(node)
                costs.append(EGT_tmp)

        if costs:
            min_cost=sys.maxsize
            min_index=-1
            for i in range(len(costs)):
                if costs[i]<min_cost:
                    min_cost=costs[i]
                    min_index=i

            node=neighbor_nodes[min_index]
            label_q.timeList.append(min_cost-gen[node][label_q.end])
            label_q.routes.append(node)
            start=int(label_q.routes[-2])
            end=int(node)
            G_carNum[start][end] = int(car_num) + 1


            if label_q.routes[-1] == label_q.end:
                print(len(label_q.routes))
                print(label_q.routes)
                tmp.append(len(label_q.routes))
                res.append(label_q)
            else:
                qp.put(label_q, label_q.timeList[-1])

    return res, G_carNum


def refining_algorithm(combination, rate, G, G_carNum, alpa):
    res = []
    pq = PriorityQueue()
    for label in combination:
        pq.put(label, label.timeList[-1])

    while not pq.empty():
        label_tmp = pq.get()
        for i in range(len(label_tmp.routes)):
            # print(label_tmp.routes[-1])
            add_change, add_routes = get_add(i, label_tmp, G, G_carNum, alpa)
            delete_change, delete_routes = get_delete(i, label_tmp, G, G_carNum, alpa)
            change, change_routes = get_change(i, label_tmp, G, G_carNum, alpa)

            max_change, flag = get_max(add_change, delete_change, change, label_tmp, i)
            if max_change and flag == "add_change":
                GT = label_tmp.timeList[i + 1] - label_tmp.timeList[i]
                if GT / (GT - max_change) > (1 + rate):
                    label_tmp.timeList.insert(i + 1, add_change[1])
                    label_tmp.timeList[i + 2] = add_change[2]
                    label_tmp.routes.insert(i + 1, add_routes[1])

            if max_change and flag == "delete_change":
                GT = label_tmp.timeList[i + 2] - label_tmp.timeList[i]
                if GT / (GT - max_change) > (1 + rate):
                    label_tmp.timeList[i + 2] = delete_change[1]
                    del (label_tmp.timeList[i + 1])
                    del (label_tmp.routes[i + 1])

            if max_change and flag == "change":
                GT = label_tmp.timeList[i + 2] - label_tmp.timeList[i]
                if GT / (GT - max_change) > (1 + rate):
                    label_tmp.timeList[i + 1] = change[1]
                    label_tmp.timeList[i + 2] = change[2]
                    label_tmp.routes[i + 1] = change_routes[1]
        res.append(label_tmp)

    return res


def get_max(add_change, delete_change, change, label_tmp, i):
    max_num = 0
    tmp = ""

    if not add_change and not delete_change and not change:
        return max_num, tmp

    if add_change:
        if label_tmp.timeList[i + 1] - label_tmp.timeList[i] - (add_change[2] - add_change[0]) > max_num:
            max_num = label_tmp.timeList[i + 1] - label_tmp.timeList[i] - (add_change[2] - add_change[0])
            tmp = "add_max"

    if delete_change:
        if label_tmp.timeList[i + 2] - label_tmp.timeList[i] - (delete_change[1] - delete_change[0]) > max_num:
            max_num = label_tmp.timeList[i + 2] - label_tmp.timeList[i] - (delete_change[1] - delete_change[0])
            tmp = "delete_change"

    if change:
        if label_tmp.timeList[i + 2] - label_tmp.timeList[i] - (change[2] - change[0]) > max_num:
            max_num = label_tmp.timeList[i + 2] - label_tmp.timeList[i] - (change[2] - change[0])
            tmp = "change"

    return max_num, tmp


def get_add(i, label_tmp, G, G_carNum, alpa):
    nodes = []
    time = []

    if i + 1 >= len(label_tmp.timeList):
        return time, nodes

    start, end = int(label_tmp.routes[i]), int(label_tmp.routes[i + 1])
    for node in G.nodes:
        node=int(node)
        if G.has_edge(start, node) and G.has_edge(node, end):
            weight1 = G.get_edge_data(start, node)["weight"]
            weight2 = G.get_edge_data(node, end)["weight"]
            if weight1 * (1 + alpa * G_carNum[start][node]) + weight2 * (1 + alpa * G_carNum[node][end]) < \
                    label_tmp.timeList[i + 1] - label_tmp.timeList[i]:
                nodes = [start, node, end]
                time = [label_tmp.timeList[i], label_tmp.timeList[i] + weight1 * (1 + alpa * G_carNum[start][node]),
                        label_tmp.timeList[i] + weight1 * (1 + alpa * G_carNum[start][node]) + weight2 * (
                                1 + alpa * G_carNum[node][end])]

    return time, nodes


def get_delete(i, label_tmp, G, G_carNum, alpa):
    nodes = []
    time = []

    if i + 2 >= len(label_tmp.timeList):
        return time, nodes

    start, mid, end = int(label_tmp.routes[i]), int(label_tmp.routes[i + 1]), int(label_tmp.routes[i + 2])
    weight = G.get_edge_data(start, end)
    if weight:
        weight = weight["weight"]

    if weight and weight * (1 + alpa * G_carNum[start][end]) < label_tmp.timeList[i + 2] - label_tmp.timeList[i]:
        nodes = [start, end]
        time = [label_tmp.timeList[i], label_tmp.timeList[i] + weight * (1 + alpa * G_carNum[start][end])]

    return time, nodes


def get_change(i, label_tmp, G, G_carNum, alpa):
    nodes = []
    time = []

    if i + 2 >= len(label_tmp.timeList):
        return time, nodes

    start, mid, end = int(label_tmp.routes[i]), int(label_tmp.routes[i + 1]), int(label_tmp.routes[i + 2])
    # print(end)
    # print("endtest")
    for node in G.nodes:
        node=int(node)
        if not node == start and not node == end and G.has_edge(start, node) and G.has_edge(node, end):
            weight1 = G.get_edge_data(start, node)["weight"]
            weight2 = G.get_edge_data(node, end)["weight"]
            if weight1 * (1 + alpa * G_carNum[start][node]) + weight2 * (1 + alpa * G_carNum[node][end]) < \
                    label_tmp.timeList[i + 2] - label_tmp.timeList[i]:
                nodes = [start, node, end]
                time = [label_tmp.timeList[i], label_tmp.timeList[i] + weight1 * (1 + alpa * G_carNum[start][node]),
                        label_tmp.timeList[i] + weight1 * (1 + alpa * G_carNum[start][node]) + weight2 * (
                                1 + alpa * G_carNum[node][end])]
    return time, nodes


def get_car_num(qp, label_q, node):
    res = 0
    start, end = label_q.routes[-1], node
    time=label_q.timeList[-1]
    for label in qp.queue:
        if len(label[-1].routes) >= 2 and label[-1].routes[-1] == end and label[-1].routes[-2] == start \
                and time<=label[-1].timeList[-1] and time>=label[-1].timeList[-2]:
            res += 1
    return res


def show_answer(combination_1):
    res = 0
    for label in combination_1:
        res += label.timeList[-1]
    return res


def planning():
    """
    main part of the GOR
    :return:
    """

    graph_size = 18262
    rate = 0.02
    alpa = 0.02
    G = init_real_graph(data)
    # G=init_small_world_graph(10000,10,0.2)
    G_carNum = np.zeros((graph_size+1, graph_size+1), dtype=int)
    # print(dict(G_weight))
    q = init_queries(query_num=2000, graph_size=graph_size)
    labels = init_labels(q)

    # gen = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    start1 = datetime.datetime.now()
    print(start1)
    gen = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    end1 = datetime.datetime.now()
    print((end1 - start1).seconds)


    # f = open('data_dijkstra.txt', 'r')
    # a = f.read()
    # gen = eval(a)
    # f.close()

    # f = open('data_dijkstra.txt', 'w')
    # f.write(str(gen))
    # f.close()

    start = datetime.datetime.now()
    print(start)
    combination_1, G_carNum = greedy_algorithm(labels, G, G_carNum, alpa, gen)
    end = datetime.datetime.now()
    print(end)
    print((end - start).seconds)
    total_time1 = show_answer(combination_1)
    print(total_time1)


    start3 = datetime.datetime.now()
    print(start3)
    combination_2 = refining_algorithm(combination_1, rate, G, G_carNum, alpa)
    end3 = datetime.datetime.now()
    print(end3)
    print((end3 - start3).seconds)
    total_time2 = show_answer(combination_2)
    print(total_time2)


planning()

# g_data = [(0, 1, 1),(2,0,5),(1,2,2),(4,2,6),(3,1,3)]
# G = nx.Graph()
# G.add_weighted_edges_from(g_data)
# print(G.edges)
# G_carNum = np.zeros((5, 5), dtype=int)
# q = init_queries(query_num=5, graph_size=5)
# print(q)
# labels = init_labels(q)
# combination_1, G_carNum = greedy_algorithm(labels, G, G_carNum, 0.2)
# total_time1 = show_answer(combination_1)
# print(total_time1)

# G=init_real_graph(data)
# for i in range(100):
#     start_node = np.random.randint(0, 18262)
#     end_node = np.random.randint(0, 18262)
#     print(list(G.neighbors(start_node)))
#     print(nx.dijkstra_path(G,start_node,end_node,weight="weight"))
#     print(len(nx.dijkstra_path(G,start_node,end_node,weight="weight")))
