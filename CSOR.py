import heapq

import networkx as nx
import numpy as np

import planning as pl
import pandas as pd

data = pd.read_csv('./data.csv')
data = data.iloc[:, 1:]
data = data.to_numpy()


# class Node:
#     def __init__(self, node):
#         self.val = node
#         self.prev = None


class Query:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.routes = []
        self.timeList = []

class Label:
    def __init__(self, start_node, start_time, end_node, end_time):
        self.start_node = start_node
        self.start_time = start_time
        self.end_node = end_node
        self.end_time = end_time

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


def init_queries(query_num, graph_size):
    """
    init queries
    :param graph_size:
    :param query_num: number of queries
    :return:
    """

    queries = []
    for i in range(query_num):
        start = np.random.randint(0, graph_size)
        end = np.random.randint(0, graph_size)
        if start == end:
            continue
        query = Query(start, end)
        queries.append(query)

    return queries


def main():
    # some parameters
    query_num = 2000
    graph_size = 18262
    alpha = 2
    beta = 2
    rate = 0.02

    # init graph G
    G = pl.init_real_graph(data)

    # calculate the distances of different routes(all routes)
    distances = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    # init queries
    queries = init_queries(query_num=query_num, graph_size=graph_size)

    labels = []

    init_search(queries, G, distances, labels=labels, alpha=alpha)


def get_flow(node, node_neighbor, query, labels):
    res = 0
    for label in labels:
        if node== label.start_node and node_neighbor == label.end_node \
                and query.timeList[-1] < label.end_time and query.timeList> label.start_time:
            res+=1
    return res


def init_search(queries, G, distances, labels, alpha, beta):
    for query in queries:
        pq = PriorityQueue()
        pq.put(query.start, distances[query.start][query.end])

        while not pq.empty():
            node = pq.get()
            nodes = G.neighbors(node)
            for node_neighbor in nodes:
                car_flow = get_flow(node, node_neighbor, query, labels)
                # 承载量暂定为100
                time_tmp = G.[node][node_neighbor]*(1+ alpha*(car_flow/100)**beta)
                if distances[node][query.end]+time_tmp<= distances[node_neighbor][query.end]:

