import heapq

import networkx as nx
import numpy as np

import planning as pl
import pandas as pd

data = pd.read_csv('./data.csv')
data = data.iloc[:, 1:]
data = data.to_numpy()

class Node:
    def __init__(self, node):
        self.node=node
        self.prev=None

class Query:
    def __init__(self, start, end):
        self.start=Node(start)
        self.end=Node(end)
        self.routes=[]
        self.timeList=[]

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

    queries=[]
    for i in range(query_num):
        start = np.random.randint(0, graph_size)
        end = np.random.randint(0, graph_size)
        if start==end:
            continue
        query = Query(start, end)
        queries.append(query)

    return queries




def main():
    # init graph G
    G = pl.init_real_graph(data)

    # calculate the distances of different routes(all routes)
    distances= dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))



def init_search(queries, G, distances):
    for query in queries:
        pq = PriorityQueue()
        pq.put(query.start, distances[query.start][query.end])

        while not pq.empty():
            node = pq.get()
            if node== query.end:
                pass





