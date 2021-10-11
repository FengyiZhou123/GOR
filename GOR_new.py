import heapq

import networkx as nx
import numpy as np
import pandas as pd


def init_real_graph():
    data = pd.read_csv('./data.csv')
    data = data.iloc[:, 1:]
    data = data.to_numpy()
    G = nx.Graph()
    for weight in data:
        G.add_edge(weight[0], weight[1], weight=weight[2])
    G.add_edge(0, 7388, weight=1.4108)

    return G


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


class Label:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        # 暂定为0
        self.timeList = [0]
        self.routes = [start]


def init_labels(query_num, graph_size):
    labels = []
    for i in range(query_num):
        start = np.random.randint(0, graph_size)
        end = np.random.randint(0, graph_size)
        if start == end:
            continue
        label = Label(start, end)
        labels.append(label)
    return labels


def greedy_algorithms(labels, G, distances):
    pq = PriorityQueue()
    for label in labels:
        pq.put(label, label.timeList[-1])

    while not pq.empty():
        label = pq.get()
        nodes = list(G.neighbors(label.routes[-1]))

        for node in nodes:
            if node not in label.routes and distances[node][label.end]< distances[]


def main():
    # 先设置场数，例如query_num等, graph_num=18262, 真实图中的点为18262
    query_num = 2000
    graph_size = 18262
    res = []

    G = init_real_graph()
    distances= dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    labels = init_labels(query_num=query_num, graph_size=graph_size)
    pass


if __name__ == '__main__':
    main()
