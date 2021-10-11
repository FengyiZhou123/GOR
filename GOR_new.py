import datetime
import heapq
import sys

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


def get_flow(init_time, init_node, labels, node):
    car_num = 0
    for label in labels:
        # 从init_node出发而且出发时间早于init_time
        if label.routes[-1] == init_node and label.routes[-1] != label.end and label.timeList[-1] < init_time:
            car_num += 1
    return car_num


def show_answer(res):
    total_cost = 0
    for label in res:
        total_cost += label.timeList[-1] - label.timeList[0]


def greedy_algorithms(labels, G, distances, alpha):
    pq = PriorityQueue()
    res = []
    for label in labels:
        pq.put(label, label.timeList[-1])

    while not pq.empty():
        label = pq.get()
        nodes = list(G.neighbors(label.routes[-1]))

        EGT_tmp = sys.maxsize
        init_time = label.timeList[-1]
        init_node = label.routes[-1]
        for node in nodes:
            if node not in label.routes and distances[node][label.end] < distances[init_node][label.end]:
                car_num = get_flow(init_time, init_node, labels, node)
                time_tmp = G[init_node][node]["weight"] * (1 + alpha * car_num)
                EGT = init_time - label.timeList[0] + time_tmp + distances[node][init_node]
                if EGT_tmp == sys.maxsize and EGT < EGT_tmp:
                    label.routes.append(node)
                    label.timeList.append(init_time + time_tmp)
                    EGT_tmp = EGT
                elif EGT_tmp != sys.maxsize and EGT < EGT_tmp:
                    label.routes[-1] = node
                    label.timeList[-1] = init_time + time_tmp
                    EGT_tmp = EGT

        if label.routes[-1] == label.end:
            res.append(label)

        else:
            pq.put(label, label.timeList[-1])
    return res


def refining_algorithm(combination, G, distances, rate):
    pass


def main():
    # 先设置场数，例如query_num等, graph_num=18262, 真实图中的点为18262
    query_num = 2000
    graph_size = 18262
    alpha = 0.02
    rate = 0.02

    G = init_real_graph()
    distances = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    labels = init_labels(query_num=query_num, graph_size=graph_size)

    # 计算算法一时间和cost
    start_time = datetime.datetime.now()
    res = greedy_algorithms(labels, G, distances, alpha)
    end_time = datetime.datetime.now()
    print("time cost1" + str((end_time - start_time).seconds))
    print("cost1" + str(show_answer(res)))

    # 计算算法二时间和cost
    # start_time = datetime.datetime.now()
    # combination2 = refining_algorithm(res, G, distances, rate)
    # end_time = datetime.datetime.now()
    # print("time cost2" + str((end_time - start_time).seconds))
    # print("cost2" + str(show_answer(res)))


if __name__ == '__main__':
    main()
