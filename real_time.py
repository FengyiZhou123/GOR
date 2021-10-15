import networkx as nx
import numpy as np
import pandas as pd
import datetime
import sys
import utils


class Query:
    def __init__(self, start, end, routes, K):
        self.start = start
        self.end = end
        self.routes = routes
        # dict1 means total value of the routes, dict2 means the routes of the route
        dict1 = []
        dict2 = []
        for i in range(K):
            dict1.append(0)
            dict2.append(routes[i])


# def init_graph():
#     """
#     init graph
#     :param graph_size:
#     :return:
#     """
#     data = pd.read_csv('./data.csv')
#     data = data.iloc[:, 1:]
#     data = data.to_numpy()
#
#     G = nx.Graph()
#     for weight in data:
#         G.add_weighted_edges_from([(weight[0], weight[1], weight[2])])
#     G.add_weighted_edges_from([(0, 7388, 1.4108)])
#     return G


def init_queries(graph_size, query_numbers, G, K):
    """
    init queries
    :param graph_size:
    :param query_numbers:
    :param G:
    :param K:
    :return:
    """
    queries = []
    i = 0
    while i < query_numbers:
        start = np.random.randint(0, graph_size)
        end = np.random.randint(0, graph_size)
        if start != end:
            routes, routes_len = utils.k_shortest_paths(G=G, source=start, target=end, k=K, weight='weight')
            query = Query(start, end, routes, K)
            queries.append(query)
            i += 1

    return queries


# def init_carNum(graph_size):
#     G_matrix=np.zeros((graph_size,graph_size), dtype=int)
#     return G_matrix


def planning(epsilon, graph_size, alpa, param, K):
    """
    main part of algorithm
    :param epsilon:
    :return:
    """
    # init graph
    G = utils.init_real_graph()

    # init queries
    queries = init_queries(graph_size, 2000, G, K)

    start = datetime.datetime.now()
    while (datetime.datetime.now() - start).seconds < 10:
        # init a matrix to memorize the number of each edge
        G_carNum = np.zeros((graph_size, graph_size), dtype=int)
        # init a matrix to memorize the reward of each edge
        G_reward = np.zeros((graph_size, graph_size), dtype=int)

        # choose action (from k paths)
        routes = []
        indexs = []
        for query in queries:
            route, index = select_from_routes(epsilon, query)
            routes.append(route)
            indexs.append(index)

        # memorize car numbers in every edge
        for route in routes:
            for i in range(len(route) - 1):
                G_carNum[route[i]][route[i + 1]] += 1

        for i in range(graph_size):
            for j in range(graph_size):
                if G.has_edge(i, j):
                    Ce0 = G.get_edge_data(i, j)["weight"] * (1 + alpa * G_carNum[i][j]) * G_carNum[i][j]
                    Ce1 = G.get_edge_data(i, j)["weight"] * (1 + alpa * (G_carNum[i][j] - 1)) * (G_carNum[i][j] - 1)
                    G_reward[i][j] = Ce1 - Ce0

        # update Q value
        index_tmp = 0
        for route in routes:
            reward = 0
            for i in range(len(route) - 1):
                reward += G_reward[route[i]][route[i + 1]]

            queries[index_tmp].dict1[indexs[index_tmp]] += param * (
                    reward - queries[index_tmp].dict1[indexs[index_tmp]])
            index_tmp += 1

    # end the while loop, select the max Q_value routes
    res_routes = []
    # res_time=[]

    for query in queries:
        max_value = 0
        route = []
        for i in range(len(query.dict1)):
            if query.dict1[i] > max_value:
                max_value = query.dict1[i]
                route = query.dict2[i]
        # time=0
        # for i in range(len(route)-1):
        #     time+=G.get_edge_data(i,i+1)["weight"]
        #
        # res_time.append(time)
        res_routes.append(route)

    return res_routes


def select_from_routes(epsilon, query):
    """
    select one of k paths(epsilon-greedy)
    :param epsilon:
    :param query:
    :return:
    """
    probabilty = np.random.random()
    max, tmp = 0, -1

    for i in range(len(query.dict1)):
        if query.dict1[i] > max:
            max = query.dict1[i]
            tmp = i

    if probabilty > epsilon:
        return query.dict2[tmp], tmp

    index = np.random.randint(0, len(query.dict1))
    while index == tmp:
        index = np.random.randint(0, len(query.dict1))

    # routes and index of the route
    return query.dict2[index], index


def main():
    epsilon = 0.1
    graph_size = 18262
    alpha = 0.2
    param = 0.1
    K = 3

    planning(epsilon, graph_size, alpha, param, K=K)


if __name__ == '__main__':
    main()

# def KSP(G, start, end, K):
#     routes = []
#     route = nx.dijkstra_path(G, start, end, weight="weight")
#     routes.append(route)
#     for i in range(K - 1):
#         min_route = []
#         min_value = sys.maxsize
#         for route in routes:
#             for i in range(len(route) - 1):
#                 delete_start = route[i]
#                 delete_end = route[i + 1]
#                 if delete_start == start or delete_end == end:
#                     continue
#                 G.remove_edge(delete_start, delete_end)
#                 tmp_value = nx.dijkstra_path_length(G, start, end, weight="weight")
#                 tmp_route = nx.dijkstra_path(G, start, end, weight="weight")
#                 if (tmp_value < min_value):
#                     min_route = tmp_route
#                     min_value = tmp_value
#         routes.append(min_route)
#     return routes
