import datetime
import sys

import utils
import pickle
from GOR_new import PriorityQueue


class Path:
    def __init__(self, routes, routes_len, start, end):
        self.start = start
        self.end = end
        self.routes = routes
        self.routes_len = routes_len


class Query:
    def __init__(self, route, start, end):
        self.route = route
        self.start = start
        self.end = end
        self.index = 0
        # 可以设置为有初始时间
        self.timeList = [0]


class FactorGraph:

    def __init__(self):
        self.variable_nodes = list()
        self.function_nodes = list()

    def find_function_node(self, start, end):
        for node in self.function_nodes:
            if node.start == start and node.end == end:
                return node
        return None

    def get_routes(self):
        routes = []
        index = 0
        # 对于每个variable_node 都找出一个最短的路径和

        for variable_node in self.variable_nodes:
            min_cost = sys.maxsize

            for route in variable_node.routes:
                tmp = 0
                for i in range(len(route) - 1):
                    start = route[i]
                    end = route[i + 1]
                    tmp_node = self.find_function_node(start, end)
                    tmp += variable_node.reward[tmp_node.id]
                if min_cost == sys.maxsize:
                    min_cost = tmp
                    routes.append(route)
                elif tmp < min_cost:
                    min_cost = tmp
                    routes[index] = route
            index += 1
        return routes


class VariableNode:
    def __init__(self, routes, id):
        self.neighbors = list()
        # 一个variable node有多个route
        self.routes = routes
        self.id = id
        # key ：第k-1次迭代发送给该node值的node的id
        self.reward = dict()

    # alpha 值待计算
    def get_alpha(self, id):
        alpha = [0, 0]
        return alpha

    def send_message(self):
        sum_tmp = [0, 0]
        # 先计算总的reward
        for node in self.neighbors:
            # 初始数值暂定为0
            sum_tmp[0] += self.reward.get(node.id, [0, 0])[0]
            sum_tmp[1] += self.reward.get(node.id, [0, 0])[1]

        # 给每个function node 发送message 意思是选0或者1 的代价
        for node in self.neighbors:
            reward_tmp = [0, 0]
            reward_tmp[0] = sum_tmp[0] - self.reward.get(node.id, [0, 0])[0] - self.get_alpha(node.id)[0]
            reward_tmp[1] = sum_tmp[1] - self.reward.get(node.id, [0, 0])[1] - self.get_alpha(node.id)[1]
            # send_message, 即把reward值赋给对应的function_node
            node.reward[self.id] = reward_tmp

    def get_min_cost_route(self):
        pass
    # def receive_message(self):
    #     pass


class FunctionNode:
    def __init__(self, start, end, id):
        self.neighbors = list()
        self.id = id
        self.start = start
        self.end = end
        # 用来存储上一次迭代发的reward
        self.reward = dict()

    def get_edge_cost(self, G, car_num, alpa):
        start = self.start
        end = self.end
        cost = G[start][end]['weight']
        return cost * (1 + alpa * car_num) * car_num

    def send_message(self, G):
        # 每个function send 一个message给相邻的variable node，也是一个list
        # 大致可以理解为 variable node 选0和选1 所耗费的cost， 且找到使其最小的分配
        total_reward = [0, 0]

        for variable_node in self.neighbors:
            tmp = list()
            send_id = variable_node.id
            car_num = 0
            for node in self.neighbors:
                if send_id == node.id:
                    continue
                tmp.append(self.reward.get(node.id, [0, 0]))

            for i in range(len(tmp)):
                for j in range(len(tmp[i])):
                    pass


def load_path():
    """load k_shortest paths"""
    with open('k_shortest_paths', 'rb') as f:
        path = pickle.load(f)
    return path


def save_path(G, queries):
    paths = []
    with open('k_shortest_paths', 'wb') as f:
        for i in queries:
            start = i[0]
            end = i[1]
            routes, routes_len = utils.k_shortest_paths(G, start, end, 3, 'weight')
            path = Path(routes=routes, routes_len=routes_len, start=start, end=end)
            paths.append(path)
        pickle.dump(paths, f)


def get_drive_time(node_start, node_end, label, time, G, alpha):
    car_num = 0
    for label_key in label:
        if node_start == label_key[0] and node_end == label_key[1]:
            time_tmp = label.get(label_key)
            if time_tmp[0] < time < time_tmp[1]:
                car_num += 1
    drive_time = G[node_start][node_end]['weight'] * (1 + alpha * car_num)
    return drive_time


def calculate_time(routes, G, alpha):
    res = 0
    label = dict()
    pq = PriorityQueue()
    for route in routes:
        query = Query(route, route[0], route[-1])
        pq.put(query, query.timeList[-1])

    while not pq.empty():
        query_tmp = pq.get()
        node_start = query_tmp.route[query_tmp.index]
        node_end = query_tmp.route[query_tmp.index + 1]
        drive_time = get_drive_time(node_start, node_end, label, query_tmp.timeList[-1], G, alpha=alpha)
        query_tmp.timeList.append(query_tmp.timeList[-1] + drive_time)
        label[node_start, node_end] = [query_tmp.timeList[-2], query_tmp.timeList[-1]]
        res += drive_time
        query_tmp.index += 1
        if node_end != query_tmp.end:
            pq.put(query_tmp, query_tmp.timeList[-1])

    return res


def change_to_factor_graph(paths):
    # 先初始化所有的function_node（去除重复的部分）， 然后初始化variable node（不可能重复）， 从保存的list集合里面建立连接（variable node->
    # function_node, 同时function node 再把variable node放入neighbor中）
    graph = FactorGraph()
    variable_index = 0
    function_index = 0
    for path in paths:
        routes = path.routes
        node = VariableNode(routes=routes, id=variable_index)
        graph.variable_nodes.append(node)
        variable_index += 1
        for route in routes:
            for i in range(len(route) - 1):
                start = route[i]
                end = route[i + 1]
                node_tmp = graph.find_function_node(start, end)
                if node_tmp is not None:
                    node_tmp.neighbors.append(node)
                    node.neighbors.append(node_tmp)
                else:
                    function_node = FunctionNode(start, end, function_index)
                    graph.function_nodes.append(function_node)
                    function_node.neighbors.append(node)
                    node.neighbors.append(function_node)
                    function_index += 1
    return graph


def main():
    query_num = 2000
    graph_size = 18262
    # generate graph
    G = utils.init_real_graph()

    # 随机初始化要生成的所有queries,初次是生成的，后来用存储的
    # queries = utils.init_queries(query_num, graph_size)
    # save_path(G, queries)

    paths = load_path()

    graph = change_to_factor_graph(paths)

    # 之后再设置终止条件,先设置为迭代100次
    k = 0
    start_time = datetime.datetime.now()
    while k < 10:
        for variable_node in graph.variable_nodes:
            # 先是variable node发送消息
            variable_node.send_message()

        # 然后function node 发送信息
        for function_node in graph.function_nodes:
            function_node.send_message(G)
        k += 1
        # 之后选取最少cost的路径

    end = datetime.datetime.now()
    # print('时间=====================')
    # print((end - start_time).seconds)
    routes = graph.get_routes()

    # 统一标准，计算消耗的时间,用以对比
    total_time = calculate_time(routes=routes, G=G, alpha=0.2)


if __name__ == '__main__':
    main()
