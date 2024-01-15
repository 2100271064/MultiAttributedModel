# import math
# import pandas as pd
import itertools
import random
from joblib import Parallel, delayed

import time
import datetime
import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import json
import sys
import argparse


'''
Parses the random walk generator arguments.
'''
parser = argparse.ArgumentParser(description="Run random walk generator.")
parser.add_argument('--dataset', type=str, default="uk")
# parser.add_argument('--pos_num_walks', type=int, default=15)
# parser.add_argument('--neg_num_walks', type=int, default=30)
# parser.add_argument('--pos_walk_length', type=int, default=6)
# parser.add_argument('--neg_walk_length', type=int, default=12)
# parser.add_argument('--sim_theta', type=float, default=0.36) # spatial>=0.6 text>=0.6 st>=0.36【有的st值会超过1，但是也不影响】
parser.add_argument('--walk_length', type=int, default=5)
parser.add_argument('--walk_nums', type=int, default=15)
parser.add_argument('--sim_spatial', type=float, default=0.6)
parser.add_argument('--sim_text', type=float, default=0.6)
args = parser.parse_args()

print(args)

def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def my_walk(self, walk_length, start_node):
        # 存储了每个节点邻居的接收概率、别名表
        alias_nodes = self.alias_nodes
        neighbors_nodes = self.neighbors_nodes
        walk = [start_node]
        stay_prop = 0
        while len(walk) < walk_length:
            cur = walk[-1]
            # print("{}的{}个邻居中，最大sim为:{}".format(str(cur), len(cur_nbrs), np.max(sim_matrix[[id_map[n] for n in cur_nbrs]])))
            # 有邻居（有Accept、别名表）继续游走
            if cur in self.alias_nodes:
                # random.random() : [0, 1)
                if random.random() >= stay_prop: # 随机值 ＞= 继续留在该节点的概率
                    # 根据节点的接受表、别名表进行采样
                    # neighbor_index = self.alias_sample(alias_nodes[cur][0], alias_nodes[cur][1], cur, cur_nbrs)
                    neighbor_index = self.alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])
                    # 没有有效采样的邻居跳出
                    # if neighbor_index == -1:
                    #     break
                    walk.append(neighbors_nodes[cur][neighbor_index])
            # 没邻居跳出
            else:
                break
        # print("是正样本采样：{}，以{}为起点的随机游走中，walk长度为{}".format(args.pos_sampling, start_node, len(walk)))
        if len(walk) != walk_length:
            print("RandomWalker-my_walk：随机游走长度不满足walk_length{}".format(walk_length))
        return walk

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk


    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G
        nodes = list(G.nodes())

        # 并行执行self._simulate_walks方法
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self.partition_num(num_walks, workers))

        # 将多个子列表连接成一个单一的列表
        walks, max_walk_len, min_walk_len = list(itertools.chain(*results))

        return walks, max_walk_len, min_walk_len

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        # num_walks=2
        i = 0
        # 最大最小应该都是 walk_length*num_walks-(num_walks-1)（5*15-14=61）
        max_walk_len = 0
        min_walk_len = num_walks * walk_length + 1
        # TODO test
        expectedLen = walk_length * num_walks - (num_walks - 1);
        for v in nodes:
            walk = []
            for _ in range(num_walks):
                # cur_walk长度至少大于等于1
                cur_walk = self.my_walk(
                    walk_length=walk_length, start_node=v)

                if len(walk) == 0:
                    walk = cur_walk
                elif len(cur_walk) > 1:
                    walk.extend(cur_walk[1:])
            if i < 5:
                print("cur_walk={}, walk={}".format(cur_walk, walk))
                i += 1
            # TODO test
            if len(walk) != expectedLen:
                print("RandomWalker-_simulate_walks：", walk, len(walk))
            if len(walk) > 1: # 如果长度为1说明没有和它匹配的，只有它自己
                walks.append(walk)
                max_walk_len = max(max_walk_len, len(walk))
                min_walk_len = min(min_walk_len, len(walk))
        return walks, max_walk_len, min_walk_len
        # for _ in range(num_walks):
        #     random.shuffle(nodes)
        #     i = 0
        #     for v in nodes:
        #         i+=1
        #         if i % 100000 == 0:
        #             print(i)
        #         # if self.p == 1 and self.q == 1:
        #         #     walks.append(self.deepwalk_walk(
        #         #         walk_length=walk_length, start_node=v))
        #         # elif self.use_rejection_sampling:
        #         #     walks.append(self.node2vec_walk2(
        #         #         walk_length=walk_length, start_node=v))
        #         # else:
        #         # walks的元素：长度为walk_length=5的元素下标列表
        #         walk = self.my_walk(
        #             walk_length=walk_length, start_node=v)
        #         if len(walk) > 1: # 如果长度为1说明没有和它匹配的，只有它自己
        #             walks.append(walk)
        # return walks

    # def my_preprocess(self):
    #     """
    #     Preprocessing of random walks.
    #     """
    #     G = self.G
    #     pos_neighbors_set = set()
    #     neg_neighbors_set = set()
    #     i = 0
    #     for node in G.nodes():
    #         if i < 30:
    #             print("邻居个数：", len(G[node]))
    #             i += 1
    #         # 获取node邻居边权重，没有则默认为1
    #         if args.pos_sampling:
    #             # 正样本 + s1*s2≥(0.6*0.6=0.36)，才采样
    #             tmp_list = [nbr for nbr in G.neighbors(node) if
    #                         (nbr != node) and (sim_matrix[id_map[node]][id_map[nbr]] >= args.sim_theta)]
    #             if len(tmp_list) == 0:
    #                 continue
    #             pos_neighbors_set.add(node)
    #         else:
    #             # 负样本 + s1*s2<(0.6*0.6=0.36)，才采样
    #             tmp_list = [nbr for nbr in G.neighbors(node) if
    #                         (nbr != node) and (sim_matrix[id_map[node]][id_map[nbr]] < args.sim_theta)]
    #             if len(tmp_list) == 0:
    #                 continue
    #             neg_neighbors_set.add(node)
    #
    #     self.pos_neighbors_set = pos_neighbors_set
    #     self.neg_neighbors_set = neg_neighbors_set
    #     return

    # def alias_sample(self, start_id, cur_nbrs):
    #     N = len(cur_nbrs)
    #     # np.random.random() : [0, 1)
    #     # i : [0, N) 整数
    #     i = int(np.random.random()*N) # 随机选了一个邻居下标
    #     return i
    #     # """
    #     # :param accept:
    #     # :param alias:
    #     # :return: sample index
    #     # """
    #     # N = len(accept)
    #     # # np.random.random() : [0, 1)
    #     # # i : [0, N) 整数
    #     # i = int(np.random.random()*N) # 随机选了一个邻居下标
    #     # r = np.random.random() # 随机值
    #     # if r < accept[i]: # 随机值 在 接受概率 范围内
    #     #     return i
    #     # else: # 随机值 不在 接受概率 范围内 (我们不会进入)
    #     #     print("alias_sample采样时进入了else！")
    #     #     return alias[i] # 根据别名表 选择别名
    #
    #     # N = len(accept)
    #     # invalid_neigh_set = set()
    #     # # np.random.random() : [0, 1)
    #     # # i : [0, N) 整数
    #     # i = int(np.random.random() * N)  # 随机选了一个邻居下标
    #     # while (len(invalid_neigh_set) < N):
    #     #     r = np.random.random()  # 随机值
    #     #     if r < accept[i]:  # 随机值 在 接受概率 范围内
    #     #         if args.pos_sampling:
    #     #             return i
    #     #         elif not self.isPosSample(center_id, cur_nbrs[i]):  # 负样本
    #     #             return i
    #     #     else:  # 随机值 不在 接受概率 范围内 (我们不会进入)
    #     #         print("alias_sample采样时进入了else！")
    #     #         return alias[i]  # 根据别名表 选择别名
    #     #     invalid_neigh_set.add(i)
    #     #     if len(invalid_neigh_set) >= N:
    #     #         break
    #     #     i = int(np.random.random() * N)
    #     #     while (i in invalid_neigh_set):
    #     #         i = int(np.random.random() * N)  # 随机选了一个邻居下标
    #     # # 说明找不到
    #     # return -1

    # 每个线程，对每个节点执行随机游走的次数
    def partition_num(self, num, workers):
        # num : 2
        # workers : 1
        if num % workers == 0:
            # [2]
            return [num//workers]*workers
        else:
            return [num//workers]*workers + [num % workers]

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        neighbors_nodes = {}
        i = 0
        for node in G.nodes():
            # 获取node邻居边权重，没有则默认为1
            # 正样本 + s1*s2≥(0.6*0.6=0.36)，才采样
            tmp_list = [nbr for nbr in G.neighbors(node) if (nbr != node)]
            if len(tmp_list) == 0:
                print("RandomWalker-preprocess_trainsition_probs：点{}的有效邻居数为0！".format(node))
                continue
            if i < 30:
                print("{}在游走中有效的邻居个数：".format(node), len(tmp_list))
                i+=1
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in tmp_list]

            # 归一化
            norm_const = sum(unnormalized_probs)
            if norm_const == 0:
                normalized_probs = [1 / len(unnormalized_probs) for u_prob in unnormalized_probs]
                # print(normalized_probs)
            else:
                normalized_probs = [
                    float(u_prob)/norm_const for u_prob in unnormalized_probs]

            # 给当前节点(邻居)创建接受表accept、别名表alias
            alias_nodes[node] = self.create_alias_table(normalized_probs)
            neighbors_nodes[node] = tmp_list

        self.alias_nodes = alias_nodes
        self.neighbors_nodes = neighbors_nodes
        return



    # 创建别名表，用以加速随机采样
    def create_alias_table(self, area_ratio):
        """

        :param area_ratio: sum(area_ratio)=1
        :return: accept,alias
        """
        # area_ratio : normalized_probs
        l = len(area_ratio)
        # accept : 存储每个元素的接受概率
        # alias : 存储每个元素的别名
        accept, alias = [0] * l, [0] * l
        # small : 存储概率小于1的下标
        # large : 存储概率大于等于1的下标
        small, large = [], []
        # 每个元素乘l
        area_ratio_ = np.array(area_ratio) * l
        # 分类存储下标
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            # 每次从small、large各取一个元素
            small_idx, large_idx = small.pop(), large.pop()
            # 概率小于1，则直接设置接受概率
            accept[small_idx] = area_ratio_[small_idx]
            # 当不接受small_idx时，需要设置另一个元素
            alias[small_idx] = large_idx
            # 更新剩余概率，这是为了保证：概率大于等于1的元素和概率小于1的元素之间的总和仍然是1
            area_ratio_[large_idx] = area_ratio_[large_idx] - \
                (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        # 剩余的 large 和 small，将其接受概率设置为1
        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        # 通过查询接收表、别名表，可以高效地从给定的概率分布中进行随机采样
        return accept, alias


    def alias_sample(self, accept, alias):
        """

        :param accept:
        :param alias:
        :return: sample index
        """
        N = len(accept)
        # np.random.random() : [0, 1)
        # i : [0, N) 整数
        i = int(np.random.random()*N) # 随机选了一个邻居下标
        r = np.random.random() # 随机值
        if r < accept[i]: # 随机值 在 接受概率 范围内
            return i
        else: # 随机值 不在 接受概率 范围内
            print("采样时进入了别名表！")
            return alias[i] # 根据别名表 选择别名

    # 判断是不是正样本
    # def isPosSample(self, center_id, i):
    #     if len(pos_dict) == 0:
    #         print("正样本还没生成！")
    #         sys.exit(-1)
    #     else:
    #         if center_id in pos_dict and i in pos_dict[center_id]:
    #             return True
    #         elif i in pos_dict and center_id in pos_dict[i]:
    #             return True
    #         else:
    #             return False


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):
        print("p:", p)
        print("q:", q)
        print("use_rejection_sampling:", use_rejection_sampling)
        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)

        print("Preprocess transition probs...")
        # 计算每个点的归一化邻居概率，并创建接受表和别名表
        self.walker.preprocess_transition_probs()
        # print("Preprocess...")
        # self.walker.my_preprocess()
        print("Preprocess end...")
        # verbose : 输出详细信息的级别
        self.sentences, self.max_walk_len, self.min_walk_len = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)
        print("正样本随机游走结果：len(self.sentences)={}，其中最大的游走长度为：{}，最小的游走长度为：{}".format(len(self.sentences), self.max_walk_len, self.min_walk_len))



'''
Get walks for graph.
'''
# -------------------------1. 加载数据-------------------------------------
start = time.time()
# 加载图信息(无向图)
dataset = args.dataset
G_path = r"/home/laixy/AHG/stgnn/{}/{}_graph(3).json".format(dataset, dataset)
f = open(G_path)
js_graph = json.load(f)
# 生成类nxgraph
nx_G = json_graph.node_link_graph(js_graph)
print("边数：", len(nx_G.edges()))
print("点数：", len(nx_G.nodes()))
# 加载sim(s1*s2)矩阵
# sim_matrix = np.loadtxt(r"/data3/laixy/stgnn/{}/spatial_text_matrix/spatial_mul_text.txt".format(dataset), dtype=float)
# TODO 打开
sim_spatial_matrix = np.loadtxt(r"/data3/laixy/stgnn/{}/spatial_text_matrix/merge_spatial.txt".format(dataset), dtype=float)
sim_text_matrix = np.loadtxt(r"/data3/laixy/stgnn/{}/spatial_text_matrix/merge_text.txt".format(dataset), dtype=float)
id_map = json.load(open(r"/data3/laixy/stgnn/{}/".format(dataset) + "id_map.json"))
id_map = {int(k): int(v) for k, v in id_map.items()}



print("sim_spatial/text_matrix.shape=", sim_spatial_matrix.shape, sim_text_matrix.shape)
print("len(id_map)", len(id_map))

print("1. 加载数据已完成！")

# --------------------------2. 生成随机游走序列（可能是非正样本，或者含有起点，或重复的点）----------------------------------

# 参数：
# nx_graph : networkx图
# walk_length : 游走步长(默认5)
# num_walks : 每个节点随机游走的次数
# p : 返回(回头)到先前节点的概率(默认1)
# q : 转移到相邻节点(BFS)/远离节点的权衡(DFS)(默认1)——q<1，更倾向DFS，q>1，更倾向BFS，q=1，退化为DeepWalk
# workers : 工作线程数(默认1)
# user_rejection_sampling : 是否使用拒绝抽样，如果是(1)，会根据节点之间的相似度进行节点选择；否则(0)按照一般的随机游走。(默认1)
model = Node2Vec(nx_G, walk_length=args.walk_length, num_walks=args.walk_nums, p=1, q=1, workers=1, use_rejection_sampling=1)
print("耗时：", time.time() - start)

# TODO 测试结束
# sys.exit(0)

walk_path = "/data3/laixy/stgnn/{}/sampling/walkLen{}_walkNum{}.txt".format(dataset, args.walk_length, args.walk_nums)
with open(walk_path, "w") as f1:
    # num_walk * node.size() 个walk列表
    for walk in model.sentences:
        # 去重 + 后面序列中去除自己
        start_node = walk[0]
        other_set = set()
        for other in walk[1:]:
            if other != start_node:
                other_set.add(other)
        new_walk = [start_node]
        for other in other_set:
            new_walk.append(other)
        f1.write(" ".join([str(w) for w in new_walk])+"\n")

print("2. 随机游走序列已完成！")


# ------------------------------3. 生成正样本文件---------------------------------------
# 格式：node node1#node2#node3#... sptial1,text1#spatial2,text2#...
# TODO 后面再补充一部分path信息到文件

sampling_type = "pos"
pos_count = 0

sample_path = "/data3/laixy/stgnn/{}/sampling/{}_sim_walkLen{}_walkNum{}_spatial{}_text{}.txt".format(dataset, sampling_type, args.walk_length, args.walk_nums, args.sim_spatial, args.sim_text)
with open(walk_path, "r") as f2:
    with open(sample_path, "w") as f3:
        line = f2.readline().rstrip()

        while line:
            line_list = line.split(" ")
            cur = int(line_list[0])
            other_str_list = line_list[1:]
            # if args.pos_sampling:
            #     # pos_dict填充
            #     if cur not in pos_dict:
            #         pos_dict[cur] = dict()

            node_list = []
            sim_str_list = []
            path_str_list = []

            for other_str in other_str_list:
                # 已经在生成随机序列的时候排序了：重复 以及 cur==other
                other = int(other_str)
                cur_spatial = sim_spatial_matrix[id_map[cur]][id_map[other]]
                if cur_spatial > 1:
                    cur_spatial = 1
                cur_text = sim_text_matrix[id_map[cur]][id_map[other]]
                if cur_text > 1:
                    cur_text = 1
                # 正样本要在地理坐标上邻近，也要在文本语义上邻近
                if cur_spatial >= args.sim_spatial and cur_text >= args.sim_text:
                    node_list.append(other_str)
                    # 格式：spatial_sim,text_sim
                    cur_sim = "{},{}".format(str(cur_spatial), str(cur_text))
                    sim_str_list.append(cur_sim)
                    # cur_path = (str)...
                    # path_str_list.append(cur_path)

            if len(node_list) == 0:
                node_list = ["-1"]
                sim_str_list = ["-1"]
                path_str_list = ["-1"]
            elif len(path_str_list) == 0:
                path_str_list = ["-1"]
            # f3.write("{} {} {}\n".format(str(cur), "#".join(node_list), "#".join(sim_str_list)))
            f3.write("{} {} {} {}\n".format(str(cur), "#".join(node_list), "#".join(sim_str_list), "#".join(path_str_list)))
            pos_count = pos_count + len(node_list)

            line = f2.readline().rstrip()

n = len(nx_G.nodes())
all_count = n * n - n
print("正样本共有{}对(不包括自己和自己)，总对数为{}(n×n-n)，占比：{:.2f}%".format(pos_count, all_count, (pos_count/all_count) * 100))

print("3. 正样本文件已完成！")

# # ------------------------------4. 生成负样本序列---------------------------------------
#
# # 加载pos_dict文件?
#
# sampling_type = "neg"
#
# model = Node2Vec(nx_G, walk_length=args.neg_walk_length, num_walks=args.neg_num_walks, p=1, q=1, workers=1, use_rejection_sampling=1)
# # print("耗时：", time.time() - start)
#
# origin_sample_path = "/data3/laixy/stgnn/{}/sampling/{}_walkLen{}_walkNum{}_theta{}.txt".format(dataset, sampling_type, args.neg_walk_length, args.neg_num_walks, args.sim_theta)
# with open(origin_sample_path, "w") as f4:
#     # num_walk * node.size() 个walk列表
#     for walk in model.sentences:
#         f4.write(" ".join([str(w) for w in walk])+"\n")
#
# print("4. 生成负样本序列已完成！")

# ------------------------------4. 生成负样本文件（不参与随机游走）---------------------------------------
# 格式：开始点 节点集1 相似度1 节点集2 相似度2 路径长度2

#      节点集1、相似度1：不符合spatial或者text（path可能符合也可能不符合）
#      节点集2、相似度2：在符合spatial和text条件下，path超过的

#      节点集格式：node1#node2#node3#...
#      相似度格式：spatial1,text1#spatial2,text2#...
#      路径长度格式：path1#path2#...

sampling_type = "neg"
neg_count = 0
sample_path = "/data3/laixy/stgnn/{}/sampling/{}_sim_walkLen{}_spatial{}_text{}.txt".format(dataset, sampling_type, args.walk_length, args.sim_spatial, args.sim_text)
with open(sample_path, "w") as f4:

    for node in nx_G.nodes():

        stNodes_str_list = []
        pathNodes_str_list = []
        stSim_str_list = []
        pathSim_str_list = []
        path_str_list = []

        for other in nx_G.nodes():
            if other == node:
                continue
            cur_spatial = sim_spatial_matrix[id_map[node]][id_map[other]]
            cur_text = sim_text_matrix[id_map[node]][id_map[other]]
            # cur_path = ...
            # 正样本要在地理坐标上邻近，也要在文本语义上邻近
            if cur_spatial < args.sim_spatial or cur_text < args.sim_text:
                stNodes_str_list.append(str(other))
                cur_sim = "{},{}".format(str(cur_spatial), str(cur_text))
                stSim_str_list.append(cur_sim)
            # elif cur_path > args.walk_length:
            #     pathNodes_str_list.append(str(other))
            #     cur_sim = "{},{}".format(str(cur_spatial), str(cur_text))
            #     pathSim_str_list.append(cur_sim)
            #     path_str_list.append(str(cur_path))
        if len(stNodes_str_list) == 0:
            stNodes_str_list = ["-1"]
            stSim_str_list = ["-1"]
        if len(pathNodes_str_list) == 0:
            pathNodes_str_list = ["-1"]
            pathSim_str_list = ["-1"]
            path_str_list = ["-1"]
        f4.write("{} {} {} {} {} {}\n".format(str(node), "#".join(stNodes_str_list), "#".join(stSim_str_list),
                                    "#".join(pathNodes_str_list), "#".join(pathSim_str_list), "#".join(path_str_list)))
        neg_count = neg_count + len(stNodes_str_list) + len(pathNodes_str_list)

print("负样本共有{}对(不包括自己和自己)，总对数为{}(n×n-n)，占比：{:.2f}%".format(neg_count, all_count, (neg_count/all_count) * 100))

print("4. 负样本文件已完成！")

