import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import json
import argparse
from collections import deque
import sys
import random

'''
Parses the random walk generator arguments.
'''
parser = argparse.ArgumentParser(description="random_pos_and_neg(BFS_pos).")
parser.add_argument('--dataset', type=str, default="uk")
parser.add_argument('--order', type=str, default="10e3")
parser.add_argument('--bfs_path', type=int, default=3)
parser.add_argument('--sim_spatial', type=float, default=0.6)
parser.add_argument('--sim_text', type=float, default=0.6)
parser.add_argument('--pos_num', type=int, default=20) # 每个点选取对应的正样本对 数量 -> 最终有 n * 20 对正样本
parser.add_argument('--neg_num', type=int, default=20) # 每个正样本对 对应的负样本对数量 -> 最终有 n * 20 * 20 对负样本
args = parser.parse_args()

print(args)

# -------------------------1. 加载数据-------------------------------------
# 加载图信息(无向图)
dataset = args.dataset
order = args.order

# G_path = r"/home/laixy/AHG/stgnn/{}/{}_graph(3).json".format(dataset, dataset)
G_path = "/data1/lxy/work2/subgraph_{}/{}/renumerated_info/renumberated_graph.json".format(dataset, order)
f = open(G_path)
js_graph = json.load(f)
# 生成类nxgraph
nx_G = json_graph.node_link_graph(js_graph)
print("边数：", len(nx_G.edges()))
print("点数：", len(nx_G.nodes()))

# sim_spatial_matrix = np.loadtxt(r"/data3/laixy/stgnn/{}/spatial_text_matrix/merge_spatial.txt".format(dataset), dtype=float)
sim_spatial_matrix = np.loadtxt("/data1/lxy/work2/subgraph_{}/{}/renumerated_info/spatial_matrix.txt".format(dataset, order), dtype=float)
# sim_text_matrix = np.loadtxt(r"/data3/laixy/stgnn/{}/spatial_text_matrix/merge_text.txt".format(dataset), dtype=float)
sim_text_matrix = np.loadtxt("/data1/lxy/work2/subgraph_{}/{}/renumerated_info/text_matrix.txt".format(dataset, order), dtype=float)
# id_map = json.load(open(r"/data3/laixy/stgnn/{}/".format(dataset) + "id_map.json"))
# id_map = {int(k): int(v) for k, v in id_map.items()}

print("sim_spatial/text_matrix.shape=", sim_spatial_matrix.shape, sim_text_matrix.shape)
# print("len(id_map)", len(id_map))

# 正负样本相关的文件里面的点均没有重编号！（subgraph的重新编号了！）
nodes_list = list(nx_G.nodes())

print("1. 加载数据已完成！")

# -------------------------2. 在BFS正样本文件中随机抽取正样本和负样本，生成文件pos20_neg20_randomIn_pos_path3_spatial0.6_text0.6.txt-------------------------------------
# 格式：node pos1 spatial1,text1,path1 neg1#neg2#... neg1#neg2#... neg_spatial1,neg_text1,neg_path1#neg_spatial2,neg_text2,neg_path2#...

total_1 = 0
total_2 = 0

# w_path = "/data3/laixy/stgnn/{}/sampling/bfs/pos{}_neg{}_randomIn_pos_path{}_spatial{}_text{}.txt".format(dataset, args.pos_num, args.neg_num, args.bfs_path, args.sim_spatial, args.sim_text)
w_path = "/data1/lxy/work2/subgraph_{}/{}/renumerated_info/pos_neg_sampling/pos{}_neg{}_randomIn_pos_path{}_spatial{}_text{}.txt".format(dataset, order, args.pos_num, args.neg_num, args.bfs_path, args.sim_spatial, args.sim_text)
# r_path = "/data3/laixy/stgnn/{}/sampling/bfs/pos_path{}_spatial{}_text{}.txt".format(dataset, args.bfs_path, args.sim_spatial, args.sim_text)
r_path = "/data1/lxy/work2/subgraph_{}/{}/renumerated_info/pos_neg_sampling/pos_path{}_spatial{}_text{}.txt".format(dataset, order, args.bfs_path, args.sim_spatial, args.sim_text)
with open(w_path, "w") as w_f:

    with open(r_path, "r") as r_f:
        # 格式：node node1#node2#node3#... sptial1,text1#spatial2,text2#... path1#path2#...
        line = r_f.readline().rstrip()
        while line:
            line_list = line.split(" ")

            if len(line_list) > 1:

                node = int(line_list[0])
                pos_arr = np.array([int(p) for p in line_list[1].split("#")]) # int
                pos_spatial_text = np.array([st for st in line_list[2].split("#")]) # str
                pos_path = np.array([path for path in line_list[3].split("#")]) # str

                # 开始随机抽取正负样本
                # 生成正样本
                actual_pos_num = args.pos_num
                if len(pos_arr) < actual_pos_num:
                    actual_pos_num = len(pos_arr)

                total_1 += actual_pos_num

                # 随机抽取正样本下标
                pos_random_idx_list = random.sample(list(range(0, len(pos_arr))), actual_pos_num) 
                # 抽取后的正样本点
                pos_random_arr = pos_arr[pos_random_idx_list]
                # 抽取后的spatail,text信息
                pos_random_spatial_text_arr = pos_spatial_text[pos_random_idx_list]
                # 抽取后的path信息
                pos_random_path_arr = pos_path[pos_random_idx_list]
                # 抽取后的spatial,text,path信息【合并】
                pos_info = []
                for i in range(len(pos_random_arr)):
                    pos_info.append(",".join([pos_random_spatial_text_arr[i], pos_random_path_arr[i]]))
                # 生成所有负样本
                neg_arr = np.array([n for n in nodes_list if n != node and n not in pos_arr])
                neg_random_total = actual_pos_num * args.neg_num
                if neg_random_total > len(neg_arr):
                    print("node={}中，负样本没有那么多可抽取！现在len(neg_list)={}，但是需要{}！".format(node, len(neg_arr), neg_random_total))
                    repeat_num = neg_random_total - len(neg_arr)
                    neg_arr = np.concatenate((neg_arr, np.random.choice(neg_arr, size=repeat_num)))
                    print("现在负样本数量为：{}".format(len(neg_arr)))
                    # sys.exit(-1)
                # 随机抽取负样本下标
                neg_random_idx_list = random.sample(list(range(0, len(neg_arr))), neg_random_total)
                # 抽取后的负样本点
                neg_random_arr = neg_arr[neg_random_idx_list]
                
                neg_i = 0
                start = neg_i * args.neg_num
                end = start + args.neg_num
                for i, pos_other in enumerate(pos_random_arr):

                    total_2 += 1

                    cur_neg_arr = neg_random_arr[start : end]
                    cur_neg_info = []
                    for neg_other in cur_neg_arr:
                        # cur_neg_spatial = str(sim_spatial_matrix[id_map[node]][id_map[neg_other]])
                        cur_neg_spatial = str(sim_spatial_matrix[node][neg_other])
                        # cur_neg_text = str(sim_text_matrix[id_map[node]][id_map[neg_other]])
                        cur_neg_text = str(sim_text_matrix[node][neg_other])
                        cur_neg_path = str(len(nx.shortest_path(nx_G, node, neg_other)) - 1)
                        cur_neg_info.append(",".join([cur_neg_spatial, cur_neg_text, cur_neg_path]))
                    cur_neg_str_list = [str(cur_neg) for cur_neg in cur_neg_arr]
                    w_f.write("{} {} {} {} {}\n".format(str(node), str(pos_other), pos_info[i], "#".join(cur_neg_str_list), "#".join(cur_neg_info)))
                    neg_i += 1
                    start = end
                    end = start + args.neg_num

            line = r_f.readline().rstrip()

print("total_1={}, total_2={}".format(str(total_1), str(total_2)))
print("finish!")