import opt
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import scipy.sparse as sp
from networkx.readwrite import json_graph
import networkx as nx
import json
from sklearn.decomposition import PCA
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup():
    """
    setup
    Return: None

    """
    print("setting:")
    setup_seed(opt.args.seed)


    # GPU是否开启
    if torch.cuda.is_available() and opt.args.cuda:
        print("Available GPU")
        opt.args.device = torch.device("cuda")
    else:
        print("Using CPU")
        opt.args.device = torch.device("cpu")

    if opt.args.switch == 1:
        # 128→256→20 ①
        opt.args.gae_n_enc_1 = 128
        opt.args.gae_n_enc_2 = 256
        opt.args.gae_n_dec_2 = 256
        opt.args.gae_n_dec_3 = 128
    elif opt.args.switch == 2 or opt.args.switch == 3:
        # 64(1)→256→20 ②
        # 64(2)→256→20 ③
        opt.args.gae_n_enc_1 = 64
        opt.args.gae_n_enc_2 = 256
        opt.args.gae_n_dec_2 = 256
        opt.args.gae_n_dec_3 = 64
    elif opt.args.switch == 4 or opt.args.switch == 5:
        # 64(1)→128→20 ④
        # 64(2)→128→20 ⑤
        opt.args.gae_n_enc_1 = 64
        opt.args.gae_n_enc_2 = 128
        opt.args.gae_n_dec_2 = 128
        opt.args.gae_n_dec_3 = 64

    print("------------------------------")
    print("dataset       : {}".format(opt.args.dataset))
    print("order      : {}".format(opt.args.order))
    print("model        : {}".format(opt.args.model))
    print("cuda        : {}".format(opt.args.cuda))
    print("gpu        : {}".format(opt.args.gpu))
    print("random seed   : {}".format(opt.args.seed))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("epoch        : {}".format(opt.args.epoch))
    print("switch       : {}".format(opt.args.switch))
    print("layer1       : {}".format(opt.args.gae_n_enc_1))
    print("layer2       : {}".format(opt.args.gae_n_enc_2))
    print("visdom_port  : {}".format(opt.args.visdom_port))
    print("isVersionTwo : {}".format(opt.args.isVersionTwo))
    print("------------------------------")


def load_data():

    dataset = opt.args.dataset
    order = opt.args.order

    # root_path = r"/home/laixy/work2/dataset/{}/".format(dataset)
    root_path = r"/home/laixy/work2/dataset/{}/subgraph/{}/".format(dataset, order)

    feature_dim = 128
    if opt.args.isVersionTwo and dataset == 'us':
        feature_dim = 64

    if opt.args.model < 3:
        # 1. 加载图信息
        topology_graph_path = root_path + r"graph.json"
        topology_nx_graph = json_graph.node_link_graph(json.load(open(topology_graph_path)))
        print("原始graph:", nx.info(topology_nx_graph))
        # TODO adj不为sp
        topology_adj = process_adj(topology_nx_graph)
        # topology_adj = nx.to_numpy_array(topology_nx_graph)
        # topology_adj = torch.FloatTensor(topology_adj)
        print("topology_adj.shape=", topology_adj.shape)
        adj_label = sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(topology_nx_graph))
        print("adj_label.shape=", adj_label.shape)

        geo_graph_path = root_path + r"geo(knn10)_graph.json"
        geo_nx_graph = json_graph.node_link_graph(json.load(open(geo_graph_path)))
        print("地理graph:", nx.info(geo_nx_graph))
        # TODO adj不为sp
        geo_adj = process_adj(geo_nx_graph)
        # geo_adj = nx.to_numpy_array(geo_nx_graph)
        # geo_adj = torch.FloatTensor(geo_adj)
        print("geo_adj.shape=", geo_adj.shape)

        text_graph_path = root_path + r"text(knn10)_graph{}.json".format("(2)" if opt.args.isVersionTwo else "")
        print("text_graph_path=", text_graph_path)
        text_nx_graph = json_graph.node_link_graph(json.load(open(text_graph_path)))
        print("语义graph:", nx.info(text_nx_graph))
        # TODO adj不为sp
        text_adj = process_adj(text_nx_graph)
        # text_adj = nx.to_numpy_array(text_nx_graph)
        # text_adj = torch.FloatTensor(text_adj)
        print("text_adj.shape=", text_adj.shape)

        # 2. 加载属性矩阵
        number = -1
        if opt.args.switch == 2 or opt.args.switch == 4:
            number = 1
        elif opt.args.switch == 3 or opt.args.switch == 5:
            number = 2
        if number == -1:
            topology_name = feature_dim
        else:
            topology_name = str(feature_dim) + "_{}".format(number)
        topology_path = root_path + r"topology_{}.npy".format(topology_name)
        # topology_path = root_path + r"topology_{}(adj).npy".format(topology_name)
        print("topology_path=", topology_path)
        topology_feature = np.load(topology_path)
        topology_feature = torch.FloatTensor(topology_feature)
        print("topology_feature.shape=", topology_feature.shape)

        coord = np.loadtxt(root_path + r"coord.txt", dtype=float)
        geo_feature = np.zeros((coord.shape[0], feature_dim))
        geo_feature[:, [0, 1]] = coord
        geo_feature = torch.FloatTensor(geo_feature)
        print("geo_feature.shape=", geo_feature.shape)

        text_feature_path = root_path + r"semantic_{}{}.npy".format(feature_dim, "(2)" if opt.args.isVersionTwo else "")
        text_feature = np.load(text_feature_path)
        print("text_feature_path=", text_feature_path)
        text_feature = torch.FloatTensor(text_feature)
        print("text_feature.shape=", text_feature.shape)

        pca_feature = None
    else: # opt.args.model == 3
        topology_graph_path = root_path + r"graph.json"
        topology_nx_graph = json_graph.node_link_graph(json.load(open(topology_graph_path)))
        print("原始graph:", nx.info(topology_nx_graph))
        # TODO adj不为sp
        topology_adj = process_adj(topology_nx_graph)
        # topology_adj = nx.to_numpy_array(topology_nx_graph)
        # topology_adj = torch.FloatTensor(topology_adj)
        print("topology_adj.shape=", topology_adj.shape)
        adj_label = sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(topology_nx_graph))
        print("adj_label.shape=", adj_label.shape)

        geo_adj = None
        text_adj = None

        # 拼接三个属性矩阵，然后进行降维，PCA降维到feature_dim(e.g 128)
        number = -1
        if opt.args.switch == 2 or opt.args.switch == 4:
            number = 1
        elif opt.args.switch == 3 or opt.args.switch == 5:
            number = 2
        if number == -1:
            topology_name = feature_dim
        else:
            topology_name = str(feature_dim) + "_{}".format(number)
        pca_feature_path = root_path + r"pca_feature_{}{}.npy".format(topology_name, "(2)" if opt.args.isVersionTwo else "")
        # pca_feature_path = root_path + r"pca_feature_{}{}(adj).npy".format(topology_name,
        #                                                               "(2)" if opt.args.isVersionTwo else "")
        print("pca_feature_path=", pca_feature_path)
        if not os.path.exists(pca_feature_path):
            topology_path = root_path + r"topology_{}.npy".format(topology_name)
            # topology_path = root_path + r"topology_{}(adj).npy".format(topology_name)
            print("topolgy_path=", topology_path)
            topology_feature = np.load(topology_path)
            coord = np.loadtxt(root_path + r"coord.txt", dtype=float)
            geo_feature = np.zeros((coord.shape[0], feature_dim))
            geo_feature[:, [0, 1]] = coord
            text_feature_path = root_path + r"semantic_{}{}.npy".format(feature_dim, "(2)" if opt.args.isVersionTwo else "")
            print("text_feature_path=", text_feature_path)
            text_feature = np.load(text_feature_path)
            concatenate_feature = np.concatenate((topology_feature, geo_feature, text_feature), axis=1)
            print("concatenate_feature.shape=", concatenate_feature.shape)
            # 进行 PCA 降维操作
            pca = PCA(n_components=feature_dim)  # 设置降维后的维度
            pca_feature = pca.fit_transform(concatenate_feature)
            np.save(pca_feature_path, pca_feature)
        else:
            pca_feature = np.load(pca_feature_path)
        # 重置
        topology_feature = None
        geo_feature = None
        text_feature = None

        # TODO 假设pca_feature为topology_feature
        # pca_feature = np.load(root_path + r"topology_128.npy")
        pca_feature = torch.FloatTensor(pca_feature)
        print("pca_feature.shape=", pca_feature.shape)

    # 3. 加载标签文件
    label = np.loadtxt(root_path + r"label.txt", dtype=int)
    print("label.shape=", label.shape)

    # 4. 加载id_map.json
    # id_map = json.load(open(root_path + "id_map.json"))
    # id_map = {int(k): int(v) for k, v in id_map.items()}
    # print("len(id_map)=", len(id_map))
    id_map = None

    # 5. 加载正负样本文件
    # context_pairs_list = []
    # context_pairs_path = root_path + \
    #           "(filter)pos2_neg20_randomIn_pos_path3_spatial0.6_text0.6{}.txt".format("(2)" if opt.args.isVersionTwo else "")
    # print("context_pairs_path=", context_pairs_path)
    # with open(context_pairs_path) as fp:
    #     for line in fp:
    #         # 格式：node pos1 spatial1,text1,path1 neg1#neg2#... neg1#neg2#... neg_spatial1,neg_text1,neg_path1#neg_spatial2,neg_text2,neg_path2#...
    #         context_pairs_list.append(list(line.split(" ")))
    #
    # print("len(context_pairs_list)=", len(context_pairs_list))
    # print(context_pairs_list[0])
    # TODO 非正负样本交叉熵损失函数
    context_pairs_list = None

    return topology_adj, adj_label, geo_adj, text_adj, topology_feature, geo_feature, text_feature, pca_feature, label \
        ,id_map, context_pairs_list

def process_adj(nx_G):
    adj = nx.to_scipy_sparse_matrix(nx_G)
    adj_ = sp.coo_matrix(adj)
    adj_ = adj_ + sp.eye(adj_.shape[0])
    adj_ = normalize(adj_)
    # TODO 注释稀疏张量的代码
    adj_ = sparse_mx_to_torch_sparse_tensor(adj_)
    return adj_

def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def tensor_to_sparse_tensor(dense_tensor):
    indices = torch.nonzero(dense_tensor)
    values = dense_tensor[indices[:, 0], indices[:, 1]]
    sparse_tensor = torch.sparse.FloatTensor(indices.t(), values, dense_tensor.size())
    return sparse_tensor


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        print("真实类：{}个,预测类仅有:{}个！".format(numclass1, numclass2))
        return -1, -1

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def eva(y_true, y_pred, epoch=0):

    acc, f1 = cluster_acc(y_true, y_pred)
    if acc == -1:
        return -1, -1, -1, -1
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

# if __name__ == '__main__':
#     load_data()