import opt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from GCN import *

# 1. 分开学 + 线性加
class MultiAttributedModel_Iso_Combine(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, n_clusters, v,
                 n_nodes, device):
        super(MultiAttributedModel_Iso_Combine, self).__init__()

        self.topology_encoder = GCN_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.geo_encoder = GCN_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.text_encoder = GCN_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # self.enc_cluster = GNNLayer(gae_n_enc_3, n_clusters)

        self.v = v
        self.n_clusters = n_clusters
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, gae_n_enc_3))
        torch.nn.init.xavier_uniform_(self.cluster_layer.data)

        self.topology_weight = nn.Parameter(torch.empty(1, gae_n_enc_3))
        torch.nn.init.xavier_uniform_(self.topology_weight)
        self.geo_weight = nn.Parameter(torch.empty(1, gae_n_enc_3))
        torch.nn.init.xavier_uniform_(self.geo_weight)
        self.text_weight = nn.Parameter(torch.empty(1, gae_n_enc_3))
        torch.nn.init.xavier_uniform_(self.text_weight)

    def forward(self, topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj):

        topology_z, topology_z_adj = self.topology_encoder(topology_feature, topology_adj)
        geo_z, geo_z_adj = self.geo_encoder(geo_feature, geo_adj)
        text_z, text_z_adj = self.text_encoder(text_feature, text_adj)

        # 加权和
        z = self.topology_weight * topology_z + self.geo_weight * geo_z + self.text_weight * text_z

        # gae_n_enc_3 = 20
        # z_cluster = self.enc_cluster(z, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q


# 2. 分开学 + 逐层加
class MultiAttributedModel_Iso_Merge(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, n_clusters, v,
                 n_nodes, device):
        super(MultiAttributedModel_Iso_Merge, self).__init__()

        self.topo_gcn_n_enc_1 = GNNLayer(n_input, gae_n_enc_1)
        self.topo_gcn_n_enc_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.topo_gcn_n_enc_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)

        self.geo_gcn_n_enc_1 = GNNLayer(n_input, gae_n_enc_1)
        self.geo_gcn_n_enc_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.geo_gcn_n_enc_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)

        self.text_gcn_n_enc_1 = GNNLayer(n_input, gae_n_enc_1)
        self.text_gcn_n_enc_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.text_gcn_n_enc_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)

        # self.enc_cluster = GNNLayer(gae_n_enc_3, n_clusters)

        self.v = v
        self.n_clusters = n_clusters
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, gae_n_enc_3))

        torch.nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj):

        topo_z1 = self.topo_gcn_n_enc_1(topology_feature, topology_adj)
        geo_z1 = self.geo_gcn_n_enc_1(geo_feature, geo_adj)
        text_z1 = self.text_gcn_n_enc_1(text_feature, text_adj)
        # 逐层加
        z1 = (topo_z1 + geo_z1 + text_z1) / 3

        # TODO 怎么融合？
        topo_z2 = self.topo_gcn_n_enc_2(z1, topology_adj)
        geo_z2 = self.geo_gcn_n_enc_2(z1, geo_adj)
        text_z2 = self.text_gcn_n_enc_2(z1, text_adj)
        z2 = (topo_z2 + geo_z2 + text_z2) / 3

        topo_z3 = self.topo_gcn_n_enc_3(z2, topology_adj)
        geo_z3 = self.geo_gcn_n_enc_3(z2, geo_adj)
        text_z3 = self.text_gcn_n_enc_3(z2, text_adj)
        z = (topo_z3 + geo_z3 + text_z3) / 3

        # gae_n_enc_3 = 20
        # z_cluster = self.enc_cluster(z, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q

# 3. 拼接学 + 普通GCN
class MultiAttributedModel_Concatenate(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, n_clusters, v,
                 n_nodes, device):
        super(MultiAttributedModel_Concatenate, self).__init__()

        self.encoder = GCN_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # self.enc_cluster = GNNLayer(gae_n_enc_3, n_clusters)

        self.v = v
        self.n_clusters = n_clusters
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, gae_n_enc_3))

        torch.nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, pca_feature, topology_adj):

        z, z_adj =  self.encoder(pca_feature, topology_adj);

        # gae_n_enc_3 = 20
        # z_cluster = self.enc_cluster(z, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q


