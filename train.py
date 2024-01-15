import os

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import opt
from util import *
from miniBatch import *
from MyLoss import *

import warnings
warnings.filterwarnings('ignore')

# ===================多进程调用（一次数据集加载，模型共享）===================
from model import *
def parallel_model(model):
    if model == 1:
        # 1. 分开学 + 线性加
        model = MultiAttributedModel_Iso_Combine(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                                 gae_n_enc_3=opt.args.gae_n_enc_3,
                                                 gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                                 gae_n_dec_3=opt.args.gae_n_dec_3,
                                                 n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                                 n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)
        opt.args.model = 1
    elif model == 2:
        # 2. 分开学 + 逐层加
        model = MultiAttributedModel_Iso_Merge(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                               gae_n_enc_3=opt.args.gae_n_enc_3,
                                               gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                               gae_n_dec_3=opt.args.gae_n_dec_3,
                                               n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                               n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)
        opt.args.model = 2
    elif model == 3:  # opt.args.model == 3
        # 3. 拼接学 + 普通GCN(topo_adj)
        model = MultiAttributedModel_Concatenate(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                                 gae_n_enc_3=opt.args.gae_n_enc_3,
                                                 gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                                 gae_n_dec_3=opt.args.gae_n_dec_3,
                                                 n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                                 n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)
        opt.args.model = 3
    print("opt.args.model = {}, 已完成校准！".format(opt.args.model))
    return model
# ===================================================================================

def train(model, adj_label, topology_adj, geo_adj, text_adj,
          topology_feature, geo_feature, text_feature,
          pca_feature, label,
          id_map, context_pairs_list,
          device):

    print("Using train method!")

    # ===================多进程调用（一次数据集加载，模型共享）===================
    # model = parallel_model(model)

    # acc_result = []
    # nmi_result = []
    # ari_result = []
    # f1_result = []
    acc_best = None
    nmi_best = None
    ari_best = None
    f1_best = None

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    with torch.no_grad():
        model_cpu = model.cpu()
        # model_cpu = model.module.cpu()  # 多卡并行解除DataParallel
        model_cpu.device = "cpu"
        if opt.args.model != 3:
            topology_feature = topology_feature.cpu()
            geo_feature = geo_feature.cpu()
            text_feature = text_feature.cpu()

            topology_adj = topology_adj.cpu()
            geo_adj = geo_adj.cpu()
            text_adj = text_adj.cpu()
            z, q = model_cpu(topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj)
            # z, q = model(topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj)
        else:
            pca_feature = pca_feature.cpu()
            topology_adj = topology_adj.cpu()
            z, q = model_cpu(pca_feature, topology_adj)
            # z, q = model(pca_feature, topology_adj)
        p = target_distribution(q.data).to(opt.args.device)

    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # kmeans.cluster_centers_  从z中获得初始聚类中心
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # model.module.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)  # 多卡并行解除DataParallel
    eva(label, y_pred, 'init')

    # TODO 新增可视化
    from visualizer import Visualizer
    # smallBatch_XXX(A)、continue_XXX、BsmallBatch_XXX(B)
    # reconstruct_XXX
    # KL_XXX
    visualizer = Visualizer(visdom_port=opt.args.visdom_port,
                            env_name="KL_{}_{}_model{}".format(opt.args.dataset, opt.args.order, opt.args.model))

    # TODO ---------分批处理------------
    # miniBatch = EdgeMinibatchIterator(id_map=id_map,
    #                                   context_pairs_list=context_pairs_list,
    #                                   batch_size=opt.args.batch_size)
    # myloss = MyLoss()


    for epoch in range(opt.args.epoch):

        # ==============训练阶段======================
        model.train()
        model = model.to(opt.args.device)
        model.device = "cuda"
        loss = 0

        if opt.args.model != 3:
            topology_feature = topology_feature.to(opt.args.device)
            geo_feature = geo_feature.to(opt.args.device)
            text_feature = text_feature.to(opt.args.device)

            topology_adj = topology_adj.to(opt.args.device)
            geo_adj = geo_adj.to(opt.args.device)
            text_adj = text_adj.to(opt.args.device)
            z, q = model(topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj)
        else:
            pca_feature = pca_feature.to(opt.args.device)
            topology_adj = topology_adj.to(opt.args.device)
            z, q = model(pca_feature, topology_adj)

        # 基础loss
        # KL
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        # # A
        # loss = F.binary_cross_entropy(torch.sigmoid(torch.mm(z, z.t())).view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # TODO ----------分批处理-----------
        #
        # # 打乱训练的点边，并将batch_num重置为0
        # miniBatch.shuffle()
        # while not miniBatch.end():
        #     # 小批量加载
        #     feed_dict = miniBatch.next_minibatch()
        #     batch_size = feed_dict['batch_size']
        #     pos1 = feed_dict['batch1']
        #     pos2 = feed_dict['batch2']
        #     pos_sim = feed_dict['batch3']
        #     nnegs = feed_dict['batch4']
        #     nnegs_sim = feed_dict['batch5']
        #
        #     optimizer.zero_grad()
        #     # 模型
        #     if opt.args.model != 3:
        #         # TODO adj为sp
        #         z_pos1, q_pos1 = model(topology_feature[pos1],
        #                      geo_feature[pos1],
        #                      text_feature[pos1],
        #                      tensor_to_sparse_tensor(topology_adj.to_dense()[pos1,:][:, pos1]),
        #                      tensor_to_sparse_tensor(geo_adj.to_dense()[pos1,:][:, pos1]),
        #                      tensor_to_sparse_tensor(text_adj.to_dense()[pos1,:][:, pos1]))
        #         z_pos2, q_pos2 = model(topology_feature[pos2],
        #                                geo_feature[pos2],
        #                                text_feature[pos2],
        #                                tensor_to_sparse_tensor(topology_adj.to_dense()[pos2, :][:, pos2]),
        #                                tensor_to_sparse_tensor(geo_adj.to_dense()[pos2, :][:, pos2]),
        #                                tensor_to_sparse_tensor(text_adj.to_dense()[pos2, :][:, pos2]))
        #         # 稀疏矩阵转换成稠密矩阵，因为并行计算不能用稀疏切片
        #         # z_pos1, q_pos1 = model(topology_feature[pos1],
        #         #                        geo_feature[pos1],
        #         #                        text_feature[pos1],
        #         #                        topology_adj.to_dense()[pos1, :][:, pos1],
        #         #                        geo_adj.to_dense()[pos1, :][:, pos1],
        #         #                        text_adj.to_dense()[pos1, :][:, pos1])
        #         # z_pos2, q_pos2 = model(topology_feature[pos2],
        #         #                        geo_feature[pos2],
        #         #                        text_feature[pos2],
        #         #                        topology_adj.to_dense()[pos2, :][:, pos2],
        #         #                        geo_adj.to_dense()[pos2, :][:, pos2],
        #         #                        text_adj.to_dense()[pos2, :][:, pos2])
        #         # z_pos1, q_pos1 = model(topology_feature[pos1],
        #         #                        geo_feature[pos1],
        #         #                        text_feature[pos1],
        #         #                        topology_adj[pos1, :][:, pos1],
        #         #                        geo_adj[pos1, :][:, pos1],
        #         #                        text_adj[pos1, :][:, pos1])
        #         # z_pos2, q_pos2 = model(topology_feature[pos2],
        #         #                        geo_feature[pos2],
        #         #                        text_feature[pos2],
        #         #                        topology_adj[pos2, :][:, pos2],
        #         #                        geo_adj[pos2, :][:, pos2],
        #         #                        text_adj[pos2, :][:, pos2])
        #         def integrate(neg_sample):
        #             # TODO adj为sp
        #             z_neg, q_neg = model(topology_feature[neg_sample],
        #                                  geo_feature[neg_sample],
        #                                  text_feature[neg_sample],
        #                                  tensor_to_sparse_tensor(topology_adj.to_dense()[neg_sample, :][:, neg_sample]),
        #                                  tensor_to_sparse_tensor(geo_adj.to_dense()[neg_sample, :][:, neg_sample]),
        #                                  tensor_to_sparse_tensor(text_adj.to_dense()[neg_sample, :][:, neg_sample]))
        #             # 稀疏矩阵转换成稠密矩阵，因为并行计算不能用稀疏切片
        #             # z_neg, q_neg = model(topology_feature[neg_sample],
        #             #                      geo_feature[neg_sample],
        #             #                      text_feature[neg_sample],
        #             #                      topology_adj.to_dense()[neg_sample, :][:, neg_sample],
        #             #                      geo_adj.to_dense()[neg_sample, :][:, neg_sample],
        #             #                      text_adj.to_dense()[neg_sample, :][:, neg_sample])
        #             # z_neg, q_neg = model(topology_feature[neg_sample],
        #             #                      geo_feature[neg_sample],
        #             #                      text_feature[neg_sample],
        #             #                      topology_adj[neg_sample, :][:, neg_sample],
        #             #                      geo_adj[neg_sample, :][:, neg_sample],
        #             #                      text_adj[neg_sample, :][:, neg_sample])
        #             return [z_neg, q_neg]
        #
        #         # 负样本 nnegs.shape=(X, 20, dim), 所以是20个20个分别进去integrate
        #         # neg_outputs.shape = (X, 20, output_dim*2)
        #         neg_outputs = [integrate(x) for x in nnegs]
        #         z_neg = torch.stack([tmp[0] for tmp in neg_outputs], dim=0).float()
        #         # q_neg = torch.stack([tmp[1] for tmp in neg_outputs], dim=0).float()
        #
        #     else:
        #         # TODO adj为sp
        #         z_pos1, q_pos1 = model(pca_feature[pos1],
        #                      tensor_to_sparse_tensor(topology_adj.to_dense()[pos1,:][:, pos1]))
        #         z_pos2, q_pos2 = model(pca_feature[pos2],
        #                                tensor_to_sparse_tensor(topology_adj.to_dense()[pos2, :][:, pos2]))
        #         # 稀疏矩阵转换成稠密矩阵，因为并行计算不能用稀疏切片
        #         # z_pos1, q_pos1 = model(pca_feature[pos1],
        #         #                        topology_adj.to_dense()[pos1, :][:, pos1])
        #         # z_pos2, q_pos2 = model(pca_feature[pos2],
        #         #                        topology_adj.to_dense()[pos2, :][:, pos2])
        #         # z_pos1, q_pos1 = model(pca_feature[pos1],
        #         #                        topology_adj[pos1, :][:, pos1])
        #         # z_pos2, q_pos2 = model(pca_feature[pos2],
        #         #                        topology_adj[pos2, :][:, pos2])
        #         def integrate(neg_sample):
        #             # TODO adj为sp
        #             z_neg, q_neg = model(pca_feature[neg_sample],
        #                      tensor_to_sparse_tensor(topology_adj.to_dense()[neg_sample,:][:, neg_sample]))
        #             # 稀疏矩阵转换成稠密矩阵，因为并行计算不能用稀疏切片
        #             # z_neg, q_neg = model(pca_feature[neg_sample],
        #             #                      topology_adj.to_dense()[neg_sample, :][:, neg_sample])
        #             # z_neg, q_neg = model(pca_feature[neg_sample],
        #             #          topology_adj[neg_sample,:][:, neg_sample])
        #             return [z_neg, q_neg]
        #
        #         # 负样本 nnegs.shape=(X, 20, dim), 所以是20个20个分别进去integrate
        #         # neg_outputs.shape = (X, 20, output_dim)
        #         neg_outputs = [integrate(x) for x in nnegs]
        #         z_neg = torch.stack([tmp[0] for tmp in neg_outputs], dim=0).float()
        #         # q_neg = torch.stack([tmp[1] for tmp in neg_outputs], dim=0).float()
        #
        #
        #     # TODO:正负样本损失计算的比例问题：负样本会不会因为数量多而占比过大？（已修正）
        #     cur_loss = myloss(z_pos1, z_pos2, z_neg, pos_sim, nnegs_sim) / batch_size
        #     loss += cur_loss.item()
        #     print("epoch = {}, {}/{}, cur loss = {}, loss = {}".format(epoch, miniBatch.batch_num, len(miniBatch),
        #                                                                cur_loss.item(), loss))
        #     # TODO 1
        #     save_directory = '/home/laixy/work2/output/subgraph/{}/{}/BsmallBatch'.format(opt.args.dataset,
        #                                                                                   opt.args.order)
        #     if not os.path.exists(save_directory):
        #         os.makedirs(save_directory)
        #     torch.save(model.state_dict(),
        #                 save_directory + "/model{}_last_model.pth"
        #                 .format(opt.args.model))
        #
        #     # loss = loss + myloss(z_pos1, z_pos2, z_neg, pos_sim, nnegs_sim)
        #     # loss = loss / batch_size
        #     # print("epoch = {}, {}/{}, loss = {}".format(epoch, miniBatch.batch_num, len(miniBatch),
        #     #                                             loss.item()))
        #     # loss.backward()
        #     cur_loss.backward()
        #     optimizer.step()
        #     # optimizer.zero_grad()  # 清零梯度

        print("epoch = {}, total loss = {}".format(epoch, loss))
        # TODO 保存模型
        # TODO 2
        save_directory = '/home/laixy/work2/output/subgraph/{}/{}/KL'.format(opt.args.dataset, opt.args.order)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(model.state_dict(), save_directory + "/model{}_epoch{}.pth".format(opt.args.model, epoch))

        # ==============评估阶段======================
        if epoch % opt.args.upd == 0:
            model.eval()

            model_cpu = model.cpu()
            # model_cpu = model.module.cpu() # 多卡并行解除DataParallel
            model_cpu.device = "cpu"

            if opt.args.model != 3:
                topology_feature = topology_feature.cpu()
                geo_feature = geo_feature.cpu()
                text_feature = text_feature.cpu()

                topology_adj = topology_adj.cpu()
                geo_adj = geo_adj.cpu()
                text_adj = text_adj.cpu()
                z, q = model_cpu(topology_feature, geo_feature, text_feature, topology_adj, geo_adj, text_adj)
            else:
                pca_feature = pca_feature.cpu()
                topology_adj = topology_adj.cpu()
                z, q = model_cpu(pca_feature, topology_adj)
            p = target_distribution(q.data).to(opt.args.device)

            # TODO
            # 聚类1
            kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            res = kmeans.labels_
            # 聚类2
            # res = q.data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(label, res, str(epoch))
            if acc != -1:
                #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
                # acc_result.append(acc)
                # nmi_result.append(nmi)
                # ari_result.append(ari)
                # f1_result.append(f1)
                if acc_best is None:
                    acc_best = acc
                    nmi_best = nmi
                    ari_best = ari
                    f1_best = f1

                # TODO 新增
                if acc >= acc_best:
                    acc_best = acc
                    nmi_best = nmi
                    ari_best = ari
                    f1_best = f1
                    print("maxAcc update! ===> epoch = {}, acc = {}, nmi = {}, ari = {}, f1 = {}".format(epoch, acc_best, nmi_best, ari_best, f1_best))
                    # TODO 3
                    save_directory = '/home/laixy/work2/output/subgraph/{}/{}/KL'.format(opt.args.dataset,
                                                                                         opt.args.order)
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)
                    torch.save(model.state_dict(),
                               save_directory + "/model{}_best_model.pth"
                               .format(opt.args.model))

                # TODO 新增
                visualizer.plot_acc(epoch, acc)
                # visualizer.plot_loss(epoch, loss) # 正负样本交叉熵
                visualizer.plot_loss(epoch, loss.item()) # 其他损失函数

    # return np.mean(acc_result), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_result)
    return acc_best, nmi_best, ari_best, f1_best