# import multiprocessing
# # 设置启动方法为'spawn'
# multiprocessing.set_start_method('spawn')

import opt
# 一定要在torch前面
import os

os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from model import *
from train import *
from util import *

if __name__ == '__main__':
    # 将超参数设置好
    setup()

    # 加载数据
    topology_adj, adj_label, geo_adj, text_adj, \
    topology_feature, geo_feature, text_feature, \
    pca_feature, label, \
    id_map, context_pairs_list = load_data()

    if topology_adj is not None:
        topology_adj = topology_adj.to(opt.args.device)
    adj_label = adj_label.to(opt.args.device)
    if geo_adj is not None:
        geo_adj = geo_adj.to(opt.args.device)
    if text_adj is not None:
        text_adj = text_adj.to(opt.args.device)
    if topology_feature is not None:
        topology_feature = topology_feature.to(opt.args.device)
    if geo_feature is not None:
        geo_feature = geo_feature.to(opt.args.device)
    if text_feature is not None:
        text_feature = text_feature.to(opt.args.device)
    if pca_feature is not None:
        pca_feature = pca_feature.to(opt.args.device)

    opt.args.n_nodes = topology_adj.shape[0]
    opt.args.input_dim = opt.args.gae_n_enc_1 # 后面使用(2)需要修改

    # =================多进程调用（一次数据集加载，模型共享）====================
    # # 创建进程池
    # process_num = 3
    # pool = multiprocessing.Pool(processes=process_num)
    # print("multiprocessing-num{}, opt.args.model待校准！".format(process_num))
    # ==================================================================

    run_round = 1
    # final_acc = []
    # final_nmi = []
    # final_ari = []
    # final_f1 = []
    # valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        if opt.args.model == 1:
            # 1. 分开学 + 线性加
            model = MultiAttributedModel_Iso_Combine(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                                     gae_n_enc_3=opt.args.gae_n_enc_3,
                                                     gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                                     gae_n_dec_3=opt.args.gae_n_dec_3,
                                                     n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                                     n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)
        elif opt.args.model == 2:
            # 2. 分开学 + 逐层加
            model = MultiAttributedModel_Iso_Merge(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                                   gae_n_enc_3=opt.args.gae_n_enc_3,
                                                   gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                                   gae_n_dec_3=opt.args.gae_n_dec_3,
                                                   n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                                   n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)

        else:  # opt.args.model == 3
            # 3. 拼接学 + 普通GCN(topo_adj)
            model = MultiAttributedModel_Concatenate(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
                                                     gae_n_enc_3=opt.args.gae_n_enc_3,
                                                     gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
                                                     gae_n_dec_3=opt.args.gae_n_dec_3,
                                                     n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
                                                     n_nodes=opt.args.n_nodes, device=opt.args.device).to(opt.args.device)

        if opt.args.load_flag:
            # model.load_state_dict(torch.load('/home/laixy/work2/output/model/{}/switch_{}/model_{}/lr{}/{}'.format(opt.args.dataset, \
            #                                                                                                        opt.args.switch, \
            #                                                                                                        opt.args.model, \
            #                                                                                                        opt.args.lr, \
            #                                                                                                        opt.args.load_filename)))
            # opt.args.load_filename的格式"model{}_epoch{}.pth"
            model.load_state_dict(torch.load('/home/laixy/work2/output/subgraph/{}/{}/{}'.format(opt.args.dataset,
                                                                                                 opt.args.order,
                                                                                                 opt.args.load_filename)))

        # ===================多卡========================
        # 使用nn.DataParallel实现多GPU并行运算
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        # 使用DistributedDataParallel实现GPU并行运算
        # model = DDP(model, device_ids=[opt.args.local_rank], output_device=opt.args.local_rank)

        # model = torch.nn.DataParallel(model).to(opt.args.device)  # 模型侧的多卡并行
        # ===========================================

        # 普通调用
        # mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = train(model,
        #                                                                topology_adj, geo_adj, text_adj,
        #                                                                topology_feature, geo_feature, text_feature,
        #                                                                pca_feature, label,
        #                                                                id_map, context_pairs_list,
        #                                                                opt.args.device)
        acc_best, nmi_best, ari_best, f1_best = train(model, adj_label,
                                                      topology_adj, geo_adj, text_adj,
                                                      topology_feature, geo_feature, text_feature,
                                                      pca_feature, label,
                                                      id_map, context_pairs_list,
                                                      opt.args.device)
        # ===================多进程调用（一次数据集加载，模型共享）===================
        # 创建参数列表，每个参数包含模型ID和共享的数据集
        # args_list = [(model_id+1,
        #               topology_adj, geo_adj, text_adj,
        #               topology_feature, geo_feature, text_feature,
        #               pca_feature, label,
        #               id_map, context_pairs_list,
        #               opt.args.device) for model_id in range(process_num)]
        # # 使用进程池并行训练模型
        # pool.map(train, args_list)

    print("====================")
    print("finish training!")
    print("final metrics : best_acc = {:.2f}, best_nmi = {:.2f}, best_ari = {:.2f}, best_f1 = {:.2f}".format(acc_best * 100, nmi_best * 100, ari_best * 100, f1_best * 100))
    print("best_acc = {:.4f},\n best_nmi = {:.4f},\n best_ari = {:.4f},\n best_f1 = {:.4f}\n".format(acc_best, nmi_best, ari_best, f1_best))

    # acc_arr = np.array(final_acc)
    # nmi_arr = np.array(final_nmi)
    # ari_arr = np.array(final_ari)
    # f1_arr = np.array(final_f1)
    # print("{} epoch × 10, 有效的epoch数：".format(opt.args.epoch), valid_epoch_num_list)
    #
    # value = np.mean(acc_arr)
    # var = np.var(acc_arr)
    # std = np.std(acc_arr)
    # print('final_acc: {}, fianl_var_acc: {}, final_std_acc:{}'.format(value, var, std))
    # print('final_acc: {:.2f}%, fianl_var_acc: {:.4f}, final_std_acc:{:.2f}%'.format(value * 100, var, std * 100))
    #
    # value = np.mean(nmi_arr)
    # var = np.var(nmi_arr)
    # std = np.std(nmi_arr)
    # print('final_nmi: {}, final_var_nmi: {}, final_std_nmi:{}'.format(value, var, std))
    # print('final_nmi: {:.2f}%, final_var_nmi: {:.4f}, final_std_nmi:{:.2f}%'.format(value * 100, var, std * 100))
    #
    # value = np.mean(ari_arr)
    # var = np.var(ari_arr)
    # std = np.std(ari_arr)
    # print('final_ari: {}, final_var_ari: {}, final_std_ari:{}'.format(value, var, std))
    # print('final_ari: {:.2f}%, final_var_ari: {:.4f}, final_std_ari:{:.2f}%'.format(value * 100, var, std * 100))
    #
    # value = np.mean(f1_arr)
    # var = np.var(f1_arr)
    # std = np.std(f1_arr)
    # print('final_f1: {}, final_var_f1: {}, final_std_f1:{}'.format(value, var, std))
    # print('final_f1: {:.2f}%, final_var_f1: {:.4f}, final_std_f1:{:.2f}%'.format(value * 100, var, std * 100))

    # print("finish!")
