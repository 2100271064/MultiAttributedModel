import os
import argparse

parser = argparse.ArgumentParser(description='MultiAttributedModel', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
# TODO
parser.add_argument('--dataset', type=str, default="us") # TODO
parser.add_argument('--model', type=int, default=3, help='1：分开学 + 线性加,'
                                                         '2：分开学 + 逐层加'
                                                         '3：拼接学 + 普通GCN(topo_adj)') # TODO
parser.add_argument('--order', type=str, default='10e4')
parser.add_argument('--visdom_port', type=int, default=8097) # 默认是8097
parser.add_argument('--isVersionTwo', type=bool, default=False) # 用version2？（注意us的输入维度变为64）【不考虑了】

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu', type=str, default="0,1,2,3")

parser.add_argument('--seed', type=int, default=3)

parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0001) # 最开始是0.0001，学的太慢
parser.add_argument('--upd', type=int, default=1, help='Update epoch.')
parser.add_argument('--batch_size', type=int, default=512) # TODO 原来是512

# GCN structure parameter
# 原来是：128→256→20 ①
# ------------
# 64(1)：topology_hidden256_output64(new).npy
# 64(2)：topology_hidden128_output64(new).npy
# ------------
# 64(1)→256→20 ②
# 64(2)→256→20 ③
# 64(1)→128→20 ④
# 64(2)→128→20 ⑤
parser.add_argument('--switch', type=int, default=1) # TODO 选择topology的组合（主要是feature维度和网络输入层和隐藏层的不同）对应上面的注释
parser.add_argument('--gae_n_enc_1', type=int, default=128) # 128/64
parser.add_argument('--gae_n_enc_2', type=int, default=256) # 256/128
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

parser.add_argument('--n_clusters', type=int, default=5) # 原本是20，现在抽取小的，变成了5

# 单机多卡GPU跑
# parser.add_argument('--save_every', type=int, default=50, help='How often to save a snapshot')
# 模型加载上次的
parser.add_argument('--load_flag', type=bool, default=False)
parser.add_argument('--load_filename', type=str, default='model3_best_model.pth') # 仅当load_flag=True时生效

args = parser.parse_args()