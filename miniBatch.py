
import numpy as np
np.random.seed(123)

class EdgeMinibatchIterator(object):

    def __init__(self, context_pairs_list=None, id_map=None, batch_size=512,
                 **kwargs):

        self.batch_size = batch_size # 512
        self.batch_num = 0
        self.train_edges = np.array(context_pairs_list)
        self.sample_num = len(self.train_edges)
        # TODO（不分批）
        if self.batch_size == -1:
            self.batch_size = self.sample_num
        # self.id_map = id_map
        print("EdgeMinibatchIterator-edges:", len(self.train_edges))
        # print("EdgeMinibatchIterator-len(id_map):", len(self.id_map))

    # 对象：训练的边，判断是否每一条边都已进入batch
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    # 格式：node pos1 spatial1,text1,path1 neg1#neg2#... neg1#neg2#... neg_spatial1,neg_text1,neg_path1#neg_spatial2,neg_text2,neg_path2#...
    def batch_data(self, batch_edges):
        batch1 = []
        batch2 = []
        batch3 = []
        batch4 = []
        batch5 = []
        for node, pos_other, pos_info, neg_other_list, neg_info in batch_edges:

            # node = self.id_map[int(node)]
            node = int(node)
            batch1.append(node)

            # pos_other = self.id_map[int(pos_other)]
            pos_other = int(pos_other)
            batch2.append(pos_other)

            pos_info_list = pos_info.split(",")
            sim_spatial = float(pos_info_list[0])
            sim_text = float(pos_info_list[1])
            path = int(pos_info_list[2])
            # A
            # total_sim = sim_spatial * sim_text / path
            # B
            total_sim = sim_spatial * sim_text
            batch3.append(total_sim)

            # nnegs = [self.id_map[int(nn)] for nn in neg_other_list.split("#")]
            nnegs = [int(float(nn)) for nn in neg_other_list.split("#")]
            batch4.append(np.array(nnegs))

            nnegs_sim = []
            for cur_neg_info in neg_info.split("#"):
                nn_spatial_sim, nn_text_sim, nn_path = cur_neg_info.split(",")
                # A
                # nnegs_sim.append(float(nn_spatial_sim) * float(nn_text_sim) / int(nn_path))
                # B
                nnegs_sim.append(float(nn_spatial_sim) * float(nn_text_sim))
            batch5.append(np.array(nnegs_sim))


        feed_dict = dict()
        feed_dict['batch_size'] = len(batch_edges)
        feed_dict['batch1'] = batch1
        feed_dict['batch2'] = batch2
        feed_dict['batch3'] = batch3
        feed_dict['batch4'] = batch4
        feed_dict['batch5'] = batch5

        return feed_dict

    # 小批量加载tran_edges
    def next_minibatch(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx: end_idx]
        return self.batch_data(batch_edges)

    # 打乱训练的点边，并将batch_num重置为0
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        # 打乱训练边的顺序
        self.train_edges = np.random.permutation(self.train_edges)
        # 将batch_num重置为0
        self.batch_num = 0

    def __len__(self):
        batch_total = (int)(self.sample_num / self.batch_size)
        if self.sample_num % self.batch_size > 0:
            batch_total = batch_total + 1
        return batch_total