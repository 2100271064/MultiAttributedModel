import torch
import torch.nn as nn
import opt

class MyLoss(nn.Module):
    def __init__(self, neg_sample_weights=1.0):
        super(MyLoss, self).__init__()
        self.neg_sample_weights = neg_sample_weights

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]
        # shape=(n, 256), 元素对应相乘, reduce_sum-shape=(n,)
        result = torch.sum(inputs1 * inputs2, dim=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """
        Returns:Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        # shape = (n, 1, output_dim)
        inputs1 = inputs1.unsqueeze(1)
        # neg_samples.shape = (n, 20, output_dim)
        # neg_aff.shape = (n, 1, 20)
        neg_aff = torch.matmul(inputs1, neg_samples.transpose(1, 2))
        # 移除为1的维度 neg_aff.shape = (n, 20)
        neg_aff = neg_aff.squeeze(1)
        return neg_aff

    # 我们自己的loss
    def forward(self, pos_inputs1, pos_inputs2, neg_inputs, pos_factor, neg_factor):
        # shape=(n,)
        aff = self.affinity(pos_inputs1, pos_inputs2)
        # neg_samples.shape = (X, 20, output_dim*2)
        # neg_aff.shape = (n, 20)
        neg_aff = self.neg_cost(pos_inputs1, neg_inputs)
        true_xent = nn.functional.binary_cross_entropy_with_logits(
                input=aff, target=torch.ones_like(aff), reduction='none')
        negative_xent = nn.functional.binary_cross_entropy_with_logits(
                input=neg_aff, target=torch.zeros_like(neg_aff), reduction='none')
        # input3是正样本损失因子, input5是负样本损失因子 对应论文中的 s(u,v)
        # TODO？负样本的neg_factor？ 负样本累加
        # loss = torch.sum(torch.Tensor(pos_factor) * true_xent) + self.neg_sample_weights * torch.sum(torch.Tensor(neg_factor) * negative_xent)
        # 方案1：加入 pos_factor、neg_factor（并且负样本累加）
        pos_factor = torch.Tensor(pos_factor).to(opt.args.device)
        neg_factor = torch.Tensor(neg_factor).to(opt.args.device)
        loss = torch.sum(pos_factor * true_xent) + self.neg_sample_weights * torch.sum(neg_factor * negative_xent)
        # 方案2：尝试不加 pos_factor、neg_factor
        # loss = torch.sum(true_xent) + self.neg_sample_weights * torch.sum(negative_xent)
        # 如果求了平均之后，每个loss会特别小
        # loss = torch.sum(pos_factor * true_xent) + self.neg_sample_weights * torch.sum(torch.mean(neg_factor * negative_xent, 1))
        return loss