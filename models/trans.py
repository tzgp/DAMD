import torch
import torch.nn as nn
import math
# import torch.nn.MultiheadAttention
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, num_patches, dropout=0.1):
        super(MultiHeadAttentionModel, self).__init__()
        # 创建6层MultiheadAttention
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_size, heads, dropout=dropout) for _ in range(6)])
        self.embed_size = embed_size
        self.num_patches = num_patches  # N: Patch的数量
    def forward(self, feature):
        # feature的形状应该是 [seq_len, batch_size, embed_size]
        # seq_len是patch数目，这里假设patch的数量等于seq_len
        # 初始化用于存储注意力得分
        scores = []
        for i in range(6):
            # 使用当前的feature和self进行自注意力计算
            feature, score = self.attention_layers[i](feature, feature, feature)
            scores.append(score)  # 存储每层的score
        # 取最后一次的score
        final_score = scores[-1]  # 最后一层的注意力得分 [batch_size, seq_len, seq_len]
        return feature, final_score
    def loss(self, final_score):
        # 假设score的形状是 [batch_size, seq_len, seq_len]，我们要计算每个位置的平均得分
        # 对于给定的patch数量N，我们希望最后的注意力得分接近1/N
        N = self.num_patches
        target = torch.ones_like(final_score) / N  # 创建一个目标矩阵，所有元素为1/N
        loss = nn.MSELoss()(final_score, target)  # 使用MSE损失计算final_score和目标之间的差异
        return loss
    #所有的patch都是一样的


# 示例：模型初始化和前向传播
embed_size = 512  # 嵌入维度
heads = 1  # MultiHeadAttention的头数
num_patches = 16  # 假设patch数目是16（即seq_len=16）
dropout = 0.1
# 模型初始化
model = MultiHeadAttentionModel(embed_size, heads, num_patches, dropout)
# 假设输入特征
batch_size = 32
seq_len = num_patches  # seq_len=patch数量
feature = torch.rand(seq_len, batch_size, embed_size)  # 随机生成输入特征 [seq_len, batch_size, embed_size]
# 前向传播
output_feature, final_score = model(feature)
# 计算损失
loss = model.loss(final_score)
print(f"Loss: {loss.item():.4f}")
# 反向传播和优化（示例）
optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()







# 示例：训练过程
# 1. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于分类任务
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 2. 模拟数据（你应该使用真实的数据）
# 假设有32个样本，每个样本长度为100，目标是10个类别中的一个
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     # 模拟输入数据和标签
#     inputs = torch.randint(0, input_dim, (32, 100))  # 随机生成输入
#     targets = torch.randint(0, num_classes, (32,))  # 随机生成目标标签
#     # 前向传播
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")