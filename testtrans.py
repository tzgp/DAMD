import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
# 自定义模型（包括6层 MultiheadAttention）
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
# 假设正常数据的训练过程
def train_normal_model(model, normal_data_loader, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in normal_data_loader:
            inputs, _ = data  # 只使用输入特征
            optimizer.zero_grad()

            feature = inputs.permute(1, 0, 2)  # 转换为 [seq_len, batch_size, embed_size] 格式
            output_feature, final_score = model(feature)

            loss = model.loss(final_score)  # 计算损失
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(normal_data_loader):.4f}")
def anomaly_detection(model, data_loader, threshold=1.0):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data  # 输入和标签
            feature = inputs.permute(1, 0, 2)  # 转换为 [seq_len, batch_size, embed_size] 格式
            output_feature, final_score = model(feature)
            # 计算注意力得分的均值，作为正常得分的参考
            score_mean = final_score.mean(dim=1)  # 对于每个数据点的得分，计算平均值 [batch_size, seq_len]

            # 计算得分的方差（标准差），作为分布的离散程度
            score_variance = final_score.var(dim=1).mean(dim=1)  # 对于每个数据点计算方差并求均值

            for i in range(len(score_mean)):
                # 计算均值和方差的差异
                score_diff = torch.abs(score_mean[i] - score_variance[i])

                # 计算score_diff的均值，将其转化为单一标量
                score_diff_value = score_diff.mean().item()  # 聚合为单一标量

                # 判断score_diff_value是否超出阈值，认为是异常
                if score_diff_value > threshold:
                    anomalies.append(i)  # 记录异常数据点的索引

    return anomalies
# 假设的超参数
embed_size = 512  # 嵌入维度
heads = 8  # MultiHeadAttention的头数
num_patches = 16  # 假设patch数目是16（即seq_len=16）
dropout = 0.1
# 初始化模型
model = MultiHeadAttentionModel(embed_size, heads, num_patches, dropout)
# 假设正常数据加载器，模拟正常数据
normal_data = torch.rand(1000, num_patches, embed_size)  # 假设正常数据有1000个样本
normal_labels = torch.zeros(1000)  # 正常标签为0
normal_dataset = torch.utils.data.TensorDataset(normal_data, normal_labels)
normal_data_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=32, shuffle=True)
# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 训练模型
train_normal_model(model, normal_data_loader, optimizer, num_epochs=5)
# 假设测试数据，模拟异常数据
test_data = torch.rand(100, num_patches, embed_size)  # 100个测试样本
test_labels = torch.ones(100)  # 异常标签为1
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
# 异常检测
threshold = 0.5  # 假设阈值为0.5
anomalies = anomaly_detection(model, test_data_loader, threshold)
# 输出检测到的异常数据点
print(f"检测到 {len(anomalies)} 个异常数据点：{anomalies}")
