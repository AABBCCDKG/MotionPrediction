import torch
import torch.nn as nn


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义生成器的网络结构
        self.model = nn.Sequential(
            # 输入是一个形状为(3, 128, 128)的图像，3表示RGB通道，128表示图像的高和宽 （数据预处理时有处理大小么？）
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),  # 卷积层，输出通道为64
            # 归一化层是否需要？
            nn.LeakyReLU(0.2, inplace = True),  # LeakyReLU激活函数
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), # 第二个卷积层，输出通道为128
            nn.LeakyReLU(0.2, inplace = True),  # LeakyReLU激活函数
            nn.Conv2d(128, 1, kernel_size = 3, stride = 1, padding = 1),  # 第三个卷积层，输出通道为1（灰度图）
            nn.Tanh()  # Tanh激活函数，将输出值限制在-1到1之间
        )

    def forward(self, x):
        return self.model(x) # 前向传播，输入图像x，输出生成图像
    
# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的网络结构
        self.model = nn.Sequential(
            # 输入是一个形状为(1, 128, 128)的图像, 1表示灰度图，128表示图像的高和宽
            nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = 1),  # 卷积层，输出通道为64，步长为2
            nn.LeakyReLU(0.2, inplace = True), # LeakyReLU激活函数
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1) , # 第二个卷积层，输出通道为128，步长为2
            nn.LeakyReLU(0.2, inplace = True), # LeakyReLU激活函数
            nn.Flatten(),  # 展平层，将多维张量展平为一维张量
            nn.Linear(128 * 32 * 32, 1),  # 全连接层，输出大小为1(判别真假的概率)
            nn.Sigmoid()  # Sigmoid激活函数，将输出值限制在0到1之间
        )

    def forward(self, x):
        return self.model(x) # 前向传播，输入图像x，输出判别结果

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # 定义 LSTM 网络结构
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM 层，输入大小为 input_size，隐藏层大小为 hidden_size，层数为 num_layers，batch_first=True 表示输入输出的第一个维度为 batch size
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层，输入大小为 hidden_size，输出大小为 output_size

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化隐藏状态，大小为 (num_layers, batch_size, hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化细胞状态，大小为 (num_layers, batch_size, hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))  # 前向传播，通过 LSTM 层，输出为 (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # 通过全连接层，取 LSTM 输出的最后一个时间步的输出作为全连接层的输入，输出为 (batch_size, output_size)
        return out  # 返回输出