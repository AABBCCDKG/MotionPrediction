import torch
import torch.nn as nn

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义生成器的网络结构
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 128, kernel_size=4, stride=1, padding=0),  # 修改：转置卷积层，输入噪声通道数为100
            nn.BatchNorm2d(128),  # 批归一化层
            nn.ReLU(True),  # ReLU激活函数
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 修改：转置卷积层
            nn.BatchNorm2d(64),  # 批归一化层
            nn.ReLU(True),  # ReLU激活函数
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 修改：最后一层转置卷积层，输出通道数为3
            nn.Tanh()  # Tanh激活函数，将输出值限制在-1到1之间
        )

    def forward(self, x):
        return self.model(x)  # 前向传播，输入噪声x，输出生成图像

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的网络结构
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 修改：卷积层，输入通道数为3
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 第二个卷积层
            nn.BatchNorm2d(128),  # 批归一化层
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Flatten(),  # 展平层，将多维张量展平为一维张量
            nn.Linear(128 * 31 * 31, 1024),  # 修改：全连接层，输出大小为1024
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(1024, 1),  # 修改：全连接层，输出大小为1(判别真假的概率)
            nn.Sigmoid()  # Sigmoid激活函数，将输出值限制在0到1之间
        )

    def forward(self, x):
        return self.model(x)  # 前向传播，输入图像x，输出判别结果
