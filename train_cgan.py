import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator  # 从 models.py 中导入生成器和判别器模型
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data_in_batches, extract_tarfile  # 从 data_preprocessing.py 中导入函数

# 定义训练cGAN模型的函数
def train_cgan(generator, discriminator, data_loader, epochs=100, batch_size=32, lr=0.0002):
    # 定义损失函数为二元交叉熵损失（binary cross-entropy loss)
    criterion = nn.BCELoss()
    # 定义生成器和判别器的优化器为Adam优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    # 开始训练循环
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)  # 获取当前批次的大小
            real_labels = torch.ones(batch_size, 1)  # 定义真实图片的标签为1
            fake_labels = torch.zeros(batch_size, 1)  # 定义假图片的标签为0

            # 训练判别器
            optimizer_d.zero_grad()  # 梯度清零
            outputs = discriminator(real_images)  # 判别器判别真实图片
            d_loss_real = criterion(outputs, real_labels)  # 计算真实图像的损失
            d_loss_real.backward()  # 反向传播

            noise = torch.randn(batch_size, 3, 128, 128)  # 生成随机噪声
            fake_images = generator(noise)  # 生成假图片
            outputs = discriminator(fake_images.detach())  # 判别器判别假图片
            d_loss_fake = criterion(outputs, fake_labels)  # 计算假图片的损失
            d_loss_fake.backward()  # 反向传播
            optimizer_d.step()  # 更新判别器参数

            d_loss = d_loss_real + d_loss_fake  # 计算判别器总损失

            # 训练生成器
            optimizer_g.zero_grad()  # 梯度清零
            outputs = discriminator(fake_images)  # 判别器判别假图片
            g_loss = criterion(outputs, real_labels)  # 计算生成器损失
            g_loss.backward()
            optimizer_g.step()

        # 打印每个epoch的损失
        print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # 保存生成器和判别器模型
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Models saved as 'generator.pth' and 'discriminator.pth'")

if __name__ == "__main__":
    # 设置文件夹路径
    tar_path = '/content/drive/My Drive/Penn_Action.tar.gz'
    extract_path = '/content/'
    frames_folder = os.path.join(extract_path, 'Penn_Action/frames')
    labels_folder = os.path.join(extract_path, 'Penn_Action/labels')

    # 解压缩数据集
    extract_tarfile(tar_path, extract_path)

    # 初始化用于储存所有帧和标签的列表
    all_frames = []
    all_labels = []
    batch_size = 50

    # 使用数据预处理函数来批量处理数据
    for batch_data in preprocess_data_in_batches(frames_folder, labels_folder, batch_size):
        for frames, labels in batch_data:
            # 将每个批次的帧和标签添加到总列表中
            all_frames.append(frames)
            all_labels.append(labels)

    # 将所有帧和标签合并为单个NumPy数组
    all_frames = np.concatenate(all_frames, axis = 0)
    all_labels = np.concatenate(all_labels, axis = 0)

    # 将NumPy数组转换成PyTorch张量
    preprocessed_frames = torch.tensor(all_frames, dtype = torch.float32)
    preprocessed_labels = torch.tensor(all_labels, dtype = torch.float32)

    # 创建TensorDataset对象，将帧和标签打包在一起
    dataset = TensorDataset(preprocessed_frames, preprocessed_labels)

    # 创建DataLoader，设置batch_size并将数据随机打乱
    data_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

    # 初始化生成器和判别器模型
    generator = Generator()
    discriminator = Discriminator()

    # 训练cGAN模型
    train_cgan(generator, discriminator, data_loader, epochs = 100, batch_size = 32, lr = 0.0002)