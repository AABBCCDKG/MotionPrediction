import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import Generator, Discriminator  # 从 models.py 中导入生成器和判别器模型
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data_in_batches, extract_tarfile  # 从 data_preprocessing.py 中导入函数
from sklearn.preprocessing import LabelEncoder

# 定义可视化函数
def visualize_data(frames, num_images=5):
    plt.figure(figsize=(10, 2 * num_images))
    for i in range(num_images):
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(frames[i].astype(np.uint8))
        plt.axis('off')
    plt.show()

# 定义训练cGAN模型的函数
def train_cgan(generator, discriminator, data_loader, epochs=100, batch_size=32, lr=0.0002):
    # 定义损失函数为二元交叉熵损失（binary cross-entropy loss)
    criterion = nn.BCELoss()
    # 定义生成器和判别器的优化器为Adam优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    # 开始训练循环
    for epoch in range(epochs):
        for i, (real_images, *labels) in enumerate(data_loader):
            batch_size = real_images.size(0)  # 获取当前批次的大小
            real_labels = torch.ones(batch_size, 1)  # 定义真实图片的标签为1
            fake_labels = torch.zeros(batch_size, 1)  # 定义假图片的标签为0

            # 训练判别器
            optimizer_d.zero_grad()  # 梯度清零
            outputs = discriminator(real_images)  # 判别器判别真实图片
            d_loss_real = criterion(outputs, real_labels)  # 计算真实图像的损失
            d_loss_real.backward()  # 反向传播

            noise = torch.randn(batch_size, 100, 1, 1)  # 生成随机噪声
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
    torch.save(generator.state_dict(), '/content/drive/My Drive/generator.pth')
    torch.save(discriminator.state_dict(), '/content/drive/My Drive/discriminator.pth')
    print("Models saved as 'generator.pth' and 'discriminator.pth' in Google Drive")

# 编码非数值类型的数据
def encode_non_numeric_data(labels):
    label_encoders = {}
    encoded_labels = {}
    
    for key in labels:
        if isinstance(labels[key][0], (np.str_, str)):
            le = LabelEncoder()
            encoded_labels[key] = le.fit_transform(labels[key].astype(str))
            label_encoders[key] = le
        else:
            encoded_labels[key] = labels[key]
    
    return encoded_labels, label_encoders

# 确保标签和帧的大小匹配
def ensure_size_match(frames, labels):
    min_length = min(frames.shape[0], min(len(v) for v in labels.values()))
    
    frames = frames[:min_length]
    for key in labels:
        labels[key] = labels[key][:min_length]
    
    return frames, labels

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
    max_sequences = 100  # 只处理前100组数据
    sequence_count = 0   # 初始化计数器

    # 使用数据预处理函数来批量处理数据
    for batch_data in preprocess_data_in_batches(frames_folder, labels_folder, batch_size, target_size=(128, 128)):
        for frames, labels in batch_data:
            if sequence_count >= max_sequences:
                break
            # 将每个批次的帧和标签添加到总列表中
            all_frames.append(frames)
            if labels and len(labels) > 0:  # 确保 labels 是非空字典
                all_labels.append(labels)
            sequence_count += 1
        if sequence_count >= max_sequences:
            break

    # 将所有帧和标签合并为单个NumPy数组
    all_frames = np.concatenate(all_frames, axis=0)
    all_labels = [label for label in all_labels if len(label) > 0]  # 确保标签字典非空
    concatenated_labels = {key: np.concatenate([d[key] for d in all_labels], axis=0) for key in all_labels[0]}

    # 编码非数值类型的数据
    encoded_labels, label_encoders = encode_non_numeric_data(concatenated_labels)

    # 确保标签和帧的大小匹配
    all_frames, encoded_labels = ensure_size_match(all_frames, encoded_labels)

    # 可视化数据
    visualize_data(all_frames)

    # 将NumPy数组转换成PyTorch张量并确保通道顺序正确
    preprocessed_frames = torch.tensor(all_frames.transpose((0, 3, 1, 2)), dtype=torch.float32)
    
    # 将数值类型的标签字段转换为浮点数张量，将其他类型转换为整型张量
    preprocessed_labels_dict = {}
    for key in encoded_labels:
        if isinstance(encoded_labels[key][0], (np.float32, np.float64)):
            preprocessed_labels_dict[key] = torch.tensor(encoded_labels[key], dtype=torch.float32)
        else:
            preprocessed_labels_dict[key] = torch.tensor(encoded_labels[key].astype(np.int64), dtype=torch.long)

    # 创建包含所有标签的TensorDataset对象
    dataset = TensorDataset(preprocessed_frames, *preprocessed_labels_dict.values())

    # 创建DataLoader，设置batch_size并将数据随机打乱
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化生成器和判别器模型
    generator = Generator()
    discriminator = Discriminator()

    # 训练cGAN模型
    train_cgan(generator, discriminator, data_loader, epochs=100, batch_size=32, lr=0.0002)
