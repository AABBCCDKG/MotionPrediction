import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import LSTMModel   # 从 models.py 中导入LSTM模型
from data_preprocessing import preprocess_data_in_batches, extract_tarfile  # 从 data_preprocessing.py 中导入函数

if __name__ == "__main__":
    # 设置文件夹路径
    tar_path = '/content/drive/My Drive/PennAction.tar.gz'
    extract_path = '/content/'
    frames_folder = os.path.join(extract_path, 'Penn_Action/frames')
    labels_folder = os.path.join(extract_path, 'Penn_Action/labels')

    # 解压缩数据集
    extract_tarfile(tar_path, extract_path)

    # 初始化用于存储所有帧和标签的列表
    all_frames = []
    all_labels = []
    batch_size = 50

    # 使用数据预处理函数来批量处理数据
    for batch_data in preprocess_data_in_batches(frames_folder, labels_folder, batch_size):
        for frames, labels in batch_data:
            all_frames.append(frames)
            all_labels.append(labels)

    # 将所有帧和标签数据合并为单个NumPy数组
    all_frames = np.concatenate(all_frames)
    all_labels = np.concatenate(all_labels)

    # 将帧和标签数据转换为PyTorch张量
    preprocessed_frames = torch.tensor(all_frames, dtype=torch.float32)
    preprocessed_labels = torch.tensor(all_labels, dtype=torch.float32)

    # 创建TensorDataset对象，将帧和标签数据打包在一起
    dataset = TensorDataset(preprocessed_frames, preprocessed_labels)

    # 创建DataLoader对象，设置batch_size并将数据随机打乱
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化LSTM模型
    input_size = 128 * 128 * 3  # 假设输入为128x128x3的图片，将其展平成一维向量
    hidden_size = 256  # 隐藏层大小
    num_layers = 2  # LSTM层数
    output_size = 128 * 128 * 3  # 输出大小，与输入大小相同

    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)  # 定义优化器为Adam优化器

    # 训练LSTM模型
    def train_lstm(model, data_loader, criterion, optimizer, epochs=100):
        for epoch in range(epochs):
            for frames, labels in data_loader:
                frames = frames.view(frames.size(0), -1)  # 将输入的图像序列展平
                labels = labels.view(labels.size(0), -1)  # 将标签展平

                # 前向传播
                output = model(frames)
                loss = criterion(output, labels)  # 计算损失

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

            # 打印每个epoch的损失
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    train_lstm(lstm_model, data_loader, criterion, optimizer)

    # 保存训练好的模型
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')
    print("Model saved as 'lstm_model.pth'")