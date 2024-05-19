# 图像帧和标签配对：每个视频序列的图像帧和对应的标签数据会被读取并配对
# 格式转换：图像帧会被转换为NumPy数组，标签数据会从MAT文件中提取出来并转为合适的格式
# 数据结构：预处理后的数据会被组织成一个列表或者数组，其中每一个元素包含一对（图像帧，标签数据）数据
from google.colab import drive
import os
import tarfile        # 解压缩`.tar.gz`文件
import scipy.io       # 读取`.mat`文件
import numpy as np    # 处理数组
from PIL import Image # 读取图像文件

# 设置文件夹路径
tar_path = '/content/drive/My Drive/Penn_Action.tar.gz'
extract_path = '/content/'
frames_folder = os.path.join(extract_path, 'Penn_Action/frames')
labels_folder = os.path.join(extract_path, 'Penn_Action/labels')

# 创建目标文件夹并解压缩
def extract_tarfile(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {tar_path} to {extract_path}")

extract_tarfile(tar_path, extract_path)

# 创建目标文件夹并解压缩
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)
print(f"Extracted {tar_path} to {extract_path}")

# 定义函数读取图像帧
def load_frames(frames_folder):
    # 获取文件夹中的所有文件，并按名称排序
    frames_files = sorted(os.listdir(frames_folder))
    frames = []
    for frames_file in frames_files:
        # 构建每个图像文件的完整路径
        frame_path = os.path.join(frames_folder, frames_file)
        # 使用PIL打开图像并转换为NumPy数组
        frame = Image.open(frame_path)
        frames.append(np.array(frame))
    # 返回包含所有图像帧的NumPy数组
    return np.array(frames)

# 定义函数读取标签文件
def load_labels(label_file):
    # 使用SciPy加载MAT文件
    mat = scipy.io.loadmat(label_file)
    # 将所有标签数据转换为NumPy数组
    labels = {key: np.array(value) for key, value in mat.items() if key[0] != '_'}
    return labels

# 定义函数来批量处理数据
def preprocess_data_in_batches(frames_folder, labels_folder, batch_size = 100):
    # 获取所有文件夹并按名称排序
    all_folders = sorted(os.listdir(frames_folder))
    total_folders = len(all_folders) # 文件夹总数
    batch_data = []   # 用于存储每个批次的数据

    for i, folder in enumerate(all_folders):
        # 构建每个帧文件夹和对应的标签文件夹的路径
        frame_folder = os.path.join(frames_folder, folder)
        label_file = os.path.join(labels_folder, folder + '.mat')
    

        # 检查帧文件夹和标签文件是否存在
        if os.path.isdir(frame_folder) and os.path.exists(label_file):
            print(f"Processing {folder}...({i + 1} / {total_folders})")
            # 读取图像帧和标签数据
            frames = load_frames(frame_folder)
            labels = load_labels(label_file)
            # 将图像帧和标签数据添加到列表中
            batch_data.append((frames, labels))
            
            # 如果批次数据达到指定大小或者已经处理完所有文件夹，则返回批次数据
            if len(batch_data) == batch_size or i == total_folders - 1:
                yield batch_data
                batch_data = [] # 清空批量数据列表，为下一个批次做准备
    # 如果还有剩余的未处理数据，返回这些数据
    if batch_data:
        yield batch_data

batch_size = 50

# 批量处理数据并打印每个批次的处理结果
for batch_data in preprocess_data_in_batches(frames_folder, labels_folder, batch_size):
    print(f"Processed {len(batch_data)} sequences.")
    # 打印示例数据以验证预处理结果
    for frames, labels in batch_data:
        print(f"Number of frames in sequence: {len(frames)}")
        print(f"Shape of first frame: {frames[0].shape}")
        for key, value in labels.items():
            print(f"{key}: {value.shape}")
print("All data processed in batches.")

'''
# 定义函数预处理所有数据
def preprocess_data(frames_folder, labels_folder):
    all_data = []  # 储存所有数据的列表
    total_folders = len(os.listdir(frames_folder))
    processed_folders = 0  # 已处理文件夹计数 

    for folder in sorted(os.listdir(frames_folder)):
        # 构建每个帧文件夹和对应的标签文件夹的路径
        frame_folder = os.path.join(frames_folder, folder)
        label_file = os.path.join(labels_folder, folder + '.mat')
        
        # 检查帧文件夹和标签文件是否存在
        if os.path.isdir(frame_folder) and os.path.exists(label_file):
            processed_folders += 1
            print(f"Processing {folder}...({processed_folders} / {total_folders})")
            # 读取图像帧和标签数据
            frames = load_frames(frame_folder)
            labels = load_labels(label_file)
            # 将图像帧和标签数据添加到列表中
            all_data.append((frames, labels))
        else:
            print(f"Skipping {folder}, missing frames or labels.")
    
    print("Preprocessing completed.")
    return all_data


# 执行数据预处理
data = preprocess_data(frames_folder, labels_folder)
print(f'Processed {len(data)} sequences.')

# 打印一些示例数据
print("Example data:")
example_frames, example_labels = data[0]
print(f"Number of frames in first sequence: {len(example_frames)}")
print(f"Shape of first frame: {example_frames[0].shape}")
print(f"Labels for first sequence: {example_labels}")

# 打印详细标签信息
print("Detailed labels for the first sequence:")
for key, value in example_labels.items():
    print(f"{key}: {value.shape}")
'''
