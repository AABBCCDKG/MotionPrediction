# 通过`scipy.io`库读取`.mat`文件，使用`os`库来遍历图像文件夹
import os
import scipy.io       # 读取`.mat`文件
import numpy as np    # 处理数组
from PIL import Image # 读取图像文件

frames_folder = '/Users/dong/Desktop/Penn_Action/frames/'
labels_folder = '/Users/dong/Desktop/Penn_Action/labels/'

# 定义函数读取图像帧
def load_frames(frames_folder):
    frames_files = sorted(os.listdir(frames_folder))
    frames = []
    for frames_file in frames_files:
        frame_path = os.path.join(frames_folder, frames_file)
        frame = Image.open(frame_path)
        frames.append(np.array(frame))
    return np.array(frames)

# 定义函数读取标签文件
def load_labels(label_file):
    mat = scipy.io.loadmat(label_file)
    return mat

# 处理所有数据
def preprocess_data(frames_folder, labels_folder):
    all_data = []
    total_folders = len(os.listdir(frames_folder))
    processed_folders = 0

    for folder in sorted(os.listdir(frames_folder)):
        frame_folder = os.path.join(frames_folder, folder)
        label_file = os.path.join(labels_folder, folder + '.mat')
        
        if os.path.isdir(frame_folder) and os.path.exists(label_file):
            processed_folders += 1
            print(f"Processing {folder}...({processed_folders} / {total_folders})")
            frames = load_frames(frame_folder)
            labels = load_labels(label_file)
            all_data.append((frames, labels))
        else:
            print(f"Skipping {folder}, missing frames or labels.")
    
    print("Preprocessing completed.")
    return all_data


# 执行数据预处理
data = preprocess_data(frames_folder, labels_folder)
print(f'Processed {len(data)} sequences.')
