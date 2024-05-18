import os

frames_folder = '/Users/dong/Desktop/Penn_Action/frames'
labels_folder = '/Users/dong/Desktop/Penn_Action/labels'

# 检查 frames 文件夹是否存在
if os.path.exists(frames_folder) and os.path.isdir(frames_folder):
    print(f"Frames folder found: {frames_folder}")
else:
    print(f"Frames folder not found: {frames_folder}")

# 检查 labels 文件夹是否存在
if os.path.exists(labels_folder) and os.path.isdir(labels_folder):
    print(f"Labels folder found: {labels_folder}")
else:
    print(f"Labels folder not found: {labels_folder}")
